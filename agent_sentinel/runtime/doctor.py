"""
``agentsentinel-runtime doctor``: prove the runtime's config is safe to run
and that the API key is not readable from the model's workspace.

Output contract: one line per check, ``PASS <check-id> <message>``,
``FAIL <check-id> <message>`` or ``SKIP <check-id> <message>``, on stdout.
Exit 0 only if no check failed (a SKIP does not fail the run), otherwise 1.
A FAIL message is the ConfigError text or names a path relative to
``workspace`` and a reason; PASS and SKIP messages are fixed strings. None
ever contains the key or the content of a file. As defence in depth every
line is printed without control characters, and a message holding any 8
characters of the key in a row is withheld.

Once the key has been read, the secret-isolation checks run:

- ``key-file-outside-workspace``: the key file's realpath is outside
  ``workspace`` (re-checked on top of ``key-file-location``).
- ``runtime-user-separate``: ``workspace`` is not owned by, and not writable
  by, the runtime's user (owner, group or other bits), and no entry under it
  is owned by the runtime's user (a file the model wrote would be).
- ``key-not-in-workspace``: no regular file under ``workspace`` holds the
  key's bytes, no name holds the key, and no link there leads to the key
  file or to a directory containing it. Directories are walked, never
  followed; a symlink to a regular file outside ``workspace`` that the runtime
  can read is scanned too (the model can read through it), and links to
  anything else outside are counted in the PASS message. Only the top-level
  ``.git/objects`` is skipped (compressed, so a byte scan proves nothing).
  Files of any size are scanned. The walk is bounded: more than 50,000
  entries (files and directories), more than 2 GiB read or more than 120 s is
  a FAIL naming the limit (the time budget is checked between reads, so one
  hung filesystem call can run past it), as is any file or directory that cannot be read:
  the check never passes without proof. Encoded copies of the key (base64,
  UTF-16, compressed archives) are not detected.
- ``key-not-in-harness-config``: the harness MCP configs under ``workspace``
  do not hold the key. Configs outside ``workspace`` (``~/.cursor/mcp.json``
  for example) are not checked, and the PASS message says so.
- ``key-not-in-environment``: ``AGENT_SENTINEL_API_KEY`` is unset and no
  value holds the key, in doctor's own environment only.
- ``upstream-reachable``: ``tools/list`` through the proxy to ``url``
  succeeds. ``--offline`` skips it, and it is not attempted after any FAIL,
  so the key is never sent to a ``url`` from a config that was not proven.

What doctor cannot prove. The file checks hold only if the model runs as a
different user from the runtime: a model running as the runtime's user can
read ``api_key_file`` wherever it is, so ``runtime-user-separate`` fails when
the workspace is owned by the runtime's user, and its PASS still states the
limit, because the harness's user cannot be observed from here. Every check
is a snapshot: a key written into the workspace after doctor ran is not seen.

Key file layout: the runtime's own user must own ``api_key_file`` with mode
600 (no group or other access). Keep it under that user's home, for example
``~/.config/agentsentinel/key``. If you use ``/etc/agentsentinel/key``, chown
it to the runtime's user: a root-owned key file fails ``key-file-permissions``.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

from . import config as runtime_config
from .config import ConfigError

if TYPE_CHECKING:
    import httpx

_MAX_SCAN_FILE_BYTES = 5 * 1024 * 1024  # harness configs only
_MAX_SCAN_FILES = 50_000  # entries: files and directories
_MAX_SCAN_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SCAN_SECONDS = 120.0
_READ_CHUNK = 64 * 1024
_UPSTREAM_TIMEOUT_SECONDS = 30
_KEY_WINDOW = 8

HARNESS_CONFIGS = (
    ".cursor/mcp.json",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".vscode/mcp.json",
)

_WITHHELD = "(message withheld: it contained part of the API key)"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    ok: bool
    message: str
    skipped: bool = False


def _printable(text: str) -> str:
    """Printable ASCII only: one line per check, whatever a file is named."""
    out = []
    for char in text:
        code = ord(char)
        if 0x20 <= code < 0x7F:
            out.append(char)
        elif code < 0x100:
            out.append(f"\\x{code:02x}")
        else:
            out.append(f"\\u{code:04x}")
    return "".join(out)


def _holds_key_part(text: str, api_key: str) -> bool:
    if not api_key:
        return False
    last = max(1, len(api_key) - _KEY_WINDOW + 1)
    return any(api_key[i : i + _KEY_WINDOW] in text for i in range(last))


def format_line(result: CheckResult, api_key: Optional[str] = None) -> str:
    status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
    message = _printable(result.message)
    if api_key is not None and _holds_key_part(message, api_key):
        message = _WITHHELD
    return f"{status} {result.check_id} {message}"


def _check(
    results: list[CheckResult],
    check_id: str,
    passed: str,
    func: Callable[..., Any],
    *args: Any,
) -> tuple[bool, Any]:
    try:
        value = func(*args)
    except ConfigError as exc:
        results.append(CheckResult(check_id, False, str(exc)))
        return False, None
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or a value
        results.append(
            CheckResult(check_id, False, f"unexpected error ({type(exc).__name__})")
        )
        return False, None
    results.append(CheckResult(check_id, True, passed))
    return True, value


def _check_key_realpath(workspace: str, api_key_file: str) -> None:
    runtime_config.check_key_location(workspace, os.path.realpath(api_key_file))


# --- runtime-user-separate ---------------------------------------------------


def _runtime_uid() -> int:
    """The runtime's uid (a seam for tests)."""
    return os.getuid()


def _workspace_owner(workspace: str) -> int:
    """The uid owning ``workspace`` (a seam for tests)."""
    return os.stat(workspace).st_uid


def _runtime_can_write(workspace: str) -> bool:
    """Whether the runtime's user can write ``workspace`` (a seam for tests).

    Effective access, plus the mode bits: world-writable (a root-owned 1777
    ``/workspace`` included), or group-writable by a group the runtime is in.
    """
    st = os.stat(workspace)
    if os.access(workspace, os.W_OK) or st.st_mode & stat.S_IWOTH:
        return True
    groups = set(os.getgroups()) | {os.getegid()}
    return bool(st.st_mode & stat.S_IWGRP) and st.st_gid in groups


_SAME_USER = (
    "a model running as this user can read api_key_file wherever it is; run "
    "the harness as a different user from the runtime"
)


def _check_runtime_user(workspace: str, runtime_owned: int) -> None:
    try:
        owner = _workspace_owner(workspace)
        writable = _runtime_can_write(workspace)
    except (OSError, ValueError):
        raise ConfigError(
            "cannot stat workspace; cannot tell whether the model runs as the "
            "runtime's user"
        ) from None
    if _runtime_uid() == 0:
        # Root bypasses file permissions: os.access is always true for it, so
        # the writable test below would give a misleading reason.
        raise ConfigError(
            "the runtime runs as root; root bypasses file permissions, so doctor "
            "cannot prove the model is separated; run the runtime as a dedicated "
            "non-root user that owns api_key_file"
        )
    if owner == _runtime_uid():
        raise ConfigError(f"workspace is owned by the runtime's user; {_SAME_USER}")
    if writable:
        raise ConfigError(
            "workspace is writable by the runtime's user (group or other "
            f"permissions); {_SAME_USER}"
        )
    if runtime_owned:
        raise ConfigError(
            f"{runtime_owned} entries under workspace are owned by the runtime's "
            f"user, so the model may run as it; {_SAME_USER}"
        )


# --- the workspace scan ------------------------------------------------------


@dataclass
class _Scan:
    entries: int = 0
    files: int = 0
    bytes_read: int = 0
    deadline: float = 0.0
    unreadable: int = 0
    runtime_owned: int = 0
    links_unchecked: int = 0
    outside_seen: set = field(default_factory=set)
    limit: Optional[str] = None
    findings: list[str] = field(default_factory=list)


class _LimitReachedError(Exception):
    pass


def _key_identities(api_key_file: str) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    """The key file's ``(st_dev, st_ino)`` and those of every directory above it."""
    real = os.path.realpath(api_key_file)
    st = os.stat(real)
    ancestors: set[tuple[int, int]] = set()
    current = os.path.dirname(real)
    while True:
        try:
            parent_st = os.stat(current)
            ancestors.add((parent_st.st_dev, parent_st.st_ino))
        except OSError:
            pass  # an unreachable ancestor cannot be a link target either
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return (st.st_dev, st.st_ino), ancestors


def _search_file(
    path: str,
    needle: bytes,
    *,
    follow: bool,
    max_bytes: Optional[int] = None,
    scan: Optional[_Scan] = None,
) -> str:
    """``"match"``, ``"clean"`` or ``"large"``; OSError if it cannot be read.

    Opened non-blocking and checked to be a regular file on the descriptor,
    so a FIFO or device cannot hang doctor. Read in chunks that overlap by
    ``len(needle) - 1`` bytes, so a key across a chunk boundary is found.
    ``max_bytes`` refuses a larger file ("large"). With ``scan``, every byte
    counts against the scan's byte and time budget, and exhausting either
    raises ``_LimitReachedError``.
    """
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY
    if not follow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError("not a regular file")
        if max_bytes is not None and st.st_size > max_bytes:
            return "large"
        keep = len(needle) - 1
        tail = b""
        remaining = max_bytes
        while remaining is None or remaining > 0:
            want = _READ_CHUNK if remaining is None else min(_READ_CHUNK, remaining)
            if scan is not None:
                if scan.bytes_read >= _MAX_SCAN_BYTES:
                    raise _LimitReachedError("bytes")
                if time.monotonic() > scan.deadline:
                    raise _LimitReachedError("time")
                want = min(want, _MAX_SCAN_BYTES - scan.bytes_read)
            chunk = os.read(fd, want)
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            if scan is not None:
                scan.bytes_read += len(chunk)
            window = tail + chunk
            if needle in window:
                return "match"
            tail = window[-keep:] if keep else b""
        return "clean"
    finally:
        os.close(fd)


def _outside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) != root
    except ValueError:
        return True


def _check_link(
    entry: Any, rel: str, scan: _Scan, needle: bytes, key_id, ancestors, ws_real: str
) -> None:
    """A symlink: what it exposes, scanned if it leads to a readable file outside.

    A dangling link exposes nothing. A link into ``workspace`` needs no
    scan: its target is walked anyway.
    """
    try:
        st = os.stat(entry.path)
    except (OSError, ValueError):
        return
    target = (st.st_dev, st.st_ino)
    if target == key_id:
        scan.findings.append(f"{rel} is a symlink to api_key_file")
        return
    if target in ancestors:
        scan.findings.append(
            f"{rel} is a symlink to a directory that contains api_key_file"
        )
        return
    if not _outside(os.path.realpath(entry.path), ws_real):
        return
    if not stat.S_ISREG(st.st_mode):
        scan.links_unchecked += 1
        return
    if target in scan.outside_seen:
        return  # several links to one file: read it once, not once per link
    scan.outside_seen.add(target)
    try:
        found = _search_file(entry.path, needle, follow=True, scan=scan)
    except OSError:
        # The model may be able to read what the runtime cannot: unproven.
        scan.findings.append(
            f"{rel} is a symlink to a file outside workspace that doctor could not read"
        )
        return
    if found == "match":
        scan.findings.append(
            f"{rel} is a symlink to a file outside workspace that contains the API key"
        )


def _scan_workspace(
    workspace: str, api_key: str, key_id: tuple[int, int], ancestors: set
) -> _Scan:
    scan = _Scan(deadline=time.monotonic() + _MAX_SCAN_SECONDS)
    needle = api_key.encode("ascii")
    ws_real = os.path.realpath(workspace)
    top_git = os.path.join(workspace, ".git")
    runtime_uid = _runtime_uid()
    named = False
    stack = [workspace]
    try:
        while stack:
            directory = stack.pop()
            try:
                iterator = os.scandir(directory)
            except OSError:
                scan.unreadable += 1
                continue
            with iterator:
                for entry in iterator:  # lazily: a huge directory is never listed
                    _scan_entry(
                        entry,
                        directory,
                        top_git,
                        workspace,
                        scan,
                        needle,
                        key_id,
                        ancestors,
                        ws_real,
                        runtime_uid,
                        stack,
                    )
                    if not named and api_key in entry.name:
                        named = True
                        # The path itself holds the key, so it is not named.
                        scan.findings.append(
                            "a file or directory name under workspace contains "
                            "the API key"
                        )
    except _LimitReachedError as exc:
        scan.limit = str(exc)
    return scan


def _scan_entry(
    entry: Any,
    directory: str,
    top_git: str,
    workspace: str,
    scan: _Scan,
    needle: bytes,
    key_id,
    ancestors,
    ws_real: str,
    runtime_uid: int,
    stack: list,
) -> None:
    scan.entries += 1
    if scan.entries > _MAX_SCAN_FILES:
        raise _LimitReachedError("entries")
    if time.monotonic() > scan.deadline:
        raise _LimitReachedError("time")
    rel = os.path.relpath(entry.path, workspace)
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        scan.unreadable += 1
        return
    if st.st_uid == runtime_uid:
        scan.runtime_owned += 1
    if stat.S_ISDIR(st.st_mode):
        if entry.name == "objects" and directory == top_git:
            return
        stack.append(entry.path)
        return
    if stat.S_ISLNK(st.st_mode):
        _check_link(entry, rel, scan, needle, key_id, ancestors, ws_real)
        return
    if not stat.S_ISREG(st.st_mode):
        return  # a FIFO, socket or device holds no stored bytes
    scan.files += 1
    try:
        found = _search_file(entry.path, needle, follow=False, scan=scan)
    except OSError:
        scan.unreadable += 1
        return
    if found == "match":
        scan.findings.append(f"{rel} contains the API key")


class _ScanFailedError(Exception):
    pass


def _summarize(findings: list[str]) -> str:
    if len(findings) == 1:
        return findings[0]
    return f"{findings[0]} ({len(findings)} findings in total)"


def _limit_text(limit: str) -> str:
    if limit == "entries":
        return f"more than {_MAX_SCAN_FILES} files and directories"
    if limit == "bytes":
        return f"more than {_MAX_SCAN_BYTES} bytes to read"
    return f"more than {_MAX_SCAN_SECONDS:g} s to scan, checked between reads"


def _workspace_scan(workspace: str, api_key_file: str, api_key: str) -> _Scan:
    try:
        key_id, ancestors = _key_identities(api_key_file)
    except (OSError, ValueError):
        raise ConfigError(
            "cannot stat api_key_file; cannot check the workspace for links to it"
        ) from None
    return _scan_workspace(workspace, api_key, key_id, ancestors)


def _check_workspace_scan(scan: _Scan) -> str:
    if scan.findings:
        raise ConfigError(_summarize(scan.findings))
    if scan.limit is not None:
        raise ConfigError(
            f"workspace too large to prove ({_limit_text(scan.limit)}); narrow workspace"
        )
    if scan.unreadable:
        raise ConfigError(
            f"{scan.unreadable} files or directories under workspace could not "
            "be read; cannot prove the key is absent"
        )
    return (
        f"no file under workspace contains the API key ({scan.files} files "
        f"checked, {scan.links_unchecked} links out of workspace to anything but "
        "a readable file not followed, top-level .git/objects not scanned; "
        "encoded copies of the key are not detected)"
    )


def _check_harness_configs(workspace: str, api_key_file: str, api_key: str) -> None:
    try:
        key_id, _ = _key_identities(api_key_file)
    except (OSError, ValueError):
        raise ConfigError(
            "cannot stat api_key_file; cannot check the harness configs"
        ) from None
    needle = api_key.encode("ascii")
    findings = []
    for rel in HARNESS_CONFIGS:
        path = os.path.join(workspace, rel)
        try:
            os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            findings.append(f"{rel} cannot be read")
            continue
        # The harness follows a symlink here, so the check does too.
        try:
            st = os.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            continue  # a dangling symlink: the harness reads nothing
        except OSError:
            findings.append(f"{rel} cannot be read")
            continue
        if (st.st_dev, st.st_ino) == key_id:
            findings.append(f"{rel} leads to api_key_file")
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        try:
            found = _search_file(
                path, needle, follow=True, max_bytes=_MAX_SCAN_FILE_BYTES
            )
        except OSError:
            findings.append(f"{rel} cannot be read")
            continue
        if found == "large":
            findings.append(f"{rel} is larger than 5 MiB and cannot be checked")
        elif found == "match":
            findings.append(
                f"{rel} contains the API key; remove it, the runtime reads the "
                "key from api_key_file"
            )
    if findings:
        raise ConfigError(_summarize(findings))


def _check_environment(environ: Mapping[str, str], api_key: str) -> None:
    problems = []
    if "AGENT_SENTINEL_API_KEY" in environ:
        problems.append(
            "AGENT_SENTINEL_API_KEY is set; unset it, the runtime reads the key "
            "from api_key_file"
        )
    holders = sorted(
        name
        for name, value in environ.items()
        if isinstance(value, str) and api_key in value
    )
    if holders:
        problems.append(f"environment variable {', '.join(holders)} holds the API key")
    if problems:
        raise ConfigError("; ".join(problems))


# --- upstream-reachable ------------------------------------------------------

_HTTP_DETAIL = re.compile(r"\(HTTP (\d{3})(?:, ([a-z0-9_]{1,64}))?\)")
_UNREACHABLE = re.compile(r"unreachable \(([A-Za-z_][A-Za-z0-9_]{0,63})\)")


def _find_exception(exc: BaseException, kind: type) -> Optional[BaseException]:
    if isinstance(exc, kind):
        return exc
    # anyio task groups wrap failures in (Base)ExceptionGroup; duck-typed so
    # Python 3.10 (the exceptiongroup backport) is covered too.
    inner = getattr(exc, "exceptions", None)
    if isinstance(inner, tuple):
        for item in inner:
            found = _find_exception(item, kind)
            if found is not None:
                return found
    return None


async def _list_tools(
    proxy_config: Any, http_transport: Optional[httpx.AsyncBaseTransport]
) -> None:
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from .. import mcp_proxy

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                partial(
                    mcp_proxy.serve,
                    proxy_config,
                    *server_streams,
                    http_transport=http_transport,
                )
            )
            try:
                with anyio.fail_after(_UPSTREAM_TIMEOUT_SECONDS):
                    async with ClientSession(*client_streams) as session:
                        await session.initialize()
                        await session.list_tools()
            finally:
                task_group.cancel_scope.cancel()


def _upstream_failure(exc: BaseException) -> str:
    """Only a status, a validated reason code or an exception type, never text."""
    from mcp.shared.exceptions import McpError

    if _find_exception(exc, TimeoutError) is not None:
        return (
            f"tools/list through the proxy timed out after {_UPSTREAM_TIMEOUT_SECONDS}s"
        )
    found = _find_exception(exc, McpError)
    if found is not None:
        error = found.error
        message = error.message if isinstance(error.message, str) else ""
        http = _HTTP_DETAIL.search(message)
        if http:
            detail = f"HTTP {http.group(1)}"
            if http.group(2):
                detail += f", {http.group(2)}"
            return f"tools/list through the proxy failed ({detail})"
        unreachable = _UNREACHABLE.search(message)
        if unreachable:
            return f"url is unreachable ({unreachable.group(1)})"
        return f"tools/list through the proxy failed (MCP error {int(error.code)})"
    return f"tools/list through the proxy failed ({type(exc).__name__})"


def _check_upstream(
    proxy_config: Any, http_transport: Optional[httpx.AsyncBaseTransport]
) -> CheckResult:
    import anyio

    try:
        anyio.run(_list_tools, proxy_config, http_transport)
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or a value
        return CheckResult("upstream-reachable", False, _upstream_failure(exc))
    return CheckResult(
        "upstream-reachable", True, "tools/list through the proxy to url succeeded"
    )


# --- the run -----------------------------------------------------------------


def _run(
    path: str,
    *,
    offline: bool,
    environ: Optional[Mapping[str, str]],
    http_transport: Optional[httpx.AsyncBaseTransport],
) -> tuple[list[CheckResult], Optional[str]]:
    results: list[CheckResult] = []
    ok, doc = _check(
        results,
        "config-file",
        "config file is a regular file owned by the current user (not writable "
        "by group or others, no other hard links) and a valid YAML mapping with "
        "only known keys",
        runtime_config.read_document,
        path,
    )
    if not ok:
        return results, None

    ok, paths = _check(
        results,
        "config-paths",
        "workspace and api_key_file are absolute paths",
        runtime_config.check_paths,
        doc,
    )
    api_key = None
    if ok:
        workspace, api_key_file = paths
        _check(
            results,
            "config-file-location",
            "config file is outside workspace",
            runtime_config.check_config_location,
            doc,
            workspace,
        )
        ok, _ = _check(
            results,
            "key-file-location",
            "api_key_file is outside workspace",
            runtime_config.check_key_location,
            workspace,
            api_key_file,
        )
        # A key file inside the workspace is not read at all.
        if ok:
            ok, raw = _check(
                results,
                "key-file-permissions",
                "api_key_file is a regular file owned by the current user (no "
                "group or other access, no other hard links)",
                runtime_config.read_key_file,
                workspace,
                api_key_file,
            )
        if ok:
            ok, api_key = _check(
                results,
                "key-file-content",
                "api_key_file holds a key that can be sent in an HTTP header",
                runtime_config.parse_key,
                raw,
            )

    _, url = _check(
        results, "config-url", "url is valid", runtime_config.check_url, doc
    )
    _, harness = _check(
        results,
        "config-harness",
        "harness is valid",
        runtime_config.check_harness,
        doc,
    )
    _, approvals = _check(
        results,
        "config-approvals",
        "approvals settings are valid",
        runtime_config.check_approvals,
        doc,
    )
    if api_key is None:
        # Nothing below can be proven without the key; the run already fails.
        return results, None

    _check(
        results,
        "config-no-key",
        "no displayed setting contains the API key",
        runtime_config.check_key_not_in_settings,
        doc,
        api_key,
    )
    _isolation_checks(
        results,
        workspace,
        api_key_file,
        api_key,
        os.environ if environ is None else environ,
    )

    if offline:
        results.append(
            CheckResult("upstream-reachable", True, "skipped (--offline)", skipped=True)
        )
    elif not all(r.ok for r in results):
        # A config that failed a check may point url anywhere: never send the key.
        results.append(
            CheckResult(
                "upstream-reachable",
                True,
                "not attempted: an earlier check failed",
                skipped=True,
            )
        )
    else:
        cfg = runtime_config.RuntimeConfig(
            url=url,
            workspace=workspace,
            api_key_file=api_key_file,
            harness=harness,
            approvals=approvals,
            api_key=api_key,
        )
        results.append(_check_upstream(cfg.proxy_config(), http_transport))
    return results, api_key


def _isolation_checks(
    results: list[CheckResult],
    workspace: str,
    api_key_file: str,
    api_key: str,
    environ: Mapping[str, str],
) -> None:
    _check(
        results,
        "key-file-outside-workspace",
        "api_key_file resolves (realpath) outside workspace",
        _check_key_realpath,
        workspace,
        api_key_file,
    )
    # The walk runs once: runtime-user-separate uses its ownership count and
    # key-not-in-workspace its findings.
    scan: Optional[_Scan] = None
    scan_error: Optional[CheckResult] = None
    try:
        scan = _workspace_scan(workspace, api_key_file, api_key)
    except ConfigError as exc:
        scan_error = CheckResult("key-not-in-workspace", False, str(exc))
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or a value
        scan_error = CheckResult(
            "key-not-in-workspace", False, f"unexpected error ({type(exc).__name__})"
        )
    _check(
        results,
        "runtime-user-separate",
        "workspace is owned by another user and not writable by the runtime's "
        "user, and no entry under it is the runtime's; doctor cannot prove the "
        "model never runs as the runtime's user, and the file checks hold only if "
        "it does not",
        _check_runtime_user,
        workspace,
        scan.runtime_owned if scan is not None else 0,
    )
    # The PASS message carries the scan's counts, so it is built here.
    try:
        if scan_error is not None:
            raise _ScanFailedError
        assert scan is not None
        passed = _check_workspace_scan(scan)
    except _ScanFailedError:
        results.append(scan_error)
    except ConfigError as exc:
        results.append(CheckResult("key-not-in-workspace", False, str(exc)))
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or a value
        results.append(
            CheckResult(
                "key-not-in-workspace",
                False,
                f"unexpected error ({type(exc).__name__})",
            )
        )
    else:
        results.append(CheckResult("key-not-in-workspace", True, passed))
    _check(
        results,
        "key-not-in-harness-config",
        f"none of {', '.join(HARNESS_CONFIGS)} under workspace contains the API "
        "key (harness configs outside workspace are not checked)",
        _check_harness_configs,
        workspace,
        api_key_file,
        api_key,
    )
    _check(
        results,
        "key-not-in-environment",
        "AGENT_SENTINEL_API_KEY is unset and no environment value holds the key "
        "(doctor's environment only; the harness's may differ)",
        _check_environment,
        environ,
        api_key,
    )


def run_checks(
    path: str,
    *,
    offline: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    http_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[CheckResult]:
    """The checks' results; a message holding part of the key is withheld."""
    results, api_key = _run(
        path, offline=offline, environ=environ, http_transport=http_transport
    )
    if not api_key:
        return results
    return [
        replace(r, message=_WITHHELD)
        if _holds_key_part(r.message, api_key)
        or _holds_key_part(_printable(r.message), api_key)
        else r
        for r in results
    ]


def main(
    path: str,
    out: Optional[Any] = None,
    *,
    offline: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    http_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> int:
    """Print one line per check; 0 if no check failed, else 1."""
    stream = out if out is not None else sys.stdout
    results, api_key = _run(
        path, offline=offline, environ=environ, http_transport=http_transport
    )
    for result in results:
        print(format_line(result, api_key), file=stream)
    return 0 if all(r.ok for r in results) else 1
