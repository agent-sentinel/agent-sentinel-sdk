"""
Load and validate the runtime's single config file (``runtime.yaml``).

::

    url: https://platform.example.com    # optional; defaults to the hosted platform
    workspace: /home/agent/work          # required; absolute; existing directory
    api_key_file: /etc/agentsentinel/key # required; absolute; outside workspace;
                                         # owned by the runtime's user, mode 600
                                         # (chown it: a root-owned key is refused)
    harness: cursor                      # optional
    approvals:                           # optional
      wait: true
      timeout_seconds: 900
      poll_interval_seconds: 5

The config file itself gets the key file's rules: a regular file owned by the
current user, not writable by group or others, with no other hard links, and
outside ``workspace``. Otherwise a model that can edit it could point ``url``
at another host and have the key sent there.

Every failure is a one-line ``ConfigError`` naming only fields and rules. No
message echoes a value from the config file or the key file, so a key pasted
into the wrong place cannot leak. Each rule is its own function so
``doctor`` can report them one by one; ``load_runtime_config`` runs them all.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .. import mcp_proxy
from ..mcp_proxy import ConfigError

__all__ = [
    "CONFIG_ENV_VAR",
    "ApprovalsConfig",
    "ConfigError",
    "RuntimeConfig",
    "ConfigDocument",
    "check_approvals",
    "check_config_location",
    "check_harness",
    "check_key_location",
    "check_key_not_in_settings",
    "check_paths",
    "check_url",
    "load_runtime_config",
    "parse_key",
    "read_document",
    "read_key_file",
    "resolve_config_path",
]

# Holds a path to the config file, never a secret.
CONFIG_ENV_VAR = "AGENT_SENTINEL_RUNTIME_CONFIG"
DEFAULT_CONFIG_PATH = "~/.config/agentsentinel/runtime.yaml"

_ALLOWED_KEYS = ("url", "workspace", "api_key_file", "harness", "approvals")
_APPROVAL_KEYS = ("wait", "timeout_seconds", "poll_interval_seconds")
_DISPLAYED_KEYS = ("url", "workspace", "api_key_file", "harness")
_MAX_KEY_FILE_BYTES = 4096
_MAX_CONFIG_FILE_BYTES = 64 * 1024


@dataclass(frozen=True)
class ApprovalsConfig:
    """Approval-wait settings. Read by the HITL wait (t28b); stored here."""

    wait: bool = True
    timeout_seconds: int = 900
    poll_interval_seconds: int = 5


@dataclass(frozen=True)
class RuntimeConfig:
    """A validated runtime config. ``api_key`` is excluded from ``repr``."""

    url: str
    workspace: str
    api_key_file: str
    harness: Optional[str]
    approvals: ApprovalsConfig
    api_key: str = field(repr=False)

    def proxy_config(self) -> mcp_proxy.ProxyConfig:
        return mcp_proxy.ProxyConfig(
            url=f"{self.url}/mcp", api_key=self.api_key, harness=self.harness
        )

    def display_dict(self) -> dict[str, Any]:
        """The resolved config for display. Never contains the key."""
        return {
            "url": self.url,
            "workspace": self.workspace,
            "api_key_file": self.api_key_file,
            "harness": self.harness,
            "approvals": {
                "wait": self.approvals.wait,
                "timeout_seconds": self.approvals.timeout_seconds,
                "poll_interval_seconds": self.approvals.poll_interval_seconds,
            },
        }


def resolve_config_path(arg: Optional[str], environ: Mapping[str, str]) -> str:
    """``--config``, else ``$AGENT_SENTINEL_RUNTIME_CONFIG``, else the default."""
    if arg:
        return arg
    from_env = environ.get(CONFIG_ENV_VAR, "").strip()
    if from_env:
        return from_env
    return os.path.expanduser(DEFAULT_CONFIG_PATH)


class ConfigDocument(dict):
    """The parsed config, plus the identity of the file it was read from.

    ``location`` is the path the file was actually opened at (resolved while
    it was open), ``spelled`` the absolute path as given, and ``identity`` its
    ``(st_dev, st_ino)``, so ``check_config_location`` checks the file that was
    read, not whatever the path names later, and refuses a symlink inside the
    workspace that points out of it.
    """

    location: str = ""
    spelled: str = ""
    identity: tuple[int, int] = (0, 0)


def _opened_location(fd: int, path: str, st: os.stat_result) -> Optional[str]:
    """The resolved path ``fd`` was opened at, or None if it no longer matches."""
    try:
        opened = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        opened = ""
    try:
        if not os.path.isabs(opened):
            opened = os.path.realpath(path)
        now = os.stat(opened)
    except (OSError, ValueError):
        return None
    if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
        return None
    return opened


def _check_config_file_stat(st: os.stat_result) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError("config file must be a regular file")
    if st.st_uid != os.getuid():
        raise ConfigError("config file must be owned by the current user")
    if st.st_mode & 0o022:
        raise ConfigError(
            "config file must not be writable by group or others (chmod 600)"
        )
    if st.st_nlink != 1:
        raise ConfigError("config file must not have other hard links")


def _read_config_file(path: str) -> tuple[bytes, str, tuple[int, int]]:
    """Check the config file's type, owner, mode and links; return its bytes."""
    # The path is never echoed: a key passed by mistake could be the path.
    try:
        st = os.stat(path)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError(f"cannot read config file ({reason})") from None
    except ValueError:
        raise ConfigError("cannot read config file (invalid path)") from None
    # Checked before open() so a FIFO or device cannot block the read.
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError("config file must be a regular file")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError(f"cannot read config file ({reason})") from None
    except ValueError:
        raise ConfigError("cannot read config file (invalid path)") from None
    try:
        st = os.fstat(fd)
        _check_config_file_stat(st)
        location = _opened_location(fd, path, st)
        if location is None:
            raise ConfigError("config file changed while it was being read; run again")
        try:
            raw = os.read(fd, _MAX_CONFIG_FILE_BYTES + 1)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise ConfigError(f"cannot read config file ({reason})") from None
        # Re-check after the read: a chmod/chown or a new hard link while the
        # bytes were being read means they cannot be trusted.
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino) != (st.st_dev, st.st_ino):
            raise ConfigError("config file changed while it was being read; run again")
        _check_config_file_stat(after)
    finally:
        os.close(fd)
    if len(raw) > _MAX_CONFIG_FILE_BYTES:
        raise ConfigError(f"config file is larger than {_MAX_CONFIG_FILE_BYTES} bytes")
    return raw, location, (st.st_dev, st.st_ino)


def read_document(path: str) -> ConfigDocument:
    """Check and parse the config file, then check its top-level keys."""
    raw, location, identity = _read_config_file(path)
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        # str(exc) quotes the offending source line, which could hold a key.
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ConfigError(f"config file is not valid YAML{where}") from None
    except Exception:  # noqa: BLE001 - e.g. ValueError for a bad timestamp
        raise ConfigError("config file is not valid YAML") from None
    if not isinstance(doc, dict):
        raise ConfigError("config file must be a YAML mapping of settings")
    if "api_key" in doc:
        raise ConfigError(
            "api_key is not allowed in the config file; "
            "put the key in a file and set api_key_file"
        )
    if any(key not in _ALLOWED_KEYS for key in doc):
        # Key names are not echoed: a pasted key could be the name itself.
        raise ConfigError(
            "config file has an unknown key; allowed keys are "
            + ", ".join(_ALLOWED_KEYS)
        )
    for required in ("workspace", "api_key_file"):
        if doc.get(required) is None:
            raise ConfigError(f"{required} is required")
    result = ConfigDocument(doc)
    result.location = location
    try:
        result.spelled = os.path.abspath(path)
    except (OSError, ValueError):
        result.spelled = ""
    result.identity = identity
    return result


def check_paths(doc: Mapping[str, Any]) -> tuple[str, str]:
    """``workspace`` and ``api_key_file`` must be absolute path strings."""
    paths = []
    for name in ("workspace", "api_key_file"):
        value = doc.get(name)
        if not isinstance(value, str):
            raise ConfigError(f"{name} must be a string")
        if "\0" in value:
            raise ConfigError(f"{name} must not contain a NUL character")
        if not os.path.isabs(value):
            raise ConfigError(f"{name} must be an absolute path")
        paths.append(value)
    return paths[0], paths[1]


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _workspace_stat(workspace: str) -> os.stat_result:
    try:
        st = os.stat(workspace)
    except (OSError, ValueError):
        raise ConfigError("workspace must be an existing directory") from None
    if not stat.S_ISDIR(st.st_mode):
        raise ConfigError("workspace must be an existing directory")
    return st


def _has_ancestor(path: str, root: os.stat_result) -> bool:
    """Whether any directory on ``path``'s way up is the ``root`` directory.

    Compared by ``(st_dev, st_ino)``, not by spelling, so a case-insensitive
    file system (macOS by default) or a symlinked parent can't hide it.
    """
    current = os.path.dirname(path)
    while True:
        try:
            st = os.stat(current)
        except (OSError, ValueError):
            st = None
        if st is not None and (st.st_dev, st.st_ino) == (root.st_dev, root.st_ino):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _inside_workspace(workspace: str, path: str) -> bool:
    ws = _workspace_stat(workspace)
    try:
        roots = {os.path.normpath(workspace), os.path.realpath(workspace)}
        paths = {os.path.normpath(path), os.path.realpath(path)}
    except (OSError, ValueError):
        return True
    if any(_is_within(p, root) for p in paths for root in roots):
        return True
    return any(_has_ancestor(p, ws) for p in paths)


def check_key_location(workspace: str, api_key_file: str) -> None:
    """The key file must not be reachable inside the workspace.

    The literal and the symlink-resolved paths are both compared, by spelling
    and by directory identity, so a key file path that passes through the
    workspace, a symlink outside it that points in, or a differently cased
    spelling of it are all refused.
    """
    if _inside_workspace(workspace, api_key_file):
        raise ConfigError("api_key_file must not be inside workspace")


def check_config_location(doc: ConfigDocument, workspace: str) -> None:
    """The config file that was read must not be inside the workspace."""
    if (
        not doc.location
        or not doc.spelled
        or _inside_workspace(workspace, doc.location)
        or _inside_workspace(workspace, doc.spelled)
    ):
        raise ConfigError("config file must not be inside workspace")
    try:
        now = os.stat(doc.location)
    except (OSError, ValueError):
        raise ConfigError(
            "config file changed while it was being read; run again"
        ) from None
    if (now.st_dev, now.st_ino) != doc.identity:
        raise ConfigError("config file changed while it was being read; run again")


KEY_OWNER_GUIDANCE = (
    "the runtime's own user must own it with mode 600: keep it under that "
    "user's home (for example ~/.config/agentsentinel/key), or chown "
    "/etc/agentsentinel/key to that user and chmod 600 it; a root-owned key "
    "file fails this check"
)

_SWAPPED = "api_key_file changed while it was being read; run again"


def _check_opened_location(
    fd: int, st: os.stat_result, workspace: str, api_key_file: str
) -> None:
    """Bind the opened descriptor to a location outside ``workspace``.

    The path the descriptor was actually opened at (``/proc/self/fd`` where
    available, else the path resolved again now) must be outside the
    workspace and still name the same file, so a symlink retargeted between
    the location check and ``open()`` is refused.
    """
    try:
        opened = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        opened = ""
    try:
        if not os.path.isabs(opened):
            opened = os.path.realpath(api_key_file)
        check_key_location(workspace, opened)
        now = os.stat(opened, follow_symlinks=False)
    except (OSError, ValueError):
        raise ConfigError(_SWAPPED) from None
    if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
        raise ConfigError(_SWAPPED)


def read_key_file(workspace: str, api_key_file: str) -> bytes:
    """Check the key file's location, type, owner and mode; return its bytes.

    The checks are repeated on the opened descriptor, and the descriptor is
    bound to a location outside ``workspace``, so the file cannot be swapped
    between the check and the read.
    """
    check_key_location(workspace, api_key_file)
    try:
        st = os.stat(api_key_file)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError(f"cannot read api_key_file ({reason})") from None
    except ValueError:
        raise ConfigError("cannot read api_key_file (invalid path)") from None
    # Checked before open() so a FIFO or device cannot block the read.
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError("api_key_file must be a regular file")
    try:
        fd = os.open(api_key_file, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError(f"cannot read api_key_file ({reason})") from None
    except ValueError:
        raise ConfigError("cannot read api_key_file (invalid path)") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfigError("api_key_file must be a regular file")
        if st.st_uid != os.getuid():
            raise ConfigError(
                f"api_key_file must be owned by the current user; {KEY_OWNER_GUIDANCE}"
            )
        if st.st_mode & 0o077:
            raise ConfigError(
                "api_key_file must not be readable or writable by group or "
                "others (chmod 600)"
            )
        # Another hard link could put the same key inside the workspace.
        if st.st_nlink != 1:
            raise ConfigError("api_key_file must not have other hard links")
        _check_opened_location(fd, st, workspace, api_key_file)
        try:
            return os.read(fd, _MAX_KEY_FILE_BYTES + 1)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise ConfigError(f"cannot read api_key_file ({reason})") from None
    finally:
        os.close(fd)


def parse_key(raw: bytes) -> str:
    """The key file holds only the agent-bound key (surrounding space allowed)."""
    if len(raw) > _MAX_KEY_FILE_BYTES:
        raise ConfigError(
            f"api_key_file is larger than {_MAX_KEY_FILE_BYTES} bytes; "
            "it must hold only the API key"
        )
    unsendable = (
        "api_key_file contains characters that cannot be sent in an HTTP header"
    )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise ConfigError(unsendable) from None
    if not text.strip():
        raise ConfigError("api_key_file is empty; it must hold an agent-bound API key")
    return mcp_proxy.check_api_key(text, "api_key_file")


def _optional_str(doc: Mapping[str, Any], name: str) -> str:
    value = doc.get(name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")
    return value


def check_url(doc: Mapping[str, Any]) -> str:
    """The platform base URL; ``mcp_proxy``'s rules apply."""
    return mcp_proxy.check_base_url(_optional_str(doc, "url"), "url")


def check_harness(doc: Mapping[str, Any]) -> Optional[str]:
    # A key pasted here is refused by ``check_key_not_in_settings``, which
    # compares against the real key rather than guessing from its shape.
    return mcp_proxy.check_harness(_optional_str(doc, "harness"), "harness")


def check_key_not_in_settings(doc: Mapping[str, Any], api_key: str) -> None:
    """No displayed setting may contain the key (``config`` prints them)."""
    for name in _DISPLAYED_KEYS:
        value = doc.get(name)
        if isinstance(value, str) and api_key in value:
            raise ConfigError(f"{name} must not contain the API key")


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    # type() rather than isinstance(): YAML true/false are bools, an int subclass.
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{name} must be an integer from {low} to {high}")
    return value


def check_approvals(doc: Mapping[str, Any]) -> ApprovalsConfig:
    section = doc.get("approvals")
    if section is None:
        return ApprovalsConfig()
    if not isinstance(section, dict):
        raise ConfigError("approvals must be a mapping")
    if any(key not in _APPROVAL_KEYS for key in section):
        raise ConfigError(
            "approvals has an unknown key; allowed keys are "
            + ", ".join(_APPROVAL_KEYS)
        )
    defaults = ApprovalsConfig()
    wait = section.get("wait", defaults.wait)
    if not isinstance(wait, bool):
        raise ConfigError("approvals.wait must be true or false")
    timeout = _bounded_int(
        section.get("timeout_seconds", defaults.timeout_seconds),
        "approvals.timeout_seconds",
        1,
        86400,
    )
    interval = _bounded_int(
        section.get("poll_interval_seconds", defaults.poll_interval_seconds),
        "approvals.poll_interval_seconds",
        1,
        300,
    )
    return ApprovalsConfig(
        wait=wait, timeout_seconds=timeout, poll_interval_seconds=interval
    )


def load_runtime_config(path: str) -> RuntimeConfig:
    """Read and fully validate the config file at ``path``.

    Raises ConfigError with a one-line message that never contains the key.
    """
    doc = read_document(path)
    workspace, api_key_file = check_paths(doc)
    check_config_location(doc, workspace)
    api_key = parse_key(read_key_file(workspace, api_key_file))
    cfg = RuntimeConfig(
        url=check_url(doc),
        workspace=workspace,
        api_key_file=api_key_file,
        harness=check_harness(doc),
        approvals=check_approvals(doc),
        api_key=api_key,
    )
    check_key_not_in_settings(doc, api_key)
    return cfg
