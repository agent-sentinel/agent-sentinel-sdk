"""
``agentsentinel-runtime doctor``: prove the runtime's config is safe to run.

Output contract: one line per check, ``PASS <check-id> <message>`` or
``FAIL <check-id> <message>``, on stdout. Exit 0 only if every check passes,
otherwise 1. A FAIL message is the ConfigError text; PASS messages are fixed
strings. Neither ever contains the key or the key file's content.

This chunk runs the config-validation checks only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from . import config as runtime_config
from .config import ConfigError


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    ok: bool
    message: str


def format_line(result: CheckResult) -> str:
    return f"{'PASS' if result.ok else 'FAIL'} {result.check_id} {result.message}"


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


def run_checks(path: str) -> list[CheckResult]:
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
        return results

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

    _check(results, "config-url", "url is valid", runtime_config.check_url, doc)
    _check(
        results,
        "config-harness",
        "harness is valid",
        runtime_config.check_harness,
        doc,
    )
    _check(
        results,
        "config-approvals",
        "approvals settings are valid",
        runtime_config.check_approvals,
        doc,
    )
    if api_key is not None:
        _check(
            results,
            "config-no-key",
            "no displayed setting contains the API key",
            runtime_config.check_key_not_in_settings,
            doc,
            api_key,
        )
    return results


def main(path: str, out: Optional[Any] = None) -> int:
    """Print one line per check; 0 if every check passed, else 1."""
    stream = out if out is not None else sys.stdout
    results = run_checks(path)
    for result in results:
        print(format_line(result), file=stream)
    return 0 if all(r.ok for r in results) else 1
