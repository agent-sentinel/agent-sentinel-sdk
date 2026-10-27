"""
Console entry point for ``agentsentinel-runtime``.

Subcommands, each taking ``--config PATH`` (else
``$AGENT_SENTINEL_RUNTIME_CONFIG``, else ``~/.config/agentsentinel/runtime.yaml``):

- ``run``: serve the ``agentsentinel-mcp`` stdio proxy with the config.
  Outbound-only: HTTPS to the platform, stdio to the harness, no listener.
  With ``approvals.wait``, a call that needs approval is held, polled and
  retried once approved (see ``approvals.py``).
- ``config``: print the resolved config as YAML (the key is never shown).
- ``doctor``: run the safety checks (see ``doctor.py``).

Errors are one line on stderr, never a traceback, never the key. Exit codes
match ``agentsentinel-mcp``: 2 for configuration, 1 for a runtime failure,
130 on interrupt.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import httpx

    from .config import RuntimeConfig

PROG = "agentsentinel-runtime"

USAGE_ERROR = f"{PROG}: unusable arguments; run '{PROG} --help'"

ENV_KEY_WARNING = (
    f"{PROG}: warning: AGENT_SENTINEL_API_KEY is set, but the harness "
    "environment is model-readable; it is ignored and the key is read from "
    "api_key_file"
)


async def serve(
    runtime_config: RuntimeConfig,
    read_stream: Any,
    write_stream: Any,
    *,
    http_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> None:
    """Run the stdio proxy with ``runtime_config`` over the given MCP streams."""
    from .. import mcp_proxy
    from .approvals import ApprovalWaiter

    proxy_config = runtime_config.proxy_config()
    waiter = None
    if runtime_config.approvals.wait:
        waiter = ApprovalWaiter(
            proxy_config,
            runtime_config.url,
            runtime_config.approvals,
            http_transport=http_transport,
        )
    await mcp_proxy.serve(
        proxy_config,
        read_stream,
        write_stream,
        http_transport=http_transport,
        approval_waiter=waiter,
    )


async def _run_stdio(runtime_config: RuntimeConfig) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await serve(runtime_config, read_stream, write_stream)


class _QuietParser(argparse.ArgumentParser):
    """argparse's errors quote the offending argument, which could be a key
    typed on the command line by mistake. Report a value-free line instead.
    Subparsers inherit this class (argparse uses ``type(self)``)."""

    def error(self, message: str) -> Any:  # noqa: ARG002 - message may hold a value
        print(USAGE_ERROR, file=sys.stderr)
        sys.exit(2)


def _parser() -> argparse.ArgumentParser:
    parser = _QuietParser(prog=PROG, description="AgentSentinel runtime for agent VMs.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True
    for name, help_text in (
        ("run", "serve the governed MCP tools over stdio"),
        ("config", "print the resolved config (never the key)"),
        ("doctor", "check that the config keeps the key out of the model's reach"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", metavar="PATH", help="runtime.yaml path")
        if name == "doctor":
            sub.add_argument(
                "--offline",
                action="store_true",
                help="skip the upstream-reachable check (no network)",
            )
    return parser


def _run(cfg: RuntimeConfig) -> int:
    import anyio

    try:
        anyio.run(_run_stdio, cfg)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or the key
        print(f"{PROG}: stopped ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Console entry point. Unusable arguments exit 2 (argparse)."""
    args = _parser().parse_args(argv)
    try:
        from . import config as runtime_config
        from . import doctor
    except ImportError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2

    path = runtime_config.resolve_config_path(args.config, os.environ)
    if args.command == "doctor":
        return doctor.main(path, offline=args.offline)

    try:
        cfg = runtime_config.load_runtime_config(path)
    except runtime_config.ConfigError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or a value
        print(
            f"{PROG}: cannot load config (unexpected {type(exc).__name__})",
            file=sys.stderr,
        )
        return 2

    if args.command == "config":
        import yaml

        sys.stdout.write(yaml.safe_dump(cfg.display_dict(), sort_keys=False))
        return 0

    if "AGENT_SENTINEL_API_KEY" in os.environ:
        print(ENV_KEY_WARNING, file=sys.stderr)
    return _run(cfg)


if __name__ == "__main__":
    sys.exit(main())
