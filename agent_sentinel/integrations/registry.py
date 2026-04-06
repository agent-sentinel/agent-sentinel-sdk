"""
Auto-registration of sentinel-guarded tools with the platform.

Discovers functions decorated with @sentinel_tool or @guarded_action
and batch-registers them with the AgentSentinel platform.
"""
from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_sentinel.integrations.registry")


def discover_tools(modules: Optional[List[ModuleType]] = None) -> List[Dict[str, Any]]:
    """
    Discover all sentinel-guarded tool functions.

    Scans modules for functions with _sentinel_tool_config or _sentinel_guarded
    attributes (set by @sentinel_tool or @guarded_action).

    Args:
        modules: Specific modules to scan. If None, scans all imported modules.

    Returns:
        List of tool definition dicts with name, produces_evidence, is_commit, etc.
    """
    tool_defs = []
    seen_names = set()

    scan_modules = modules if modules is not None else list(sys.modules.values())

    for module in scan_modules:
        if module is None:
            continue
        try:
            for attr_name in dir(module):
                try:
                    obj = getattr(module, attr_name)
                except Exception:
                    continue

                config = getattr(obj, "_sentinel_tool_config", None) or getattr(obj, "_sentinel_config", None)
                if config is None:
                    continue

                tool_name = config.get("name") or attr_name
                if tool_name in seen_names:
                    continue
                seen_names.add(tool_name)

                tool_defs.append({
                    "name": tool_name,
                    "agent_id": "",  # Will be set by caller
                    "produces_evidence": config.get("produces_evidence", False),
                    "is_commit": config.get("is_commit", False),
                    "requires": config.get("requires") or [],
                    "argument_constraints": config.get("argument_constraints"),
                    "evidence_max_age_seconds": config.get("evidence_max_age_seconds"),
                    "cost_usd": config.get("cost_usd", 0.0),
                    "tags": config.get("tags") or [],
                })
        except Exception as e:
            logger.debug(f"Error scanning module {module}: {e}")

    return tool_defs


def auto_register_tools(
    platform_url: str,
    api_token: str,
    agent_id: str,
    modules: Optional[List[ModuleType]] = None,
    sdk_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Discover and batch-register sentinel tools with the platform.

    Args:
        platform_url: Platform API base URL
        api_token: JWT or API key for authentication
        agent_id: Agent identifier
        modules: Specific modules to scan (default: all imported)
        sdk_version: SDK version string

    Returns:
        Registration response from platform
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for auto-registration. "
            "Install with: pip install agent-sentinel[remote]"
        )

    tool_defs = discover_tools(modules)

    if not tool_defs:
        logger.info("No sentinel tools found to register")
        return {"status": "success", "registered": 0, "updated": 0, "total": 0}

    # Set agent_id on all tools
    for tool in tool_defs:
        tool["agent_id"] = agent_id

    payload = {
        "agent_id": agent_id,
        "sdk_version": sdk_version or "0.1.0",
        "actions": tool_defs,
    }

    url = f"{platform_url.rstrip('/')}/api/v1/actions/register"
    headers = {"Authorization": f"ApiKey {api_token}"}

    response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
    response.raise_for_status()

    result = response.json()
    logger.info(
        f"Registered {result.get('registered', 0)} tools, "
        f"updated {result.get('updated', 0)} tools"
    )
    return result
