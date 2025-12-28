"""
Model Context Protocol (MCP) Client for AgentSentinel SDK

This module provides MCP client functionality for agents to discover and
interact with AgentSentinel's platform capabilities via MCP.

MCP enables LLMs and agents to:
- Discover available tools and resources
- Query agent runs, actions, and policies
- Manage approvals and compliance
- Access real-time statistics and metrics

Example usage:
    from agent_sentinel.mcp import MCPClient
    
    # Initialize MCP client
    client = MCPClient(
        platform_url="https://api.agentsentinel.dev",
        api_token="your-jwt-token"
    )
    
    # Discover available tools
    tools = await client.list_tools()
    
    # Call a tool
    result = await client.call_tool(
        "create_policy",
        {
            "name": "Production Limits",
            "run_budget": 5.0,
            "enabled": True
        }
    )
    
    # Query resources
    latest_runs = await client.get_resource("agentsentinel://runs/latest")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning(
        "httpx not installed. MCP client requires httpx. "
        "Install with: pip install agent-sentinel[remote]"
    )

logger = logging.getLogger("agent_sentinel.mcp")


# ============================================================================
# MCP Data Models
# ============================================================================

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    type: str
    input_schema: Dict[str, Any]
    
    def __repr__(self) -> str:
        return f"MCPTool(name={self.name!r}, type={self.type!r})"


@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    
    def __repr__(self) -> str:
        return f"MCPResource(uri={self.uri!r}, name={self.name!r})"


@dataclass
class MCPPrompt:
    """MCP Prompt template"""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    
    def __repr__(self) -> str:
        return f"MCPPrompt(name={self.name!r})"


@dataclass
class MCPToolCallResult:
    """Result of an MCP tool call"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        status = "success" if self.success else "error"
        return f"MCPToolCallResult(status={status})"


# ============================================================================
# MCP Client
# ============================================================================

class MCPClient:
    """
    MCP Client for AgentSentinel
    
    Provides programmatic access to AgentSentinel's MCP server, enabling
    LLMs and agents to:
    - Discover available tools, resources, and prompts
    - Execute tool calls (create policies, query runs, manage approvals)
    - Access resources (latest runs, pending approvals, statistics)
    - Use prompt templates for common workflows
    
    Args:
        platform_url: AgentSentinel platform URL (e.g., "https://api.agentsentinel.dev")
        api_token: JWT authentication token
        timeout: Request timeout in seconds (default: 30)
    
    Example:
        >>> client = MCPClient(
        ...     platform_url="https://api.agentsentinel.dev",
        ...     api_token="your-jwt-token"
        ... )
        >>> 
        >>> # Discover tools
        >>> tools = await client.list_tools()
        >>> print(f"Found {len(tools)} tools")
        >>> 
        >>> # Call a tool
        >>> result = await client.call_tool("list_runs", {"limit": 10})
        >>> if result.success:
        ...     print(f"Runs: {result.data}")
    """
    
    def __init__(
        self,
        platform_url: str,
        api_token: str,
        timeout: float = 30.0
    ):
        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for MCP client. "
                "Install with: pip install agent-sentinel[remote]"
            )
        
        self.platform_url = platform_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        
        # Build API base URL
        self.api_base = f"{self.platform_url}/api/v1/mcp"
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Cache
        self._tools_cache: Optional[List[MCPTool]] = None
        self._resources_cache: Optional[List[MCPResource]] = None
        self._prompts_cache: Optional[List[MCPPrompt]] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    # ========================================================================
    # Discovery API
    # ========================================================================
    
    async def list_tools(self, use_cache: bool = True) -> List[MCPTool]:
        """
        List all available MCP tools.
        
        Args:
            use_cache: Use cached tools if available (default: True)
            
        Returns:
            List of MCPTool objects
            
        Example:
            >>> tools = await client.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        if use_cache and self._tools_cache:
            return self._tools_cache
        
        client = await self._get_client()
        response = await client.get(f"{self.api_base}/tools")
        response.raise_for_status()
        
        data = response.json()
        tools = [
            MCPTool(
                name=t["name"],
                description=t["description"],
                type=t["type"],
                input_schema=t["input_schema"]
            )
            for t in data["tools"]
        ]
        
        self._tools_cache = tools
        return tools
    
    async def list_resources(self, use_cache: bool = True) -> List[MCPResource]:
        """
        List all available MCP resources.
        
        Args:
            use_cache: Use cached resources if available (default: True)
            
        Returns:
            List of MCPResource objects
            
        Example:
            >>> resources = await client.list_resources()
            >>> for resource in resources:
            ...     print(f"{resource.uri}: {resource.description}")
        """
        if use_cache and self._resources_cache:
            return self._resources_cache
        
        client = await self._get_client()
        response = await client.get(f"{self.api_base}/resources")
        response.raise_for_status()
        
        data = response.json()
        resources = [
            MCPResource(
                uri=r["uri"],
                name=r["name"],
                description=r["description"],
                mime_type=r.get("mime_type", "application/json")
            )
            for r in data["resources"]
        ]
        
        self._resources_cache = resources
        return resources
    
    async def list_prompts(self, use_cache: bool = True) -> List[MCPPrompt]:
        """
        List all available MCP prompt templates.
        
        Args:
            use_cache: Use cached prompts if available (default: True)
            
        Returns:
            List of MCPPrompt objects
            
        Example:
            >>> prompts = await client.list_prompts()
            >>> for prompt in prompts:
            ...     print(f"{prompt.name}: {prompt.description}")
        """
        if use_cache and self._prompts_cache:
            return self._prompts_cache
        
        client = await self._get_client()
        response = await client.get(f"{self.api_base}/prompts")
        response.raise_for_status()
        
        data = response.json()
        prompts = [
            MCPPrompt(
                name=p["name"],
                description=p["description"],
                arguments=p.get("arguments", [])
            )
            for p in data["prompts"]
        ]
        
        self._prompts_cache = prompts
        return prompts
    
    # ========================================================================
    # Tool Execution API
    # ========================================================================
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> MCPToolCallResult:
        """
        Execute an MCP tool call.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool input arguments (optional)
            
        Returns:
            MCPToolCallResult with success status and data/error
            
        Example:
            >>> # Create a policy
            >>> result = await client.call_tool(
            ...     "create_policy",
            ...     {
            ...         "name": "Production Limits",
            ...         "run_budget": 5.0,
            ...         "enabled": True
            ...     }
            ... )
            >>> if result.success:
            ...     print(f"Policy created: {result.data['id']}")
            >>> else:
            ...     print(f"Error: {result.error}")
        """
        client = await self._get_client()
        
        payload = {
            "tool_name": tool_name,
            "arguments": arguments or {}
        }
        
        try:
            response = await client.post(
                f"{self.api_base}/call",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return MCPToolCallResult(
                success=data["success"],
                data=data.get("data"),
                error=data.get("error"),
                metadata=data.get("metadata")
            )
        
        except httpx.HTTPError as e:
            logger.error(f"MCP tool call failed: {e}")
            return MCPToolCallResult(
                success=False,
                error=f"HTTP error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return MCPToolCallResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    # ========================================================================
    # Resource Access API
    # ========================================================================
    
    async def get_resource(self, resource_uri: str) -> Optional[Any]:
        """
        Get data from an MCP resource.
        
        Args:
            resource_uri: Resource URI (e.g., "agentsentinel://runs/latest")
            
        Returns:
            Resource data (typically a dict) or None on error
            
        Example:
            >>> # Get latest runs
            >>> data = await client.get_resource("agentsentinel://runs/latest")
            >>> print(f"Latest runs: {data['runs']}")
            >>> 
            >>> # Get pending approvals
            >>> data = await client.get_resource("agentsentinel://approvals/pending")
            >>> print(f"Pending: {data['count']}")
        """
        client = await self._get_client()
        
        # Remove 'agentsentinel://' prefix if present
        uri = resource_uri.replace("agentsentinel://", "")
        
        try:
            response = await client.get(f"{self.api_base}/resources/{uri}")
            response.raise_for_status()
            
            data = response.json()
            return data.get("data")
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get resource '{resource_uri}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get resource '{resource_uri}': {e}")
            return None
    
    # ========================================================================
    # Prompt Execution API
    # ========================================================================
    
    async def execute_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Execute an MCP prompt template.
        
        Args:
            prompt_name: Name of the prompt template
            arguments: Prompt-specific arguments (optional)
            
        Returns:
            Prompt execution result or None on error
            
        Example:
            >>> # Get budget policy recommendations
            >>> result = await client.execute_prompt(
            ...     "create_budget_policy",
            ...     {
            ...         "use_case": "customer support",
            ...         "risk_level": "medium"
            ...     }
            ... )
            >>> print(f"Recommended budget: ${result['recommendation']['recommended_run_budget']}")
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.api_base}/prompts/{prompt_name}",
                json=arguments or {}
            )
            response.raise_for_status()
            
            return response.json()
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to execute prompt '{prompt_name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to execute prompt '{prompt_name}': {e}")
            return None
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    async def create_policy(
        self,
        name: str,
        run_budget: Optional[float] = None,
        session_budget: Optional[float] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method to create a policy.
        
        Args:
            name: Policy name
            run_budget: Maximum cost per run in USD
            session_budget: Maximum cost per session in USD
            **kwargs: Additional policy parameters
            
        Returns:
            Created policy data or None on error
        """
        arguments = {"name": name, **kwargs}
        if run_budget is not None:
            arguments["run_budget"] = run_budget
        if session_budget is not None:
            arguments["session_budget"] = session_budget
        
        result = await self.call_tool("create_policy", arguments)
        return result.data if result.success else None
    
    async def list_runs(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convenience method to list runs.
        
        Args:
            limit: Maximum number of runs to return
            status: Filter by status (running, completed, failed)
            **kwargs: Additional filter parameters
            
        Returns:
            List of run data or None on error
        """
        arguments = {"limit": limit, **kwargs}
        if status:
            arguments["status"] = status
        
        result = await self.call_tool("list_runs", arguments)
        return result.data if result.success else None
    
    async def get_pending_approvals(self) -> Optional[Dict[str, Any]]:
        """
        Convenience method to get pending approvals.
        
        Returns:
            Pending approvals data or None on error
        """
        return await self.get_resource("agentsentinel://approvals/pending")
    
    async def approve_action(
        self,
        action_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Convenience method to approve an action.
        
        Args:
            action_id: UUID of the action to approve
            notes: Optional approval notes
            
        Returns:
            True if successful, False otherwise
        """
        arguments = {"action_id": action_id}
        if notes:
            arguments["notes"] = notes
        
        result = await self.call_tool("approve_action", arguments)
        return result.success
    
    async def reject_action(
        self,
        action_id: str,
        reason: str
    ) -> bool:
        """
        Convenience method to reject an action.
        
        Args:
            action_id: UUID of the action to reject
            reason: Reason for rejection
            
        Returns:
            True if successful, False otherwise
        """
        result = await self.call_tool("reject_action", {
            "action_id": action_id,
            "reason": reason
        })
        return result.success
    
    async def get_agent_stats(self, days: int = 7) -> Optional[Dict[str, Any]]:
        """
        Convenience method to get agent statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics data or None on error
        """
        result = await self.call_tool("get_agent_stats", {"days": days})
        return result.data if result.success else None


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_client: Optional[MCPClient] = None


def set_default_client(
    platform_url: str,
    api_token: str,
    timeout: float = 30.0
):
    """
    Set the default MCP client for module-level functions.
    
    Args:
        platform_url: AgentSentinel platform URL
        api_token: JWT authentication token
        timeout: Request timeout in seconds
    
    Example:
        >>> from agent_sentinel.mcp import set_default_client
        >>> set_default_client(
        ...     platform_url="https://api.agentsentinel.dev",
        ...     api_token="your-jwt-token"
        ... )
    """
    global _default_client
    _default_client = MCPClient(
        platform_url=platform_url,
        api_token=api_token,
        timeout=timeout
    )


def get_default_client() -> Optional[MCPClient]:
    """Get the default MCP client"""
    return _default_client


