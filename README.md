# Agent Sentinel SDK

[![PyPI](https://img.shields.io/pypi/v/agentsentinel-sdk)](https://pypi.org/project/agentsentinel-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/agentsentinel-sdk)](https://pypi.org/project/agentsentinel-sdk/)
[![License: MIT](https://img.shields.io/pypi/l/agentsentinel-sdk)](LICENSE)

**The runtime authority for AI agents.** Every tool call your agent makes is
authorized before it runs, budgets are reserved before execution, and what
happened is recorded in a ledger you can verify.

[Documentation](https://docs.agentsentinel.dev) ·
[Console](https://console.agentsentinel.dev) ·
[Website](https://agentsentinel.dev) ·
[Changelog](CHANGELOG.md)

## What it does

Observability tells you what happened. Agent Sentinel decides what is allowed
to happen, before it does. Two ways to put it in front of your agent:

- **In-process, with the SDK.** Decorate functions with `@guarded_action`.
  Policy, budgets, approvals, loop protection and PII/moderation guardrails run
  before the function body does. No approval, no execution.
- **In front of every MCP tool call.** `agentsentinel-mcp` and
  `agentsentinel-runtime` connect Claude Code, Cursor or any MCP client to the
  hosted gateway, so each `tools/call` is authorized first and the model never
  holds the credential.

What you get:

- Hard budget caps per action, run and session, reserved before execution
- Deny lists, allow lists, rate limits and policy-as-YAML
- Human approval workflows for sensitive actions
- Fail-closed by default for writes (since 0.4.0)
- Secrets redacted from ledger inputs before they are written or uploaded
- A local JSONL ledger, plus remote sync to the hosted platform, where ledger
  rows are Ed25519-signed and the chain can be verified from the console or API
- Remediation on every block (retry guidance, safe alternatives) so an agent can
  self-correct

## Installation

```bash
pip install agentsentinel-sdk                  # core: decorator, policy, local ledger
pip install "agentsentinel-sdk[remote]"        # sync to the hosted platform
pip install "agentsentinel-sdk[mcp]"           # agentsentinel-mcp and agentsentinel-runtime
pip install "agentsentinel-sdk[langchain]"     # LangChain callbacks
pip install "agentsentinel-sdk[crewai]"        # CrewAI callbacks
pip install "agentsentinel-sdk[llm]"           # OpenAI, Anthropic, Grok, Gemini helpers
pip install "agentsentinel-sdk[integrations]"  # all framework integrations
pip install "agentsentinel-sdk[all]"           # everything
```

The package needs Python 3.11+. For development, work from
`packages/agent-sentinel-sdk` in the
[monorepo](https://github.com/agent-sentinel/agent-sentinel) with `uv sync`.

## Quick Start

### Basic Usage

```python
from agent_sentinel import guarded_action

# Decorate any function to track it. impact="read" declares this as a
# read: since 0.4.0 the default enforcement mode blocks unconfigured
# actions of impact "medium"/"high" (see Enforcement modes below).
@guarded_action(name="search_tool", cost_usd=0.01, tags=["search", "api"], impact="read")
def search_web(query: str) -> dict:
    """Search the web for a query"""
    results = call_search_api(query)
    return {"results": results, "count": len(results)}

# Use it normally - telemetry happens automatically
response = search_web("Python best practices")
```

### Async Support

```python
@guarded_action(name="generate_text", cost_usd=0.05, tags=["llm"])
async def generate_text(prompt: str) -> str:
    """Generate text using an LLM"""
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Async works automatically
result = await generate_text("Write a haiku about Python")
```

### Complex Objects

The SDK handles complex, non-serializable objects gracefully:

```python
from dataclasses import dataclass
import socket

@dataclass
class Context:
    user_id: int
    session: str
    connection: socket.socket  # Not JSON-serializable!

@guarded_action(name="process_data", cost_usd=0.001)
def process_data(ctx: Context, data: dict):
    # Complex objects are automatically stringified
    return {"processed": True, "user": ctx.user_id}

# No crashes even with non-serializable objects!
ctx = Context(user_id=123, session="abc", connection=socket.socket())
process_data(ctx, {"value": 42})
```

## MCP authority

Every MCP `tools/call` goes through the gateway's authorize decision before the
tool executes (allow, deny or require approval), and the result is recorded as a
ledger row. Two stdio entry points ship in the `[mcp]` extra:

### `agentsentinel-mcp`

A stdio MCP server for MCP-native harnesses such as Claude Code and Cursor. It
answers `initialize` locally and forwards `tools/list` and `tools/call` to the
platform's `/mcp` endpoint with an agent-bound API key, so the harness gets
governed tools and the model never sees a credential. Results pass through
unchanged, including `isError` and `structuredContent.reason_code` on refusals.

```json
{
  "mcpServers": {
    "agentsentinel": {
      "command": "agentsentinel-mcp",
      "env": {
        "AGENT_SENTINEL_API_KEY": "<agent-bound API key>",
        "AGENT_SENTINEL_HARNESS": "claude-code"
      }
    }
  }
}
```

Put this in `.mcp.json` for Claude Code or `.cursor/mcp.json` for Cursor. The
file then holds the key, so keep it out of version control, or use
`agentsentinel-runtime` below to keep the key out of the workspace entirely.

- `AGENT_SENTINEL_API_KEY` (required) must be an agent-bound key.
- `AGENT_SENTINEL_URL` (optional) overrides the hosted platform URL for
  self-hosted or staging; the proxy targets `<url>/mcp`. `https` is required
  unless the host is `localhost` or `127.0.0.1`.
- `AGENT_SENTINEL_HARNESS` (optional) is sent as `X-Sentinel-Harness`.

A missing key or a disallowed URL exits non-zero with a one-line message on
stderr. The key is only ever sent in the upstream `Authorization` header and
never appears in output, logs or MCP messages. Upstream 401/403 or transport
failures surface as an MCP error (`tools/list`) or an `isError` tool result
(`tools/call`). See the
[Claude Code](https://docs.agentsentinel.dev/ai-tools/claude-code) and
[Cursor](https://docs.agentsentinel.dev/ai-tools/cursor) guides.

### `agentsentinel-runtime` (0.5+)

Serves the same stdio proxy from one validated `runtime.yaml` (`url`,
`workspace`, `api_key_file`, `harness`, `approvals`), reading the key from a file
**outside the model's workspace**. It is outbound-only. Subcommands: `run`,
`config` and `doctor`.

- **Approvals (0.6+).** With `approvals.wait: true`, a `tools/call` that the
  platform answers with `require_approval` is held while the runtime polls the
  approval outbound with the same key, then retried once if approved. Rejected,
  expired, cancelled and info-requested outcomes return a clear tool error; a
  wait that times out does not cancel the approval.
- **`agentsentinel-runtime doctor` (0.6+)** checks that the key is out of the
  model's reach: key and config file location, ownership and permissions, a
  bounded scan of the workspace for the key's bytes, known harness config
  files, the environment, and upstream reachability (`--offline` skips it). It
  never passes what it cannot prove, for example when the runtime runs as root or
  as the user that owns or can write the workspace.

`agent_sentinel.mcp` remains the client for the legacy REST facade.

## Log Format

Logs are written as JSON Lines (`.jsonl`) to `.agent-sentinel/ledger.jsonl`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-01-15T10:30:00.123456Z",
  "action": "search_tool",
  "cost_usd": 0.01,
  "duration_ms": 234.5,
  "outcome": "success",
  "tags": ["search", "api"],
  "payload": {
    "inputs": {
      "args": [],
      "kwargs": {"query": "Python best practices"}
    },
    "outputs": {
      "results": ["..."],
      "count": 10
    }
  }
}
```

### Fields Explained

- `id`: Unique UUID for this action
- `timestamp`: ISO 8601 timestamp (UTC)
- `action`: Name of the action (from decorator or function name)
- `cost_usd`: Cost in USD for this action
- `duration_ms`: Execution time in milliseconds
- `outcome`: `"success"` or `"error"`
- `tags`: Custom tags for categorization
- `payload.inputs`: Function arguments (args and kwargs)
- `payload.outputs`: Return value (on success) or error message (on failure)

## Advanced Usage

### Budget Enforcement

**Programmatic Configuration:**

```python
from agent_sentinel import PolicyEngine, CostTracker, BudgetExceededError

# Configure budget limits
PolicyEngine.configure(
    session_budget=5.0,    # Max $5 for entire session
    run_budget=1.0,        # Max $1 per run
    action_budgets={
        "expensive_llm": 0.50,  # Max $0.50 for this action
    }
)

# Your actions now respect these limits
@guarded_action(name="expensive_llm", cost_usd=0.30)
def call_gpt4(prompt: str) -> str:
    # This will be blocked if budget exceeded
    return generate_text(prompt)

# Check current costs
print(f"Run total: ${CostTracker.get_run_total():.2f}")

# Reset between agent runs
CostTracker.reset_run()
```

**YAML Configuration:**

Create `callguard.yaml`:

```yaml
budgets:
  session: 5.0
  run: 1.0
  actions:
    expensive_llm: 0.50

denied_actions:
  - delete_database
  - dangerous_operation

strict_mode: true
```

Load it:

```python
from agent_sentinel import PolicyEngine

PolicyEngine.load_from_yaml("callguard.yaml")
# Now all decorated actions respect these policies
```

### Action Control Lists

**Deny Dangerous Actions:**

```python
PolicyEngine.configure(
    denied_actions=["delete_database", "send_money"]
)

@guarded_action(name="delete_database")
def delete_database():
    # This will raise PolicyViolationError immediately
    pass
```

**Allowlist Safe Actions:**

```python
PolicyEngine.configure(
    allowed_actions=["read_file", "search_web"]
)

# Only these actions are permitted
# All others will raise PolicyViolationError
```

### Remote Sync to Platform

**Enable Background Sync:**

```python
from agent_sentinel import enable_remote_sync, flush_and_stop

# Start background sync to the hosted platform. The URL defaults to the hosted
# platform (0.5+); set AGENT_SENTINEL_PLATFORM_URL to target a self-hosted or
# staging deployment.
sync = enable_remote_sync(
    api_token="your-api-key",  # or set AGENT_SENTINEL_API_KEY
    flush_interval=10.0,  # Upload every 10 seconds
)

# Use your agent normally - logs are synced automatically

# At exit, flush remaining logs
flush_and_stop()
```

**Manual Sync Control:**

```python
from agent_sentinel import BackgroundSync, SyncConfig

# Create custom sync config
config = SyncConfig(
    api_token="your-api-key",
    run_id="custom-run-id",  # Optional
    flush_interval=5.0,
    batch_size=50,
    max_retries=3,
)

# Start sync
sync = BackgroundSync(config)
sync.start()

# Trigger immediate flush
sync.flush_now()

# Stop with final flush
sync.stop()
```

**Platform URL:** every client (`configure`, `enable_remote_sync`,
`SyncConfig`, `PolicyEngine.enable_remote_sync`, `KillSwitchClient`,
`ApprovalClient`, `RemoteCostBackend`, `MCPClient`) connects to the hosted
AgentSentinel platform by default, so `platform_url` is optional. The API key
falls back to `AGENT_SENTINEL_API_KEY` when not passed.

| Variable | Purpose |
| --- | --- |
| `AGENT_SENTINEL_API_KEY` | Platform API key, used when `api_token`/`api_key` is not passed |
| `AGENT_SENTINEL_PLATFORM_URL` | Overrides the hosted platform URL (for self-hosted or staging) |
| `AGENT_SENTINEL_URL` | Same override, lower precedence than `AGENT_SENTINEL_PLATFORM_URL` |

Precedence: explicit `platform_url` argument > `AGENT_SENTINEL_PLATFORM_URL` >
`AGENT_SENTINEL_URL` > hosted platform.

**Fail-Open Design:**

The sync is designed to never crash your agent:
- If platform is unreachable, logs stay local
- Retries with exponential backoff
- Continues logging locally even if all uploads fail
- Agent performance unaffected by network issues

### Custom Action Names

```python
# Use function name (default)
@guarded_action(cost_usd=0.01)
def my_function():
    pass  # Action name: "my_function"

# Or specify explicitly
@guarded_action(name="custom_name", cost_usd=0.01)
def my_function():
    pass  # Action name: "custom_name"
```

### Tagging and Organization

```python
@guarded_action(
    name="expensive_operation",
    cost_usd=1.50,
    tags=["llm", "gpt4", "production"]
)
def expensive_operation():
    pass
```

### Error Handling

The decorator **never** catches your exceptions - they propagate normally:

```python
@guarded_action(name="might_fail", cost_usd=0.01)
def risky_operation():
    if something_bad:
        raise ValueError("Oops!")  # This exception is raised normally
    return "success"

try:
    risky_operation()
except ValueError as e:
    print(f"Caught: {e}")
    # Exception is logged with outcome="error" and error message in outputs
```

### Configuration

```python
import os

# Change log directory via environment variable
os.environ["AGENT_SENTINEL_HOME"] = "/var/log/my-agent"

# Now logs go to /var/log/my-agent/ledger.jsonl
```

### Reading Logs Programmatically

```python
import json
from pathlib import Path

ledger_path = Path(".agent-sentinel/ledger.jsonl")

# Read all entries
with open(ledger_path) as f:
    entries = [json.loads(line) for line in f]

# Calculate total cost
total_cost = sum(e["cost_usd"] for e in entries)
print(f"Total cost: ${total_cost:.2f}")

# Filter by outcome
errors = [e for e in entries if e["outcome"] == "error"]
print(f"Found {len(errors)} errors")
```

## API Reference

### `@guarded_action(name=None, cost_usd=0.0, tags=None, ...)`

Decorator to wrap a function with telemetry and cost tracking.

**Parameters:**
- `name` (str, optional): Name for the action. Defaults to function name.
- `cost_usd` (float, optional): Cost in USD for this action. Default: 0.0
- `tags` (list[str], optional): Tags for categorization. Default: []
- `requires_human_approval` (bool, optional): Hold the call until a human
  approves it. `approval_description` customizes what the approver sees.
- `idempotency_key` / `idempotency_ttl_seconds` (optional): Replay a prior
  result instead of running a duplicate side effect.
- `agent_id`, `task_id`, `mission_id` (str, optional): Attribution for the
  ledger and platform.
- `impact` (`"read"` | `"low"` | `"medium"` | `"high"`, optional): Declared
  consequentiality of the action. Default: `"medium"`. Sent to the platform
  in the ledger entry and used by the `fail_closed_writes` enforcement mode
  — see [Enforcement modes](#enforcement-modes).

**Returns:**
- Decorated function that logs telemetry and re-raises exceptions

**Example:**
```python
@guarded_action(name="my_action", cost_usd=0.05, tags=["api", "external"])
def my_function(arg1, arg2):
    return arg1 + arg2
```

### Exceptions

All exceptions inherit from `AgentSentinelError`:

```python
from agent_sentinel import (
    AgentSentinelError,       # Base exception
    BudgetExceededError,       # Cost limit exceeded (Phase 2)
    PolicyViolationError,      # Policy rule violated (Phase 2)
    ReplayDivergenceError,     # Replay mismatch (coming soon)
)

# Handle budget violations
try:
    expensive_operation()
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    # Take corrective action

# Handle policy violations
try:
    denied_operation()
except PolicyViolationError as e:
    print(f"Policy violation: {e}")
    # Log security incident
```

### Enforcement modes

As of 0.4.0, the default is `fail_closed_writes`: an unconfigured or
unreachable `PolicyEngine` blocks write-impact actions and permits
read-impact ones. **This is a breaking change from 0.3.x**, where an
unconfigured engine permitted everything unconditionally — see the escape
hatch below if you need the old behaviour for now.

Every `@guarded_action` declares an `impact`:

```python
from agent_sentinel import guarded_action

@guarded_action(name="read_balance", impact="read")   # or "low" — always proceeds
def read_balance(account_id):
    ...

@guarded_action(name="transfer_funds")   # impact defaults to "medium" — a write
def transfer_funds(from_account, to_account, amount):
    ...
```

`impact` is one of `"read"`, `"low"`, `"medium"`, or `"high"` (default
`"medium"`). It is sent to the platform in the ledger/ingest entry
(`entry["impact"]`) and consulted locally whenever the policy layer cannot
vouch for the action — unconfigured, a remote sync that has never produced a
usable local cache, or policy evaluation itself raising an unexpected error:

- `impact="medium"`/`"high"` (a write): **blocked** with
  `NoPolicyForWriteError` (`reason_code="NO_POLICY_FOR_WRITE"`).
- `impact="read"`/`"low"`: **permitted**, with a one-time warning logged.

```python
from agent_sentinel import PolicyEngine

PolicyEngine.set_enforcement_mode("fail_open")     # or AGENT_SENTINEL_ENFORCEMENT=fail_open
```

**Escape hatch:** set `fail_open` explicitly (env var or
`set_enforcement_mode`) to restore the exact pre-0.4 behaviour: an
unconfigured engine permits every action regardless of declared impact. It
exists for migrations, not as a recommended mode.

```python
PolicyEngine.set_enforcement_mode("fail_closed")   # or AGENT_SENTINEL_ENFORCEMENT=fail_closed
```

`fail_closed` remains available too: checking any action with no policy
loaded raises `PolicyNotConfiguredError`, regardless of declared impact —
use it in any deployment whose job is enforcing everything, not just writes.

An unrecognised value in `AGENT_SENTINEL_ENFORCEMENT` raises rather than
falling back — an operator who typed it was turning enforcement on.

`PolicyEngine.reset()` is a test/ops utility with its own, independent
contract: it reverts an unpinned mode to `fail_open`, not to the SDK's
`fail_closed_writes` default. A real process that never calls `reset()` or
`set_enforcement_mode()` sees `fail_closed_writes` from the start.

### Sharing budgets across processes

`CostTracker` keeps counters in process memory by default. That is exact for a
single agent using `@guarded_action`, and wrong for anything multi-process — a
proxy, a worker pool, a CLI invoked per tool call — where every process starts
from a zero balance and budget enforcement silently stops working.

```python
from agent_sentinel import CostTracker, FileCostBackend

CostTracker.set_backend(FileCostBackend("/var/lib/sentinel/costs.json"))
```

or, the usual path for a container:

```bash
AGENT_SENTINEL_COST_BACKEND=file
AGENT_SENTINEL_COST_PATH=/var/lib/sentinel/costs.json
```

**`@guarded_action` reserves atomically.** Checking the budget and recording
the spend as two separate calls is check-then-act: concurrent workers can all
pass the check before any of them records, and the cap gets overshot. Measured
with 6 threads against a $1.00 run budget at $0.50 each, the two-step path
spends $3.00; reserving holds it at exactly $1.00 with 2 permitted and 4
blocked.

You get this by default — the decorator calls `PolicyEngine.check_and_reserve()`
for you. **Do not call it yourself on an action that is already decorated**, or
the spend is counted twice.

```python
# Only for code that is NOT using @guarded_action:
PolicyEngine.check_and_reserve("transfer_funds", cost=0.50)
# ...run the action. The reservation is the record; do not add the cost again.
```

`check_action()` remains available for callers that want to ask without
reserving. For enforcement across multiple hosts, implement `CostBackend`
against Redis or Postgres — the contract is five methods.

### CostTracker

Track costs across your application:

```python
from agent_sentinel import CostTracker

# Get current totals
session_total = CostTracker.get_session_total()
run_total = CostTracker.get_run_total()

# Get per-action statistics
stats = CostTracker.get_action_stats("expensive_llm")
print(f"Calls: {stats['count']}, Cost: ${stats['total_cost']:.2f}")

# Get all action stats
all_stats = CostTracker.get_action_stats()
for action, cost in all_stats['costs'].items():
    count = all_stats['counts'][action]
    print(f"{action}: {count} calls, ${cost:.2f}")

# Reset for new run
CostTracker.reset_run()

# Get complete snapshot
snapshot = CostTracker.get_snapshot()
```

### PolicyEngine

Configure and enforce policies:

```python
from agent_sentinel import PolicyEngine

# Programmatic configuration
PolicyEngine.configure(
    session_budget=10.0,
    run_budget=1.0,
    action_budgets={"expensive_action": 0.50},
    denied_actions=["dangerous_op"],
    allowed_actions=None,  # None = all allowed (except denied)
    strict_mode=True
)

# Load from YAML
PolicyEngine.load_from_yaml("callguard.yaml")

# Check if configured
if PolicyEngine.is_configured():
    config = PolicyEngine.get_config()
    print(f"Run budget: ${config.run_budget}")

# Reset (for testing)
PolicyEngine.reset()
```

### Ledger

Direct access to the ledger (advanced usage):

```python
from agent_sentinel import Ledger

# Manually record an entry
Ledger.record(
    action="manual_entry",
    inputs={"key": "value"},
    outputs={"result": 42},
    cost_usd=0.01,
    duration_ms=100.0,
    outcome="success",
    tags=["manual"]
)
```

## Use Cases

### 1. LLM Agent Monitoring

```python
@guarded_action(name="llm_call", cost_usd=0.03, tags=["openai", "gpt4"])
async def call_llm(messages: list[dict]) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content
```

### 2. Tool Call Tracking

```python
@guarded_action(name="database_query", cost_usd=0.001, tags=["db"])
def query_database(sql: str) -> list:
    return database.execute(sql).fetchall()

@guarded_action(name="api_request", cost_usd=0.01, tags=["external"])
def call_external_api(endpoint: str) -> dict:
    return requests.get(f"{BASE_URL}/{endpoint}").json()
```

### 3. Multi-Agent Systems

```python
# Agent 1
@guarded_action(name="planner_think", cost_usd=0.02, tags=["agent1", "planning"])
def plan_tasks(goal: str) -> list[str]:
    return ["task1", "task2", "task3"]

# Agent 2
@guarded_action(name="executor_run", cost_usd=0.05, tags=["agent2", "execution"])
def execute_task(task: str) -> dict:
    return {"status": "done", "task": task}
```

## Framework Integrations

AgentSentinel provides robust integrations with popular AI frameworks, featuring active policy enforcement and intervention tracking.

### LangChain Integration

The LangChain integration provides **4 critical features**:

1. **Pre-Authorization Checks**: Blocks tools/LLMs if budget exceeded or policy violated (like a credit card terminal checking funds)
2. **Intervention Recording**: Tracks blocked/modified actions for dashboard visibility
3. **Async Support**: Fully compatible with async LangChain agents in production
4. **Context Propagation**: Maintains agent identity and budget across nested chains

**Basic Usage:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from agent_sentinel.integrations.langchain import SentinelCallbackHandler

# Create callback handler with enforcement
sentinel = SentinelCallbackHandler(
    run_name="my_agent_run",
    track_costs=True,
    track_tools=True,
    enforce_policies=True  # Enable active blocking
)

# Use with LangChain LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    callbacks=[sentinel]
)

# Or with agents
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[sentinel],
    verbose=True
)

# Run agent - enforcement happens automatically
try:
    result = agent.run("What's the weather in SF?")
except BudgetExceededError as e:
    print(f"Blocked: {e}")
    # View interventions
    from agent_sentinel import InterventionTracker
    interventions = InterventionTracker.get_interventions()
```

**With Policy Enforcement:**

```python
from agent_sentinel import PolicyEngine

# Configure budget and deny lists
PolicyEngine.configure(
    run_budget=0.50,  # Max $0.50 per run
    denied_actions=["delete_database", "send_email"],
    strict_mode=True
)

# Create handler - it will enforce policies before each action
sentinel = SentinelCallbackHandler(
    run_name="policy_enforced_agent",
    enforce_policies=True
)

llm = ChatOpenAI(callbacks=[sentinel])

# Second LLM call will be BLOCKED if over budget
# Dangerous tools will be BLOCKED before execution
# All blocks are recorded as interventions for dashboard
```

**Async Agents:**

```python
# Works with async LangChain agents automatically
async def run_async_agent():
    sentinel = SentinelCallbackHandler(
        run_name="async_agent",
        enforce_policies=True
    )
    
    llm = ChatOpenAI(callbacks=[sentinel])
    # All async callbacks are supported
    result = await llm.apredict("Hello!")
    return result
```

**Rate Limiting:**

```python
PolicyEngine.configure(
    rate_limits={
        "expensive_search": {
            "max_count": 10,
            "window_seconds": 60
        }
    }
)

# Tool will be blocked after 10 calls per minute
sentinel = SentinelCallbackHandler(enforce_policies=True)
agent = initialize_agent(tools, llm, callbacks=[sentinel])
```

**View Interventions:**

```python
from agent_sentinel import InterventionTracker

# After running your agent
interventions = InterventionTracker.get_interventions()

for intervention in interventions:
    print(f"Type: {intervention.intervention_type.value}")
    print(f"Outcome: {intervention.outcome.value}")
    print(f"Action: {intervention.action_name}")
    print(f"Reason: {intervention.reason}")
    print(f"Risk Level: {intervention.risk_level}")
```

**Examples:**

- Basic tracking: `examples/langchain_integration.py`
- Policy enforcement: `examples/langchain_policy_enforcement.py`

### CrewAI Integration

```python
from agent_sentinel.integrations.crewai import SentinelCrewCallbacks
from crewai import Agent, Task, Crew

# Create callbacks
callbacks = SentinelCrewCallbacks(
    run_name="crewai_agent",
    track_costs=True
)

# Create agent
agent = Agent(
    role="Researcher",
    goal="Research topics",
    backstory="Expert researcher",
    callbacks=callbacks
)

# Create and run crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

See `examples/crewai_integration.py` for more details.

### OpenAI Integration

```python
from agent_sentinel.integrations.openai import track_openai

# Track OpenAI calls automatically
track_openai()

# Now all OpenAI calls are tracked
import openai
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

See `examples/llm_instrumentation.py` for more integrations.

## Troubleshooting

### Logs not appearing?

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check if directory is writable
from pathlib import Path
log_dir = Path(".agent-sentinel")
print(f"Log directory: {log_dir.absolute()}")
print(f"Exists: {log_dir.exists()}")
print(f"Writable: {log_dir.exists() and log_dir.is_dir()}")
```

### Large logs?

The SDK appends to the same file. Rotate logs periodically:

```bash
# Rotate logs (keep last 5)
mv .agent-sentinel/ledger.jsonl .agent-sentinel/ledger-$(date +%Y%m%d).jsonl
# Compress old logs
gzip .agent-sentinel/ledger-*.jsonl
```

## License

MIT License - see LICENSE file for details

## Contributing

See the [monorepo](https://github.com/agent-sentinel/agent-sentinel) for contribution guidelines.

## Support

- Issues: [GitHub Issues](https://github.com/agent-sentinel/agent-sentinel/issues)
- Email: hello@agentsentinel.dev
- Docs: https://docs.agentsentinel.dev

