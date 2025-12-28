# AgentSentinel SDK

> **The operational circuit breaker for autonomous agents**

Runtime authority and control for autonomous AI agents. Decide whether your agents are allowed to act — before they do.

## 🎯 What is AgentSentinel?

AgentSentinel gives developers and organizations active control over the financial, logical, and compliance risks of autonomous agents in production.

**We are not:**
- A logging tool (that's a camera)
- An analytics dashboard (that's a report)
- An observability platform (that's hindsight)

**We are:**
- The brakes on the car
- The circuit breaker in the system
- The authorization layer between intent and action

> **Core Value:** AgentSentinel exists to decide whether an autonomous system is allowed to act — before it does.

## ✨ The Three Pillars

### 1. Active Safety 🛡️
Runtime enforcement that overrides agent intent. Every check happens BEFORE execution.

- ⚡ Budget enforcement (hard caps on cost per action, run, session)
- 🚫 Action bans (deny list prevents execution)
- ✅ Action allowlists (restrict to approved actions only)
- ⏱️ Rate limiting (time-windowed execution limits)
- 🛡️ Fail-safe design (never fails open without authorization)

### 2. Governance 📋
Complete authority over what happened, when, and why.

- 📝 Immutable execution ledger
- 💰 Per-action, per-run, per-session cost tracking
- 🔄 Replay mode for deterministic testing
- 📊 Decision rationale and data lineage
- 🇪🇺 EU AI Act Article 14 compliance metadata

### 3. Operations ⚙️
The bridge between autonomous agents and human authority.

- 👤 Human-in-the-loop approval workflows
- 📋 Policy engine with YAML configuration
- 🔌 Remote policy synchronization
- 📡 Real-time intervention tracking
- ⏸️ Interruptible execution with resumable authority

## 📦 Installation

```bash
# Basic installation
pip install agentsentinel-sdk

# With remote sync to platform
pip install agentsentinel-sdk[remote]

# With LLM integrations (OpenAI, Anthropic, Grok, Gemini)
pip install agentsentinel-sdk[llm]

# With framework integrations (LangChain, CrewAI, MCP)
pip install agentsentinel-sdk[integrations]

# With everything
pip install agentsentinel-sdk[all]

# Development installation
git clone https://github.com/agent-sentinel/agent-sentinel-sdk.git
cd agent-sentinel-sdk
uv sync
```

## 🚀 Quick Start

### Basic Usage

```python
from agent_sentinel import guarded_action

# Decorate any function to track it
@guarded_action(name="search_tool", cost_usd=0.01, tags=["search", "api"])
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

## 📊 Log Format

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

## 🎨 Advanced Usage

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

# Start background sync
sync = enable_remote_sync(
    platform_url="https://api.agentsentinel.dev",
    api_token="your-jwt-token",
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
    platform_url="https://api.agentsentinel.dev",
    api_token="your-jwt-token",
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

## 🔧 API Reference

### `@guarded_action(name=None, cost_usd=0.0, tags=None)`

Decorator to wrap a function with telemetry and cost tracking.

**Parameters:**
- `name` (str, optional): Name for the action. Defaults to function name.
- `cost_usd` (float, optional): Cost in USD for this action. Default: 0.0
- `tags` (list[str], optional): Tags for categorization. Default: []

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

## 🎯 Use Cases

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

## 🐛 Troubleshooting

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

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

See the main repository README for contribution guidelines.

## 📧 Support

- Issues: [GitHub Issues](https://github.com/agent-sentinel/agent-sentinel/issues)
- Email: hello@agentsentinel.dev
- Docs: https://docs.agentsentinel.dev (coming soon)

