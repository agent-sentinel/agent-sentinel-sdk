# AutoGen Integration

Microsoft's AutoGen is one of the "Big Three" agent frameworks (alongside LangChain and CrewAI). This integration brings enterprise-grade monitoring and control to AutoGen multi-agent systems.

## Why AutoGen?

AutoGen is the leading framework in Microsoft shops and enterprise environments. Ignoring AutoGen means ignoring a massive slice of the enterprise market.

## Why This Integration is Simple

Unlike LangChain (callbacks) or CrewAI (wrappers), AutoGen has a **built-in hook system** (`register_reply`) designed exactly for this. You don't need to monkey-patch or wrap distinct tools; you just register a "Sentinel Hook" that sits between agents.

## The "Visa Terminal" Pattern

Think of Sentinel as a credit card terminal:
1. **Authorization**: Before an agent replies, check budget/policy
2. **Accounting**: Track token costs and message flow
3. **Audit**: Record all interventions for dashboard visibility

## Quick Start

### Install

```bash
pip install agent-sentinel pyautogen
```

### Basic Usage

```python
from autogen import AssistantAgent, UserProxyAgent
from agent_sentinel.integrations.autogen import SentinelInspector
from agent_sentinel.policy import PolicyEngine

# 1. Configure policies
PolicyEngine.configure(
    run_budget=0.50,      # $0.50 max per run
    session_budget=1.0,   # $1.00 max per session
)

# 2. Create AutoGen agents (no changes needed!)
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER"
)

# 3. Create Sentinel inspector
sentinel = SentinelInspector(
    run_name="my_conversation",
    enforce_policies=True,  # Active blocking
    track_costs=True,       # Cost tracking
)

# 4. Secure agents (one line each!)
sentinel.register(assistant)
sentinel.register(user_proxy)

# 5. Run as normal - fully monitored
sentinel.start_run()

user_proxy.initiate_chat(
    assistant,
    message="What's the weather in San Francisco?"
)

sentinel.end_run()

# 6. Review results
summary = sentinel.get_run_summary()
print(f"Cost: ${summary['run_cost_usd']:.6f}")
print(f"Messages: {summary['message_count']}")
print(f"LLM Calls: {summary['llm_call_count']}")
```

## How It Works

### 1. Register Reply Hook

```python
sentinel.register(agent)
```

This injects a hook into AutoGen's reply chain at **position 0** (highest priority), so it runs **before** any LLM calls or tool execution.

### 2. Authorization Check

Before each agent reply, Sentinel checks:
- ✅ Is the current budget exceeded?
- ✅ Is this action on the deny list?
- ✅ Does this violate rate limits?

If **any** check fails, Sentinel returns a blocking message and the agent **never** generates a reply.

### 3. Cost Tracking

Sentinel wraps the agent's LLM generation method to track:
- Token usage (prompt + completion)
- Model name
- Calculated cost
- Duration

### 4. Message Flow

Every agent-to-agent message is logged with:
- Sender and recipient
- Message content (preview)
- Timestamp and sequence
- Message count per agent

## Features

### Active Policy Enforcement

Unlike passive logging tools, Sentinel **blocks** actions that violate policies:

```python
# Set strict budget
PolicyEngine.configure(run_budget=0.10)

sentinel = SentinelInspector(
    run_name="limited_run",
    enforce_policies=True
)

sentinel.register(assistant)
sentinel.start_run()

try:
    user_proxy.initiate_chat(
        assistant,
        message="Write a 10,000 word essay..."
    )
except BudgetExceededError as e:
    print(f"BLOCKED: {e}")
```

When blocked:
1. Agent receives a blocking message
2. Intervention is recorded for dashboard
3. Exception is raised to stop execution
4. No LLM tokens are consumed (blocked **before** API call)

### Run Lifecycle Tracking

Mark run boundaries for accurate cost attribution:

```python
sentinel.start_run()

try:
    # Your agent conversation
    user_proxy.initiate_chat(assistant, message="...")
    sentinel.end_run(outcome="completed")
except Exception as e:
    sentinel.end_run(outcome="failed")
    raise
```

### Detailed Summaries

Get rich statistics after execution:

```python
summary = sentinel.get_run_summary()

# {
#   "run_name": "my_conversation",
#   "duration_seconds": 15.3,
#   "run_cost_usd": 0.042,
#   "message_count": 8,
#   "llm_call_count": 4,
#   "agent_message_counts": {
#     "assistant": 4,
#     "user_proxy": 4
#   },
#   "action_costs": {
#     "llm_call:gpt-4": 0.042
#   }
# }
```

### Multi-Agent Support

Secure as many agents as you need:

```python
sentinel = SentinelInspector(run_name="team_conversation")

# Secure all agents
sentinel.register(researcher)
sentinel.register(analyst)
sentinel.register(writer)
sentinel.register(reviewer)

# All are now monitored
sentinel.start_run()
# ... run your multi-agent workflow
sentinel.end_run()
```

## Advanced Usage

### Convenience Function

Create and secure agents in one step:

```python
from agent_sentinel.integrations.autogen import create_sentinel_agents
from autogen import AssistantAgent, UserProxyAgent

agents, sentinel = create_sentinel_agents(
    agent_configs=[
        {
            "agent_class": AssistantAgent,
            "name": "assistant",
            "llm_config": {"model": "gpt-4", "api_key": "..."}
        },
        {
            "agent_class": UserProxyAgent,
            "name": "user_proxy",
            "human_input_mode": "NEVER"
        }
    ],
    run_name="my_run",
    enforce_policies=True,
)

# Agents are already secured
sentinel.start_run()
agents[1].initiate_chat(agents[0], message="Hello!")
sentinel.end_run()
```

### Custom Tags

Add tags for better filtering in the dashboard:

```python
sentinel = SentinelInspector(
    run_name="production_run",
    tags=["autogen", "production", "customer_service"]
)
```

### Hook Priority

Control when Sentinel runs in the reply chain:

```python
# Default: position=0 (highest priority - runs first)
sentinel.register(agent, position=0)

# Run after other hooks
sentinel.register(agent, position=5)
```

## Comparison to Other Frameworks

| Feature | AutoGen | LangChain | CrewAI |
|---------|---------|-----------|---------|
| Integration Method | `register_reply` hook | Callback handler | Wrapper class |
| Complexity | ⭐ Simple | ⭐⭐ Medium | ⭐⭐⭐ Complex |
| Agent Communication | Native | Via chains | Via tasks |
| Multi-Agent | Native | Via agents | Native |
| Cost Tracking | Token-level | Token-level | Action-level |

**Winner**: AutoGen is the simplest because it has a built-in hook system!

## Best Practices

### ✅ DO

- Call `start_run()` and `end_run()` for accurate tracking
- Set `run_name` to something meaningful
- Use `enforce_policies=True` in production
- Review `get_run_summary()` after each run
- Register agents **before** starting conversation

### ❌ DON'T

- Register agents mid-conversation (hook won't apply to earlier messages)
- Forget to call `end_run()` (metrics won't be finalized)
- Use the same `SentinelInspector` for multiple runs (create new instance per run)
- Disable `enforce_policies` in production (defeats the purpose!)

## Troubleshooting

### "Agent not being monitored"

Make sure you:
1. Called `sentinel.register(agent)` **before** starting the conversation
2. Registered **all** agents in the conversation
3. The agent has a valid `name` attribute

### "Cost not tracked"

AutoGen cost tracking requires:
1. Agent has `llm_config` with a `model` specified
2. Model is in the pricing database (see `pricing.py`)
3. Agent uses OpenAI-compatible API (for token usage)

### "Policies not enforced"

Ensure:
1. `enforce_policies=True` when creating inspector
2. `PolicyEngine.configure()` was called before creating inspector
3. Policies are not being reset between runs

## Examples

See the `examples/` directory for complete examples:

- `autogen_example.py`: Basic usage and policy enforcement
- `autogen_multi_agent.py`: Multi-agent team with roles
- `autogen_budget_demo.py`: Budget limit demonstration

## Architecture

```
User Code
   ↓
   ↓ initiate_chat()
   ↓
UserProxyAgent ──────────→ AssistantAgent
   ↑                            ↑
   │                            │
   │ [Sentinel Hook]            │ [Sentinel Hook]
   │  ├─ Authorize             │  ├─ Authorize
   │  ├─ Track Message         │  ├─ Track Message
   │  └─ Log to Ledger         │  └─ Log to Ledger
   │                            │
   └──────────────────────────┘
              ↓
         [LLM Wrapper]
          ├─ Track Tokens
          ├─ Calculate Cost
          └─ Update CostTracker
              ↓
          OpenAI API
```

## Integration with Platform

For enterprise features (dashboard, approvals, compliance):

```python
from agent_sentinel import enable_remote_sync

enable_remote_sync(
    platform_url="https://api.agentsentinel.dev",
    api_token="as_your_token_here",
)

# Everything is now synced to platform
sentinel = SentinelInspector(run_name="production_run")
# ... rest of your code
```

## Contributing

Found a bug or have a feature request? Open an issue on GitHub!

## License

MIT License - see LICENSE file for details.

---

**Built by the Agent Sentinel team** 🛡️

For more information, visit [agentsentinel.dev](https://agentsentinel.dev)
