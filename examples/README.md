# Agent Sentinel Examples

This directory contains working examples for all major framework integrations.

## Quick Start

### Install Dependencies

```bash
# Core SDK
pip install agent-sentinel

# Framework of your choice (or all three):
pip install pyautogen        # For AutoGen examples
pip install langchain langchain-openai  # For LangChain examples
pip install crewai           # For CrewAI examples
```

### Set API Keys

```bash
export OPENAI_API_KEY="your-key-here"
# or for platform integration:
export AGENT_SENTINEL_API_TOKEN="your-token"
```

## Examples

### 🎯 AutoGen Example

**File**: `autogen_example.py`

The simplest integration! Shows:
- Basic agent monitoring
- Policy enforcement
- Budget limits
- Cost tracking
- Blocking demonstration

```bash
python autogen_example.py
```

**Key Features**:
- One-line registration per agent
- Active policy enforcement
- Message flow tracking
- Run summaries

### 🔗 LangChain Example

**File**: Coming soon - `langchain_example.py`

Shows:
- Callback handler setup
- Tool tracking
- Chain monitoring
- Agent execution

```bash
python langchain_example.py
```

### 🚢 CrewAI Example

**File**: Coming soon - `crewai_example.py`

Shows:
- Crew wrapping
- Task tracking
- Agent roles
- Multi-agent coordination

```bash
python crewai_example.py
```

### 📊 Framework Comparison

**File**: `all_frameworks_comparison.py`

Side-by-side comparison of all three frameworks showing:
- Integration patterns
- Code differences
- Feature parity
- Usage patterns

```bash
python all_frameworks_comparison.py
```

## Example Structure

Each example follows this pattern:

```python
# 1. Import framework + Sentinel
from framework import Agent
from agent_sentinel.integrations.framework import SentinelIntegration

# 2. Configure policies
from agent_sentinel.policy import PolicyEngine
PolicyEngine.configure(run_budget=0.50)

# 3. Create agents (normal code)
agent = Agent(...)

# 4. Secure with Sentinel (1-3 lines)
sentinel = SentinelIntegration(...)
sentinel.register(agent)

# 5. Run as normal
# ... your code here ...

# 6. Review results
summary = sentinel.get_run_summary()
print(f"Cost: ${summary['cost']:.6f}")
```

## Common Patterns

### Setting Budget Limits

```python
from agent_sentinel.policy import PolicyEngine

PolicyEngine.configure(
    run_budget=0.50,      # $0.50 per run
    session_budget=1.0,   # $1.00 per session
)
```

### Tracking Costs

```python
from agent_sentinel.cost import CostTracker

# During execution
current = CostTracker.get_run_total()
print(f"Spent so far: ${current:.6f}")

# After completion
summary = sentinel.get_run_summary()
print(f"Total: ${summary['run_cost_usd']:.6f}")
```

### Handling Blocks

```python
from agent_sentinel.errors import BudgetExceededError

try:
    # Your agent code
    agent.run(...)
except BudgetExceededError as e:
    print(f"BLOCKED: {e}")
    # Budget exceeded - action prevented
```

## Platform Integration

For enterprise features (dashboard, approvals):

```python
from agent_sentinel import enable_remote_sync

enable_remote_sync(
    platform_url="https://api.agentsentinel.dev",
    api_token="as_your_token_here",
)

# Everything is now synced to platform
```

## Troubleshooting

### "Module not found" errors

Make sure you've installed the framework:
```bash
pip install pyautogen  # or langchain, crewai
```

### "API key not found"

Set your OpenAI key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "No pricing data" warnings

Some models may not have pricing info. Either:
1. Use a known model (gpt-4, gpt-3.5-turbo)
2. Add custom pricing to `pricing.py`

### "Callbacks not working" (LangChain)

Make sure to propagate callbacks:
```python
llm = ChatOpenAI(callbacks=[handler])
agent = create_agent(..., callbacks=[handler])
result = agent.invoke(..., config={"callbacks": [handler]})
```

## Advanced Examples

### Multi-Agent Teams

See `autogen_example.py` for multi-agent conversation monitoring.

### Custom Tools

Add cost tracking to your custom tools:
```python
from agent_sentinel import guarded_action

@guarded_action(name="my_tool", cost_usd=0.01)
def my_tool(query: str) -> str:
    # Your tool implementation
    return result
```

### Rate Limiting

Prevent API abuse:
```python
PolicyEngine.configure(
    rate_limits={
        "openai_chat_completion": {
            "max_count": 10,
            "window_seconds": 60
        }
    }
)
```

## Example Output

Typical output from an example:

```
================================================================================
AutoGen + Agent Sentinel Example
================================================================================

✅ Policy configured: Run budget = $0.50, Session budget = $1.00
✅ Created AutoGen agents: assistant and user_proxy
✅ Sentinel registered with both agents
   - Policy enforcement: ENABLED
   - Cost tracking: ENABLED
   - Message tracking: ENABLED

================================================================================
Starting Agent Conversation
================================================================================

user_proxy (to assistant):
What's the weather in San Francisco?

assistant (to user_proxy):
I don't have real-time weather data...

✅ Conversation completed successfully

================================================================================
Run Summary
================================================================================

Run Name: autogen_demo
Duration: 3.45 seconds
Cost: $0.012340
Messages Exchanged: 4
LLM Calls: 2

Agent Message Counts:
  - assistant: 2 messages
  - user_proxy: 2 messages

Action Costs:
  - llm_call:gpt-4: $0.012340 (2 calls)

Budget Status:
  - Run: $0.012340 / $0.50 ($0.487660 remaining)
  - Session: $0.012340 / $1.00 ($0.987660 remaining)
```

## Learning Path

1. Start with `autogen_example.py` (simplest)
2. Try `all_frameworks_comparison.py` (see patterns)
3. Experiment with policy limits
4. Add your own agents/tools
5. Connect to platform for dashboard

## Resources

- **Documentation**: https://agentsentinel.dev/docs
- **API Reference**: https://agentsentinel.dev/api
- **GitHub**: https://github.com/agent-sentinel/agent-sentinel
- **Discord**: https://discord.gg/agentsentinel

## Contributing

Have a great example to share? Open a PR!

Guidelines:
- Keep examples focused (one concept per file)
- Add comments explaining key points
- Handle missing API keys gracefully
- Show expected output in comments

## License

MIT - See LICENSE file for details
