# AgentSentinel Framework Integrations

Out-of-the-box integrations for popular AI frameworks and LLM providers.

## Overview

AgentSentinel provides seamless integrations with major AI frameworks and LLM providers, enabling automatic cost tracking, action monitoring, and governance without modifying your application code.

### Supported Integrations

- **LangChain** - Callback handler for chains, agents, and tools
- **CrewAI** - Wrapper for crew actions and task execution
- **OpenAI** - Automatic cost tracking for GPT models
- **Anthropic** - Automatic cost tracking for Claude models
- **Grok/xAI** - Automatic cost tracking for Grok models
- **Google Gemini** - Automatic cost tracking for Gemini models

## Installation

### Basic Installation

```bash
pip install agent-sentinel
```

### With Integration Dependencies

Install specific integrations as needed:

```bash
# LangChain integration
pip install agent-sentinel langchain langchain-openai

# CrewAI integration
pip install agent-sentinel crewai crewai-tools

# LLM instrumentation (all providers)
pip install agent-sentinel openai anthropic google-generativeai

# Everything (all integrations)
pip install agent-sentinel[remote] langchain crewai openai anthropic google-generativeai
```

## Quick Start

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from agent_sentinel.integrations.langchain import SentinelCallbackHandler

# Create callback handler
sentinel = SentinelCallbackHandler(
    run_name="my_agent_run",
    track_costs=True
)

# Use with LangChain
llm = ChatOpenAI(temperature=0, callbacks=[sentinel])
response = llm.predict("What is Python?")

# Get cost summary
summary = sentinel.get_run_summary()
print(f"Cost: ${summary['run_cost_usd']:.6f}")
```

### CrewAI Integration

```python
from crewai import Agent, Task
from agent_sentinel.integrations.crewai import SentinelCrew

# Create agents and tasks
researcher = Agent(
    role="Research Analyst",
    goal="Analyze topics",
    backstory="Expert researcher"
)

task = Task(
    description="Research AI trends",
    agent=researcher,
    expected_output="Analysis report"
)

# Create SentinelCrew (automatically tracks execution)
crew = SentinelCrew(
    agents=[researcher],
    tasks=[task],
    run_name="ai_research"
)

# Execute with automatic tracking
result = crew.kickoff()
summary = crew.get_run_summary()
```

### LLM Instrumentation

```python
from agent_sentinel.integrations.llm import instrument_openai, get_token_costs
import openai

# Enable automatic tracking (one-time setup)
instrument_openai()

# Use OpenAI normally - costs tracked automatically!
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Get cost summary
costs = get_token_costs()
print(f"Total cost: ${costs['total_usd']:.6f}")
print(f"By model: {costs['by_model']}")
```

## Features

### LangChain Integration

**SentinelCallbackHandler** tracks:
- ✅ LLM calls with token usage and costs
- ✅ Tool/action executions
- ✅ Chain starts/ends
- ✅ Agent actions
- ✅ Errors and retries
- ✅ Run summaries with cost breakdowns

**Features:**
- Automatic token cost calculation
- Support for all OpenAI models
- Chain and agent tracking
- Tool execution monitoring
- Cost aggregation by action type

### CrewAI Integration

**SentinelCrew** provides:
- ✅ Crew execution tracking
- ✅ Task completion monitoring
- ✅ Agent action tracking
- ✅ Cost tracking per task
- ✅ Human approval workflows

**Additional utilities:**
- `@wrap_crew_action` - Decorator for custom actions
- `SentinelAgent` - Granular agent tracking
- `wrap_existing_crew()` - Wrap existing crews

### LLM Instrumentation

**Supported Providers:**
- ✅ **OpenAI** - GPT-3.5, GPT-4, GPT-4o, o1 models
- ✅ **Anthropic** - Claude 3 Opus, Sonnet, Haiku
- ✅ **Grok/xAI** - Grok-1, Grok-beta
- ✅ **Gemini** - Gemini 1.5 Pro, Flash

**Token Pricing:**
- Accurate pricing data for all major models
- Automatic token cost calculation
- Real-time cost tracking
- Multi-provider cost aggregation

## Advanced Usage

### Budget Enforcement with LLM Instrumentation

```python
from agent_sentinel import PolicyEngine, PolicyConfig, guarded_action
from agent_sentinel.integrations.llm import instrument_openai
import openai

# Configure budget
policy = PolicyConfig(
    budget_limits={
        "action": 0.01,  # Max $0.01 per action
        "run": 0.10,     # Max $0.10 per run
    }
)
PolicyEngine.set_policy(policy)

# Instrument OpenAI
instrument_openai()
client = openai.OpenAI()

# Wrap calls with budget enforcement
@guarded_action(name="gpt_call", cost_usd=0.0)
def call_gpt(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Automatically enforces budget limits!
try:
    result = call_gpt("Explain quantum computing")
except Exception as e:
    print(f"Budget exceeded: {e}")
```

### Multi-Provider Cost Tracking

```python
from agent_sentinel.integrations.llm import (
    instrument_openai,
    instrument_anthropic,
    instrument_gemini,
    get_token_costs,
)

# Instrument all providers
instrument_openai()
instrument_anthropic()
instrument_gemini()

# Use all providers in your app...
# Costs are automatically tracked!

# Get unified cost summary
costs = get_token_costs()
print(f"Total: ${costs['total_usd']:.6f}")
print(f"By provider: {costs['by_provider']}")
print(f"By model: {costs['by_model']}")
```

### LangChain with Human Approval

```python
from langchain.agents import Tool
from agent_sentinel.integrations.langchain import SentinelCallbackHandler
from agent_sentinel import guarded_action

@guarded_action(
    name="send_email",
    requires_human_approval=True,
    approval_description="Send customer email"
)
def send_email(to: str, subject: str) -> str:
    # This will pause and request approval
    # (requires ApprovalClient configuration)
    return f"Email sent to {to}"

# Tool will automatically request approval when used
tool = Tool(
    name="SendEmail",
    func=send_email,
    description="Send an email"
)

# Use in agent...
```

### CrewAI with Custom Actions

```python
from agent_sentinel.integrations.crewai import wrap_crew_action

@wrap_crew_action(name="web_search", cost_usd=0.02)
def search_web(query: str) -> str:
    # Automatically tracked with $0.02 cost
    return perform_search(query)

@wrap_crew_action(
    name="database_update",
    cost_usd=0.05,
    requires_human_approval=True
)
def update_database(query: str) -> str:
    # Requires approval before execution
    return execute_query(query)
```

## Token Pricing

All integrations use accurate, up-to-date pricing data (as of December 2025):

### OpenAI Pricing (per 1M tokens)
- GPT-4o: $5.00 input / $15.00 output
- GPT-4o-mini: $0.15 input / $0.60 output
- GPT-4: $30.00 input / $60.00 output
- GPT-3.5-turbo: $0.50 input / $1.50 output

### Anthropic Pricing (per 1M tokens)
- Claude 3.5 Sonnet: $3.00 input / $15.00 output
- Claude 3 Opus: $15.00 input / $75.00 output
- Claude 3 Haiku: $0.25 input / $1.25 output

### Gemini Pricing (per 1M tokens)
- Gemini 1.5 Pro: $3.50 input / $10.50 output
- Gemini 1.5 Flash: $0.075 input / $0.30 output

*Pricing data is automatically maintained in the SDK.*

## Examples

See the `/examples` directory for complete working examples:

- `langchain_integration.py` - LangChain examples with multiple use cases
- `crewai_integration.py` - CrewAI crew tracking and custom actions
- `llm_instrumentation.py` - Multi-provider LLM cost tracking

Run an example:

```bash
export OPENAI_API_KEY='your-key'
python examples/langchain_integration.py
```

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export GOOGLE_API_KEY='your-gemini-key'
export XAI_API_KEY='your-grok-key'

# AgentSentinel Platform (optional)
export AGENTSENTINEL_API_KEY='your-sentinel-key'
export AGENTSENTINEL_API_URL='https://api.agentsentinel.dev'
```

### Remote Sync

Enable remote sync to the AgentSentinel platform:

```python
from agent_sentinel import enable_remote_sync, flush_and_stop

enable_remote_sync(
    api_url="https://api.agentsentinel.dev",
    api_key="your-api-key"
)

# Your code with integrations...

# Flush logs before exit
flush_and_stop()
```

## API Reference

### LangChain

#### `SentinelCallbackHandler`

```python
SentinelCallbackHandler(
    run_name: Optional[str] = None,
    track_costs: bool = True,
    track_tools: bool = True,
    auto_log: bool = True,
    tags: Optional[List[str]] = None,
)
```

#### `create_sentinel_handler()`

Convenience function for quick setup.

### CrewAI

#### `SentinelCrew`

```python
SentinelCrew(
    agents: List[Agent],
    tasks: List[Task],
    run_name: Optional[str] = None,
    track_costs: bool = True,
    track_tasks: bool = True,
    auto_log: bool = True,
    tags: Optional[List[str]] = None,
    **crew_kwargs
)
```

#### `@wrap_crew_action`

```python
@wrap_crew_action(
    name: Optional[str] = None,
    cost_usd: float = 0.0,
    tags: Optional[List[str]] = None,
    requires_human_approval: bool = False,
)
```

### LLM Instrumentation

#### `instrument_openai()`
```python
instrument_openai(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
)
```

#### `instrument_anthropic()`
```python
instrument_anthropic(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
)
```

#### `instrument_grok()`
```python
instrument_grok(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
)
```

#### `instrument_gemini()`
```python
instrument_gemini(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
)
```

#### `get_token_costs()`
```python
get_token_costs() -> Dict[str, Any]
```

Returns cost breakdown by provider and model.

## Best Practices

1. **Instrument Early** - Call `instrument_*()` at the start of your application
2. **Use Budget Limits** - Set reasonable budget limits to prevent runaway costs
3. **Tag Appropriately** - Use tags to categorize and filter actions
4. **Monitor Costs** - Regularly check `get_token_costs()` during development
5. **Enable Remote Sync** - Sync to platform for dashboards and alerts

## Troubleshooting

### No Costs Tracked

- Ensure you've called `instrument_*()` before using the LLM client
- Check that the model name matches pricing database
- Verify token usage is returned by the API

### Import Errors

- Install the required dependencies for your integration
- LangChain requires: `pip install langchain`
- CrewAI requires: `pip install crewai`
- LLMs require their respective SDKs

### Cost Mismatch

- Verify pricing data is up-to-date
- Check model name normalization
- Submit issue if pricing is incorrect

## Contributing

Found a bug or want to add a new integration? See [CONTRIBUTING.md](../CONTRIBUTING.md).

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- Documentation: https://docs.agentsentinel.dev
- GitHub Issues: https://github.com/agent-sentinel/agent-sentinel/issues
- Discord: https://discord.gg/agentsentinel


