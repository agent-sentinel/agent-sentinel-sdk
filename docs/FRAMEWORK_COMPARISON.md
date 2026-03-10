# Framework Integration Comparison

Agent Sentinel supports the "Big Three" agent frameworks with tailored integrations:

## Quick Comparison

| Framework | Integration Pattern | Complexity | Best For |
|-----------|-------------------|------------|----------|
| **AutoGen** | Hook-based (`register_reply`) | ⭐ Simple | Microsoft shops, multi-agent conversations |
| **LangChain** | Callback handler | ⭐⭐ Medium | Tool chains, sequential workflows |
| **CrewAI** | Wrapper class | ⭐⭐⭐ Complex | Role-based teams, task orchestration |

## AutoGen Integration

**Pattern**: Hook-based (uses AutoGen's built-in `register_reply`)

**Why Simple**: AutoGen has a native hook system designed for this exact use case.

```python
from agent_sentinel.integrations.autogen import SentinelInspector

sentinel = SentinelInspector(run_name="my_run")
sentinel.register(assistant)  # One line per agent
sentinel.register(user_proxy)
```

**What's Tracked**:
- Agent-to-agent messages
- LLM token costs
- Message flow and sequence
- Policy violations

**Best For**:
- Multi-agent conversations
- Enterprise/Microsoft environments
- Complex agent orchestration
- Research and experimentation

---

## LangChain Integration

**Pattern**: Callback handler (uses LangChain's callback system)

**Why Medium**: Requires callback propagation through chains/agents/tools.

```python
from agent_sentinel.integrations.langchain import SentinelCallbackHandler

handler = SentinelCallbackHandler(run_name="my_run")

# Must pass callbacks at multiple levels
llm = ChatOpenAI(callbacks=[handler])
agent = create_agent(llm, tools, callbacks=[handler])
executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
result = executor.invoke(input, config={"callbacks": [handler]})
```

**What's Tracked**:
- LLM calls (with token usage)
- Tool invocations
- Chain execution
- Agent actions and reasoning

**Best For**:
- Sequential workflows
- Tool chains
- RAG applications
- Question answering systems

---

## CrewAI Integration

**Pattern**: Wrapper class (wraps Crew/Agent/Task objects)

**Why Complex**: Requires wrapping multiple objects and intercepting tool execution.

```python
from agent_sentinel.integrations.crewai import SentinelCrew

crew = SentinelCrew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    run_name="my_run"
)
result = crew.kickoff()
```

**What's Tracked**:
- Agent actions
- Task execution
- Tool calls
- LLM costs
- Runaway agent detection

**Best For**:
- Role-based agent teams
- Content generation
- Research workflows
- Task delegation patterns

---

## Feature Comparison

| Feature | AutoGen | LangChain | CrewAI |
|---------|---------|-----------|---------|
| **Policy Enforcement** | ✅ Active | ✅ Active | ✅ Active |
| **Cost Tracking** | ✅ Token-level | ✅ Token-level | ✅ Action-level |
| **Message Tracking** | ✅ Full | ⚠️ Via chains | ⚠️ Via tasks |
| **Tool Wrapping** | Manual | Automatic | Automatic |
| **Multi-Agent** | ✅ Native | ⚠️ Via agents | ✅ Native |
| **Async Support** | ✅ Yes | ✅ Yes | ⚠️ Limited |
| **Integration Effort** | 🟢 Low | 🟡 Medium | 🔴 High |

---

## Code Examples

### AutoGen: Simplest

```python
from autogen import AssistantAgent
from agent_sentinel.integrations.autogen import SentinelInspector

sentinel = SentinelInspector(run_name="autogen_run")
sentinel.register(assistant)  # One line!

sentinel.start_run()
user_proxy.initiate_chat(assistant, message="...")
sentinel.end_run()
```

### LangChain: Callback Propagation

```python
from langchain.agents import create_openai_functions_agent
from agent_sentinel.integrations.langchain import SentinelCallbackHandler

handler = SentinelCallbackHandler(run_name="langchain_run")

# Must propagate callbacks
llm = ChatOpenAI(callbacks=[handler])
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[handler]  # Don't forget!
)

result = executor.invoke(
    {"input": "..."},
    config={"callbacks": [handler]}  # And here!
)
```

### CrewAI: Wrapper Pattern

```python
from crewai import Agent, Task
from agent_sentinel.integrations.crewai import SentinelCrew

# Create agents and tasks
researcher = Agent(role="Researcher", ...)
writer = Agent(role="Writer", ...)
research_task = Task(description="...", agent=researcher)
writing_task = Task(description="...", agent=writer)

# Wrap in SentinelCrew
crew = SentinelCrew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    run_name="crewai_run"
)

result = crew.kickoff()
```

---

## When to Use What?

### Use AutoGen When:
- ✅ Building multi-agent conversations
- ✅ Working in Microsoft environments
- ✅ Need agent-to-agent communication tracking
- ✅ Want simplest integration

### Use LangChain When:
- ✅ Building sequential workflows
- ✅ Creating tool chains
- ✅ Need extensive ecosystem (vector stores, embeddings, etc.)
- ✅ Want structured callback system

### Use CrewAI When:
- ✅ Building role-based teams
- ✅ Need task delegation
- ✅ Creating content generation pipelines
- ✅ Want agent specialization

---

## Migration Path

Already using one framework? Easy to add Sentinel:

### Existing AutoGen Code

```python
# Before (your existing code)
assistant = AssistantAgent(...)
user_proxy = UserProxyAgent(...)
user_proxy.initiate_chat(assistant, message="...")

# After (add 3 lines)
from agent_sentinel.integrations.autogen import SentinelInspector

sentinel = SentinelInspector(run_name="my_run")
sentinel.register(assistant)
sentinel.register(user_proxy)

# Rest unchanged
user_proxy.initiate_chat(assistant, message="...")
```

### Existing LangChain Code

```python
# Before
llm = ChatOpenAI(...)
agent = create_agent(llm, tools, prompt)
result = agent.invoke(...)

# After (add handler)
from agent_sentinel.integrations.langchain import SentinelCallbackHandler

handler = SentinelCallbackHandler(run_name="my_run")
llm = ChatOpenAI(callbacks=[handler])
agent = create_agent(llm, tools, prompt)
result = agent.invoke(..., config={"callbacks": [handler]})
```

### Existing CrewAI Code

```python
# Before
crew = Crew(agents=[...], tasks=[...])
result = crew.kickoff()

# After (change import)
from agent_sentinel.integrations.crewai import SentinelCrew

crew = SentinelCrew(  # Changed from Crew
    agents=[...],
    tasks=[...],
    run_name="my_run"
)
result = crew.kickoff()  # Same API
```

---

## Combining Integrations

You can use multiple integrations together:

```python
from agent_sentinel.integrations import (
    instrument_openai,  # Low-level instrumentation
    SentinelCallbackHandler,  # LangChain
    SentinelInspector,  # AutoGen
)

# Global instrumentation (tracks all OpenAI calls)
instrument_openai()

# Framework-specific tracking (adds structure)
langchain_handler = SentinelCallbackHandler(run_name="langchain_part")
autogen_sentinel = SentinelInspector(run_name="autogen_part")

# Use both in your application
# LangChain for tool chains
llm = ChatOpenAI(callbacks=[langchain_handler])
# ...

# AutoGen for multi-agent
autogen_sentinel.register(assistant)
# ...
```

---

## Performance Impact

| Framework | Integration Overhead | Notes |
|-----------|---------------------|-------|
| AutoGen | < 1ms per message | Hook is very lightweight |
| LangChain | < 1ms per callback | Depends on callback frequency |
| CrewAI | < 5ms per action | Wrapper has some overhead |

All integrations are designed to fail-open: if logging fails, your agents continue working.

---

## Support Matrix

| Framework Version | AutoGen | LangChain | CrewAI |
|-------------------|---------|-----------|---------|
| Latest | ✅ | ✅ | ✅ |
| 1.x | ✅ | ✅ | ✅ |
| 0.x | ⚠️ | ✅ | ⚠️ |

✅ = Fully supported  
⚠️ = Partial support  
❌ = Not supported

---

## Getting Help

- **Documentation**: [agentsentinel.dev/docs](https://agentsentinel.dev/docs)
- **Examples**: See `examples/` directory for each framework
- **Issues**: [GitHub Issues](https://github.com/agent-sentinel/agent-sentinel/issues)
- **Discord**: [Join our community](https://discord.gg/agentsentinel)

---

## Summary

**AutoGen = Simplest** (hook-based)  
**LangChain = Most Structured** (callback-based)  
**CrewAI = Most Specialized** (wrapper-based)

Choose based on your framework, not the integration difficulty. All three are production-ready and battle-tested.
