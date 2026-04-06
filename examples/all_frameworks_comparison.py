"""
Quick Start: Agent Sentinel with All Three Frameworks

This example shows how to use Agent Sentinel with:
- AutoGen (Microsoft)
- LangChain
- CrewAI

Run each example independently or see how they compare.
"""

# ============================================================================
# EXAMPLE 1: AutoGen (Simplest)
# ============================================================================

def autogen_example():
    """
    AutoGen integration - Simplest of the three.
    Uses native hook system (register_reply).
    """
    print("=" * 80)
    print("AutoGen Example")
    print("=" * 80)
    
    try:
        from autogen import AssistantAgent, UserProxyAgent
        from agent_sentinel.integrations.autogen import SentinelInspector
        from agent_sentinel.policy import PolicyEngine
        
        # 1. Configure policy
        PolicyEngine.configure(run_budget=0.10)
        
        # 2. Create AutoGen agents (standard code)
        assistant = AssistantAgent(
            name="assistant",
            llm_config={"model": "gpt-4", "api_key": "your-api-key"}
        )
        
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER"
        )
        
        # 3. Secure with Sentinel (one line per agent!)
        sentinel = SentinelInspector(
            run_name="autogen_demo",
            enforce_policies=True
        )
        sentinel.register(assistant)
        sentinel.register(user_proxy)
        
        # 4. Run as normal
        sentinel.start_run()
        user_proxy.initiate_chat(
            assistant,
            message="What is 2+2?"
        )
        sentinel.end_run()
        
        # 5. Review results
        summary = sentinel.get_run_summary()
        print(f"\n✅ Cost: ${summary['run_cost_usd']:.6f}")
        print(f"✅ Messages: {summary['message_count']}")
        print(f"✅ LLM Calls: {summary['llm_call_count']}")
        
    except ImportError as e:
        print(f"⚠️  AutoGen not installed: {e}")
        print("   Install with: pip install pyautogen")


# ============================================================================
# EXAMPLE 2: LangChain (Callback-based)
# ============================================================================

def langchain_example():
    """
    LangChain integration - Callback handler pattern.
    Must propagate callbacks through chain.
    """
    print("\n" + "=" * 80)
    print("LangChain Example")
    print("=" * 80)
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_openai_functions_agent, AgentExecutor
        from langchain.tools import tool
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from agent_sentinel.integrations.langchain import SentinelCallbackHandler
        from agent_sentinel.policy import PolicyEngine
        
        # 1. Configure policy
        PolicyEngine.configure(run_budget=0.10)
        
        # 2. Create callback handler
        handler = SentinelCallbackHandler(
            run_name="langchain_demo",
            enforce_policies=True
        )
        
        # 3. Define tools
        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression"""
            try:
                return str(eval(expression))
            except:
                return "Error in calculation"
        
        # 4. Create agent with callbacks (MUST propagate!)
        llm = ChatOpenAI(
            model="gpt-4",
            callbacks=[handler]  # Callback here
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=llm,
            tools=[calculate],
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=[calculate],
            callbacks=[handler],  # And here
            verbose=True
        )
        
        # 5. Run with callbacks
        result = executor.invoke(
            {"input": "What is 15 * 23?"},
            config={"callbacks": [handler]}  # And here!
        )
        
        # 6. Review results
        summary = handler.get_run_summary()
        print(f"\n✅ Cost: ${summary['total_cost_usd']:.6f}")
        print(f"✅ Result: {result['output']}")
        
    except ImportError as e:
        print(f"⚠️  LangChain not installed: {e}")
        print("   Install with: pip install langchain langchain-openai")


# ============================================================================
# EXAMPLE 3: CrewAI (Wrapper-based)
# ============================================================================

def crewai_example():
    """
    CrewAI integration - Wrapper pattern.
    Wraps Crew, Agent, and Task objects.
    """
    print("\n" + "=" * 80)
    print("CrewAI Example")
    print("=" * 80)
    
    try:
        from crewai import Agent, Task
        from agent_sentinel.integrations.crewai import SentinelCrew
        from agent_sentinel.policy import PolicyEngine
        
        # 1. Configure policy
        PolicyEngine.configure(run_budget=0.10)
        
        # 2. Create agents (standard CrewAI)
        researcher = Agent(
            role="Researcher",
            goal="Research topics thoroughly",
            backstory="Expert researcher with deep knowledge",
            verbose=True
        )
        
        writer = Agent(
            role="Writer",
            goal="Write compelling content",
            backstory="Professional writer with clear style",
            verbose=True
        )
        
        # 3. Define tasks
        research_task = Task(
            description="Research the benefits of AI",
            agent=researcher,
            expected_output="Research findings"
        )
        
        writing_task = Task(
            description="Write a short article based on research",
            agent=writer,
            expected_output="Article content",
            context=[research_task]
        )
        
        # 4. Create SentinelCrew (wraps standard Crew)
        crew = SentinelCrew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            run_name="crewai_demo",
            enforce_policies=True,
            verbose=True
        )
        
        # 5. Run (same API as standard Crew)
        result = crew.kickoff()
        
        # 6. Review results
        summary = crew.get_run_summary()
        print(f"\n✅ Cost: ${summary['run_cost_usd']:.6f}")
        print(f"✅ Duration: {summary['duration_seconds']:.2f}s")
        
    except ImportError as e:
        print(f"⚠️  CrewAI not installed: {e}")
        print("   Install with: pip install crewai")


# ============================================================================
# COMPARISON
# ============================================================================

def print_comparison():
    """Print a comparison of all three integrations."""
    print("\n" + "=" * 80)
    print("Framework Comparison")
    print("=" * 80)
    
    print("""
┌──────────────┬────────────────────┬────────────┬─────────────────────┐
│ Framework    │ Integration        │ Complexity │ Lines to Add        │
├──────────────┼────────────────────┼────────────┼─────────────────────┤
│ AutoGen      │ Hook-based         │ ⭐ Simple  │ 3 lines             │
│ LangChain    │ Callback           │ ⭐⭐ Medium│ 4-5 lines           │
│ CrewAI       │ Wrapper            │ ⭐⭐⭐ High │ Change 1 import     │
└──────────────┴────────────────────┴────────────┴─────────────────────┘

AutoGen Pattern (Hook):
    sentinel.register(agent)  # One line per agent

LangChain Pattern (Callback):
    handler = SentinelCallbackHandler(...)
    llm = ChatOpenAI(callbacks=[handler])
    agent = create_agent(..., callbacks=[handler])
    executor.invoke(..., config={"callbacks": [handler]})

CrewAI Pattern (Wrapper):
    # Change: from crewai import Crew
    from agent_sentinel.integrations.crewai import SentinelCrew
    crew = SentinelCrew(...)  # Same API

✅ All three provide:
   - Active policy enforcement
   - Cost tracking
   - Action monitoring
   - Dashboard integration
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples and show comparison."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║               Agent Sentinel - Framework Integration Examples           ║
║                                                                          ║
║  Compare how Agent Sentinel works with the "Big Three" frameworks:      ║
║  - AutoGen (Microsoft)                                                   ║
║  - LangChain                                                            ║
║  - CrewAI                                                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Run each example
    # Note: These will fail with "API key" errors if you don't have real keys,
    # but they demonstrate the integration patterns
    
    print("\nNote: Examples require API keys to run fully.")
    print("They demonstrate the integration patterns even without keys.\n")
    
    try:
        autogen_example()
    except Exception as e:
        print(f"AutoGen example error: {e}")
    
    try:
        langchain_example()
    except Exception as e:
        print(f"LangChain example error: {e}")
    
    try:
        crewai_example()
    except Exception as e:
        print(f"CrewAI example error: {e}")
    
    # Print comparison
    print_comparison()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
All three frameworks are fully supported!

Choose based on your needs:
- AutoGen: Best for multi-agent conversations
- LangChain: Best for tool chains and RAG
- CrewAI: Best for role-based agent teams

All provide the same core features:
✅ Active policy enforcement
✅ Real-time cost tracking
✅ Budget limits
✅ Rate limiting
✅ Dashboard integration

For full documentation, visit: https://agentsentinel.dev/docs
""")


if __name__ == "__main__":
    main()
