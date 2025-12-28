"""
Example: LangChain Integration with AgentSentinel

This example demonstrates how to use the SentinelCallbackHandler
to automatically track LangChain chains, agents, and LLM costs.

Requirements:
    pip install agent-sentinel[remote] langchain langchain-openai openai
"""
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agent_sentinel.integrations.langchain import SentinelCallbackHandler
from agent_sentinel.cost import CostTracker


def example_basic_chain():
    """Example 1: Basic LLM chain with cost tracking."""
    print("\n=== Example 1: Basic LLM Chain ===\n")
    
    # Create the callback handler
    sentinel = SentinelCallbackHandler(
        run_name="basic_chain_example",
        track_costs=True,
    )
    
    # Create LLM with callback
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    # Create a simple chain
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short haiku about {topic}."
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    result = chain.run(topic="artificial intelligence")
    
    print(f"Result: {result}\n")
    
    # Get summary
    summary = sentinel.get_run_summary()
    print(f"Run Summary:")
    print(f"  - Duration: {summary['session_duration_seconds']:.2f}s")
    print(f"  - Cost: ${summary['run_cost_usd']:.6f}")
    print(f"  - Actions: {summary['action_counts']}")


def example_agent_with_tools():
    """Example 2: Agent with tools and automatic cost tracking."""
    print("\n=== Example 2: Agent with Tools ===\n")
    
    # Define some simple tools
    def search_tool(query: str) -> str:
        """Simulated search tool."""
        return f"Search results for: {query}"
    
    def calculator_tool(expression: str) -> str:
        """Simple calculator tool."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="Search for information"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Perform calculations"
        ),
    ]
    
    # Create callback handler
    sentinel = SentinelCallbackHandler(
        run_name="agent_example",
        track_costs=True,
        track_tools=True,
    )
    
    # Create agent
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[sentinel],
        verbose=True,
    )
    
    # Run agent
    result = agent.run(
        "What is 15 * 23, and then search for information about that number?"
    )
    
    print(f"\nFinal Answer: {result}\n")
    
    # Get summary
    summary = sentinel.get_run_summary()
    print(f"Agent Summary:")
    print(f"  - Duration: {summary['session_duration_seconds']:.2f}s")
    print(f"  - Total Cost: ${summary['run_cost_usd']:.6f}")
    print(f"  - Action Counts: {summary['action_counts']}")
    print(f"  - Cost Breakdown: {summary['action_costs']}")


def example_multiple_llm_calls():
    """Example 3: Multiple LLM calls with aggregated cost tracking."""
    print("\n=== Example 3: Multiple LLM Calls ===\n")
    
    # Reset cost tracker for this example
    CostTracker.reset_run()
    
    # Create callback handler
    sentinel = SentinelCallbackHandler(
        run_name="multi_call_example",
        track_costs=True,
    )
    
    # Create LLM
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    # Make multiple calls
    topics = ["Python", "JavaScript", "Rust"]
    
    for topic in topics:
        print(f"\nGenerating haiku about {topic}...")
        prompt = f"Write a haiku about {topic} programming language."
        response = llm.predict(prompt)
        print(f"  {response}")
    
    # Get final summary
    summary = sentinel.get_run_summary()
    print(f"\n\nFinal Summary:")
    print(f"  - Total Calls: {sum(summary['action_counts'].values())}")
    print(f"  - Total Cost: ${summary['run_cost_usd']:.6f}")
    print(f"  - Average Cost per Call: ${summary['run_cost_usd'] / len(topics):.6f}")


def example_with_different_models():
    """Example 4: Compare costs across different models."""
    print("\n=== Example 4: Cost Comparison Across Models ===\n")
    
    models = [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
    ]
    
    prompt_text = "Explain quantum computing in one sentence."
    
    for model in models:
        # Reset for each model
        CostTracker.reset_run()
        
        sentinel = SentinelCallbackHandler(
            run_name=f"model_comparison_{model}",
            track_costs=True,
        )
        
        llm = ChatOpenAI(
            temperature=0,
            model=model,
            callbacks=[sentinel]
        )
        
        print(f"\nTesting {model}...")
        response = llm.predict(prompt_text)
        
        summary = sentinel.get_run_summary()
        print(f"  - Response: {response[:100]}...")
        print(f"  - Cost: ${summary['run_cost_usd']:.6f}")


def example_with_remote_sync():
    """Example 5: LangChain with remote sync to platform."""
    print("\n=== Example 5: With Remote Platform Sync ===\n")
    
    from agent_sentinel import enable_remote_sync, flush_and_stop
    
    # Enable remote sync (requires AGENTSENTINEL_API_KEY env var)
    enable_remote_sync(
        api_url=os.getenv("AGENTSENTINEL_API_URL", "http://localhost:8000"),
        api_key=os.getenv("AGENTSENTINEL_API_KEY", "dev-key"),
    )
    
    # Create callback handler
    sentinel = SentinelCallbackHandler(
        run_name="langchain_with_sync",
        track_costs=True,
    )
    
    # Create and run chain
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    response = llm.predict("What are the three laws of robotics?")
    print(f"Response: {response}\n")
    
    summary = sentinel.get_run_summary()
    print(f"Cost: ${summary['run_cost_usd']:.6f}")
    
    # Flush logs to platform
    print("\nFlushing logs to platform...")
    flush_and_stop()
    print("Done! Check your AgentSentinel dashboard.")


if __name__ == "__main__":
    # Make sure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        exit(1)
    
    # Run examples
    try:
        example_basic_chain()
        # example_agent_with_tools()  # Uncomment to run
        # example_multiple_llm_calls()  # Uncomment to run
        # example_with_different_models()  # Uncomment to run
        # example_with_remote_sync()  # Uncomment to run
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


