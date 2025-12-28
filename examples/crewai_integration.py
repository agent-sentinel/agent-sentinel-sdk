"""
Example: CrewAI Integration with AgentSentinel

This example demonstrates how to use SentinelCrew to automatically
track CrewAI crew execution, task completion, and costs.

Requirements:
    pip install agent-sentinel[remote] crewai crewai-tools openai
"""
import os
from crewai import Agent, Task, Crew, Process
from agent_sentinel.integrations.crewai import (
    SentinelCrew,
    wrap_crew_action,
    SentinelAgent,
)


def example_basic_crew():
    """Example 1: Basic crew with automatic tracking."""
    print("\n=== Example 1: Basic Crew with Tracking ===\n")
    
    # Create agents
    researcher = Agent(
        role="Research Analyst",
        goal="Analyze and summarize technical topics",
        backstory="You are an expert technical researcher with deep knowledge.",
        verbose=True,
    )
    
    writer = Agent(
        role="Technical Writer",
        goal="Write clear and engaging technical content",
        backstory="You are a skilled writer who can explain complex topics simply.",
        verbose=True,
    )
    
    # Create tasks
    research_task = Task(
        description="Research the latest developments in quantum computing",
        agent=researcher,
        expected_output="A summary of recent quantum computing advances",
    )
    
    writing_task = Task(
        description="Write a blog post about quantum computing based on the research",
        agent=writer,
        expected_output="A 200-word blog post about quantum computing",
    )
    
    # Create SentinelCrew (wraps CrewAI Crew with tracking)
    crew = SentinelCrew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        run_name="quantum_computing_crew",
        process=Process.sequential,
        verbose=True,
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    # Get execution summary
    summary = crew.get_run_summary()
    print(f"\nCrew Execution Summary:")
    print(f"  - Run Name: {summary['run_name']}")
    print(f"  - Agents: {summary['num_agents']}")
    print(f"  - Tasks: {summary['num_tasks']}")
    print(f"  - Duration: {summary['duration_seconds']:.2f}s")
    print(f"  - Total Cost: ${summary['total_cost_usd']:.6f}")


def example_with_custom_actions():
    """Example 2: Crew with custom tracked actions."""
    print("\n=== Example 2: Crew with Custom Actions ===\n")
    
    # Define custom actions with tracking
    @wrap_crew_action(name="web_search", cost_usd=0.02)
    def search_web(query: str) -> str:
        """Simulated web search with cost tracking."""
        print(f"  [Action] Searching web for: {query}")
        return f"Search results for {query}: [simulated results]"
    
    @wrap_crew_action(name="data_analysis", cost_usd=0.05)
    def analyze_data(data: str) -> str:
        """Simulated data analysis with cost tracking."""
        print(f"  [Action] Analyzing data...")
        return "Analysis complete: Key insights found"
    
    # Create agent that uses these actions
    analyst = Agent(
        role="Data Analyst",
        goal="Gather and analyze data",
        backstory="Expert in data collection and analysis",
        verbose=True,
    )
    
    # Create task
    task = Task(
        description="Search for AI trends and analyze the data",
        agent=analyst,
        expected_output="Analysis report on AI trends",
    )
    
    # Create crew
    crew = SentinelCrew(
        agents=[analyst],
        tasks=[task],
        run_name="data_analysis_crew",
        verbose=True,
    )
    
    # Execute
    result = crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    # Show summary with custom action costs
    summary = crew.get_run_summary()
    print(f"\nSummary:")
    print(f"  - Duration: {summary['duration_seconds']:.2f}s")
    print(f"  - Total Cost: ${summary['total_cost_usd']:.6f}")
    print(f"  - Actions: {summary['action_counts']}")
    print(f"  - Cost Breakdown: {summary['action_costs']}")


def example_sentinel_agent():
    """Example 3: Using SentinelAgent for granular tracking."""
    print("\n=== Example 3: SentinelAgent Granular Tracking ===\n")
    
    # Create base agent
    base_agent = Agent(
        role="Research Assistant",
        goal="Conduct thorough research",
        backstory="Methodical researcher with attention to detail",
        verbose=True,
    )
    
    # Wrap with SentinelAgent for granular tracking
    sentinel_agent = SentinelAgent(
        agent=base_agent,
        agent_id="researcher_001",
        track_actions=True,
        tags=["research", "assistant"],
    )
    
    # Manually track actions
    sentinel_agent.track_action(
        action_name="initialize_research",
        cost_usd=0.01,
        metadata={"topic": "AI safety"},
    )
    
    # Create task
    task = Task(
        description="Research AI safety best practices",
        agent=sentinel_agent.agent,  # Use the wrapped agent
        expected_output="Summary of AI safety best practices",
    )
    
    # Create crew
    crew = SentinelCrew(
        agents=[sentinel_agent.agent],
        tasks=[task],
        run_name="ai_safety_research",
        verbose=True,
    )
    
    result = crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    # Track completion
    sentinel_agent.track_action(
        action_name="finalize_research",
        cost_usd=0.01,
        metadata={"status": "completed"},
    )
    
    summary = crew.get_run_summary()
    print(f"\nSummary:")
    print(f"  - Duration: {summary['duration_seconds']:.2f}s")
    print(f"  - Cost: ${summary['total_cost_usd']:.6f}")


def example_with_approval():
    """Example 4: Crew with human approval for sensitive actions."""
    print("\n=== Example 4: Crew with Human Approval ===\n")
    
    # Define action that requires approval
    @wrap_crew_action(
        name="send_email",
        cost_usd=0.01,
        requires_human_approval=True,
    )
    def send_email(to: str, subject: str, body: str) -> str:
        """Send email - requires human approval."""
        print(f"  [Action] Sending email to {to}")
        return f"Email sent to {to}"
    
    # Create agent
    emailer = Agent(
        role="Email Manager",
        goal="Manage email communications",
        backstory="Handles all email correspondence",
        verbose=True,
    )
    
    # Create task
    task = Task(
        description="Send a follow-up email to the team",
        agent=emailer,
        expected_output="Email sent confirmation",
    )
    
    # Create crew
    crew = SentinelCrew(
        agents=[emailer],
        tasks=[task],
        run_name="email_crew",
        verbose=True,
    )
    
    # Note: In production, this would pause for approval
    # For demo, it will use the default approval handler
    result = crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    summary = crew.get_run_summary()
    print(f"\nSummary:")
    print(f"  - Duration: {summary['duration_seconds']:.2f}s")
    print(f"  - Cost: ${summary['total_cost_usd']:.6f}")


def example_wrap_existing_crew():
    """Example 5: Wrap an existing CrewAI Crew."""
    print("\n=== Example 5: Wrap Existing Crew ===\n")
    
    from agent_sentinel.integrations.crewai import wrap_existing_crew
    
    # Create standard CrewAI crew
    agent = Agent(
        role="Assistant",
        goal="Help with tasks",
        backstory="Helpful AI assistant",
    )
    
    task = Task(
        description="Write a haiku about Python",
        agent=agent,
        expected_output="A Python haiku",
    )
    
    # Standard CrewAI Crew
    standard_crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )
    
    # Wrap it with Sentinel tracking
    sentinel_crew = wrap_existing_crew(
        crew=standard_crew,
        run_name="wrapped_crew_example",
    )
    
    # Execute with tracking
    result = sentinel_crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    summary = sentinel_crew.get_run_summary()
    print(f"\nSummary:")
    print(f"  - Duration: {summary['duration_seconds']:.2f}s")
    print(f"  - Cost: ${summary['total_cost_usd']:.6f}")


def example_with_remote_sync():
    """Example 6: Crew execution synced to platform."""
    print("\n=== Example 6: With Platform Sync ===\n")
    
    from agent_sentinel import enable_remote_sync, flush_and_stop
    
    # Enable remote sync
    enable_remote_sync(
        api_url=os.getenv("AGENTSENTINEL_API_URL", "http://localhost:8000"),
        api_key=os.getenv("AGENTSENTINEL_API_KEY", "dev-key"),
    )
    
    # Create crew
    agent = Agent(
        role="Content Creator",
        goal="Create engaging content",
        backstory="Creative content specialist",
    )
    
    task = Task(
        description="Write a tweet about AgentSentinel",
        agent=agent,
        expected_output="A compelling tweet",
    )
    
    crew = SentinelCrew(
        agents=[agent],
        tasks=[task],
        run_name="tweet_creator_crew",
        verbose=True,
    )
    
    result = crew.kickoff()
    
    print(f"\n\nResult:\n{result}\n")
    
    # Flush to platform
    print("\nSyncing to platform...")
    flush_and_stop()
    print("Done! Check your AgentSentinel dashboard.")


if __name__ == "__main__":
    # Make sure OpenAI API key is set (CrewAI uses OpenAI by default)
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        exit(1)
    
    # Run examples
    try:
        example_basic_crew()
        # example_with_custom_actions()  # Uncomment to run
        # example_sentinel_agent()  # Uncomment to run
        # example_with_approval()  # Uncomment to run
        # example_wrap_existing_crew()  # Uncomment to run
        # example_with_remote_sync()  # Uncomment to run
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


