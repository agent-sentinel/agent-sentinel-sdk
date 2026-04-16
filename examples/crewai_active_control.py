"""
Example: CrewAI Active Control Integration with AgentSentinel

This example demonstrates the upgraded "Visa-like" active control integration
that automatically secures all tools, monitors LLM costs, and prevents runaway agents.

Key Features Demonstrated:
1. Automatic Tool Injection - Tools are secured without manual decoration
2. LLM Cost Monitoring - Token costs tracked and budgets enforced in real-time
3. Runaway Agent Detection - Step limits and loop detection prevent infinite loops
4. Policy Enforcement - Blocked actions stop execution BEFORE spending/damage

Requirements:
    pip install agent-sentinel[remote] crewai crewai-tools openai
"""
import os
from crewai import Agent, Task, Process
from agent_sentinel.integrations.crewai import SentinelCrew
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.cost import CostTracker


def example_automatic_tool_security():
    """
    Example 1: Automatic Tool Injection (The "Chip Reader")
    
    Demonstrates how SentinelCrew automatically secures tools without manual decoration.
    Even if you forget to wrap a tool, it's automatically protected.
    """
    print("\n" + "="*80)
    print("Example 1: Automatic Tool Security (Chip Reader)")
    print("="*80 + "\n")
    
    try:
        # Import a standard CrewAI tool (no manual wrapping needed!)
        from crewai_tools import SerperDevTool, WebsiteSearchTool
        
        # Configure policy to limit tool usage
        PolicyEngine.configure(
            run_budget=0.10,  # $0.10 budget for this run
            denied_actions=["dangerous_operation"],  # Example blocked action
            strict_mode=True,
        )
        
        print("✓ Policy configured: $0.10 run budget\n")
        
        # Create agent with standard tools - NO MANUAL WRAPPING!
        researcher = Agent(
            role="Research Analyst",
            goal="Find information on the web",
            backstory="Expert researcher who uses search tools",
            tools=[
                SerperDevTool(),  # Auto-secured by SentinelCrew!
                WebsiteSearchTool(),  # Also auto-secured!
            ],
            verbose=True,
        )
        
        task = Task(
            description="Search for 'AgentSentinel AI monitoring' and summarize findings",
            agent=researcher,
            expected_output="A brief summary of search results",
        )
        
        print("✓ Agent created with SerperDevTool and WebsiteSearchTool")
        print("✓ Tools will be automatically secured by SentinelCrew\n")
        
        # SentinelCrew automatically secures all tools!
        crew = SentinelCrew(
            agents=[researcher],
            tasks=[task],
            run_name="tool_security_demo",
            enforce_policies=True,  # Active blocking enabled
            max_agent_steps=20,     # Prevent runaway
            verbose=True,
        )
        
        print("\n🔒 SentinelCrew secured tools automatically!")
        print("   - SerperDevTool will check PolicyEngine before each search")
        print("   - WebsiteSearchTool will check budget before scraping")
        print("   - No manual @wrap_crew_action decorator needed!\n")
        
        # Execute - tools are now protected
        result = crew.kickoff()
        
        print(f"\n✅ Result:\n{result}\n")
        
        # Show cost summary
        summary = crew.get_run_summary()
        print(f"\n📊 Execution Summary:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Total Cost: ${summary['total_cost_usd']:.6f}")
        print(f"   Budget Remaining: ${0.10 - summary['total_cost_usd']:.6f}\n")
        
    except ImportError:
        print("⚠️  crewai-tools not installed. Install with: pip install crewai-tools")
        print("   This example demonstrates automatic tool security.\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   This might be expected if budget was exceeded!\n")
    
    finally:
        # Reset for next example
        PolicyEngine.reset()
        CostTracker.reset_session()


def example_llm_monitoring():
    """
    Example 2: LLM Cost Monitoring (The "Meter")
    
    Demonstrates how SentinelCrew automatically monitors LLM costs and enforces
    budget limits BEFORE expensive API calls are made.
    """
    print("\n" + "="*80)
    print("Example 2: LLM Cost Monitoring (The Meter)")
    print("="*80 + "\n")
    
    # Configure a tight budget to demonstrate blocking
    PolicyEngine.configure(
        run_budget=0.01,  # Very tight budget: $0.01
        strict_mode=True,
    )
    
    print("✓ Policy configured: $0.01 run budget (very tight!)\n")
    
    # Create an agent that will make LLM calls
    writer = Agent(
        role="Content Writer",
        goal="Write engaging content",
        backstory="Professional writer who creates high-quality content",
        verbose=True,
    )
    
    # Task that will trigger multiple LLM calls
    task = Task(
        description=(
            "Write a comprehensive 500-word article about artificial intelligence, "
            "including history, current applications, and future trends. "
            "Be thorough and detailed."
        ),
        agent=writer,
        expected_output="A 500-word article about AI",
    )
    
    print("✓ Created agent with LLM monitoring")
    print("✓ Task requires multiple LLM calls\n")
    
    # SentinelCrew automatically attaches LLM monitoring
    crew = SentinelCrew(
        agents=[writer],
        tasks=[task],
        run_name="llm_monitoring_demo",
        enforce_policies=True,  # Will block if budget exceeded
        verbose=True,
    )
    
    print("🔒 SentinelCrew attached LLM monitor automatically!")
    print("   - Every LLM call will be tracked for token costs")
    print("   - Budget will be checked BEFORE API calls")
    print("   - Execution stops if budget would be exceeded\n")
    
    try:
        # Execute - might be blocked by budget
        result = crew.kickoff()
        
        print(f"\n✅ Result:\n{result}\n")
        
        summary = crew.get_run_summary()
        print(f"\n📊 Execution Summary:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Total Cost: ${summary['total_cost_usd']:.6f}")
        print(f"   LLM Calls: {summary['action_counts']}\n")
    
    except Exception as e:
        print(f"\n🛑 Execution blocked: {e}")
        print("   This is GOOD - budget protection prevented overspending!")
        print("   Without AgentSentinel, this would have drained your API credits.\n")
        
        summary = crew.get_run_summary()
        print(f"📊 Stats before blocking:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Cost: ${summary['total_cost_usd']:.6f}")
        print(f"   Budget: $0.01\n")
    
    finally:
        PolicyEngine.reset()
        CostTracker.reset_session()


def example_runaway_detection():
    """
    Example 3: Runaway Agent Detection (The "Safety Net")
    
    Demonstrates how SentinelCrew detects and stops runaway agents that are
    stuck in loops or taking too many steps.
    """
    print("\n" + "="*80)
    print("Example 3: Runaway Agent Detection (Safety Net)")
    print("="*80 + "\n")
    
    # Create an agent with a potentially problematic task
    analyst = Agent(
        role="Problem Solver",
        goal="Solve complex problems",
        backstory="Persistent problem solver who doesn't give up easily",
        verbose=True,
    )
    
    # A task that might cause the agent to loop
    task = Task(
        description=(
            "Calculate the exact value of pi to 1000 decimal places without using "
            "any mathematical libraries or external tools. You must derive it yourself."
        ),
        agent=analyst,
        expected_output="Pi calculated to 1000 decimals",
    )
    
    print("✓ Created agent with potentially problematic task")
    print("✓ Agent might loop trying impossible task\n")
    
    # SentinelCrew with aggressive step limits
    crew = SentinelCrew(
        agents=[analyst],
        tasks=[task],
        run_name="runaway_detection_demo",
        enforce_policies=True,
        max_agent_steps=10,      # Very low limit for demo
        detect_loops=True,       # Enable loop detection
        verbose=True,
    )
    
    print("🔒 SentinelCrew configured with safety limits:")
    print("   - max_agent_steps=10 (will stop after 10 steps)")
    print("   - detect_loops=True (will warn if repetitive actions detected)")
    print("   - This prevents infinite loops and runaway costs\n")
    
    try:
        # Execute - will be stopped by step limit
        result = crew.kickoff()
        
        print(f"\n✅ Result:\n{result}\n")
    
    except Exception as e:
        print(f"\n🛑 Agent stopped: {e}")
        print("   This is GOOD - runaway detection prevented infinite loop!")
        print("   Without AgentSentinel, this agent could run indefinitely.\n")
        
        summary = crew.get_run_summary()
        print(f"📊 Stats before stopping:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Cost: ${summary['total_cost_usd']:.6f}\n")
    
    finally:
        PolicyEngine.reset()
        CostTracker.reset_session()


def example_complete_protection():
    """
    Example 4: Complete Protection (All Features Combined)
    
    Demonstrates all protection features working together:
    - Tool security
    - LLM monitoring
    - Runaway detection
    - Policy enforcement
    """
    print("\n" + "="*80)
    print("Example 4: Complete Protection (Visa-Like Integration)")
    print("="*80 + "\n")
    
    # Configure comprehensive policies
    PolicyEngine.configure(
        run_budget=0.50,  # $0.50 budget
        action_budgets={
            "web_search": 0.10,  # Max $0.10 on searches
        },
        denied_actions=[
            "file_delete",
            "database_drop",
            "send_email",  # Block potentially dangerous actions
        ],
        rate_limits={
            "web_search": {
                "max_count": 5,
                "window_seconds": 60,
            },
        },
        strict_mode=True,
    )
    
    print("✓ Comprehensive policies configured:")
    print("   - Run budget: $0.50")
    print("   - Search budget: $0.10")
    print("   - Blocked: file_delete, database_drop, send_email")
    print("   - Rate limit: 5 searches per minute\n")
    
    try:
        from crewai_tools import SerperDevTool
        
        # Create a team of agents with various tools
        researcher = Agent(
            role="Senior Researcher",
            goal="Conduct thorough research",
            backstory="Expert researcher with access to search tools",
            tools=[SerperDevTool()],
            verbose=True,
        )
        
        analyst = Agent(
            role="Data Analyst",
            goal="Analyze research findings",
            backstory="Analytical thinker who processes information",
            verbose=True,
        )
        
        writer = Agent(
            role="Content Writer",
            goal="Write compelling content",
            backstory="Professional writer who creates engaging content",
            verbose=True,
        )
        
        # Create tasks
        research_task = Task(
            description="Research the top 3 trends in AI safety for 2024",
            agent=researcher,
            expected_output="Summary of top 3 AI safety trends",
        )
        
        analysis_task = Task(
            description="Analyze the research and identify key implications",
            agent=analyst,
            expected_output="Analysis of implications",
        )
        
        writing_task = Task(
            description="Write a 200-word article about the findings",
            agent=writer,
            expected_output="A 200-word article",
        )
        
        print("✓ Created 3-agent crew with multiple tasks\n")
        
        # SentinelCrew with full protection
        crew = SentinelCrew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            run_name="complete_protection_demo",
            process=Process.sequential,
            enforce_policies=True,      # ✓ Policy enforcement
            max_agent_steps=30,         # ✓ Runaway protection
            detect_loops=True,          # ✓ Loop detection
            track_costs=True,           # ✓ Cost tracking
            verbose=True,
        )
        
        print("🔒 All protections active:")
        print("   ✓ Tools automatically secured")
        print("   ✓ LLM costs monitored and enforced")
        print("   ✓ Runaway agents will be stopped")
        print("   ✓ Policies checked before every action")
        print("   ✓ Rate limits enforced")
        print("   ✓ Dangerous actions blocked\n")
        
        print("Starting crew execution...\n")
        
        # Execute with full protection
        result = crew.kickoff()
        
        print(f"\n✅ Result:\n{result}\n")
        
        # Detailed summary
        summary = crew.get_run_summary()
        print(f"\n📊 Final Summary:")
        print(f"   ✓ Completed successfully!")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Total Cost: ${summary['total_cost_usd']:.6f}")
        print(f"   Budget: $0.50 (${0.50 - summary['total_cost_usd']:.6f} remaining)")
        print(f"   Actions: {summary['action_counts']}")
        print(f"\n   All 3 agents completed their tasks safely!")
        print(f"   No overspending, no runaway agents, no policy violations.\n")
    
    except ImportError:
        print("⚠️  crewai-tools not installed. Install with: pip install crewai-tools\n")
    
    except Exception as e:
        print(f"\n🛑 Protection triggered: {e}")
        print("   One of the safety mechanisms stopped execution.")
        print("   This prevented potential issues!\n")
        
        summary = crew.get_run_summary()
        print(f"📊 Stats at stop:")
        print(f"   Duration: {summary['duration_seconds']:.2f}s")
        print(f"   Cost: ${summary['total_cost_usd']:.6f}\n")
    
    finally:
        PolicyEngine.reset()
        CostTracker.reset_session()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("  CrewAI Active Control Integration Examples")
    print("  AgentSentinel: Visa-Like Security for AI Agents")
    print("="*80)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'\n")
        return
    
    print("\n✅ Prerequisites met. Starting examples...\n")
    
    try:
        # Run examples
        example_automatic_tool_security()
        
        input("\nPress Enter to continue to LLM Monitoring example...")
        example_llm_monitoring()
        
        input("\nPress Enter to continue to Runaway Detection example...")
        example_runaway_detection()
        
        input("\nPress Enter to continue to Complete Protection example...")
        example_complete_protection()
        
        print("\n" + "="*80)
        print("  All examples completed!")
        print("="*80)
        print("\n✅ Key Takeaways:")
        print("   1. Tools are automatically secured - no manual decoration needed")
        print("   2. LLM costs are tracked and budgets enforced in real-time")
        print("   3. Runaway agents are detected and stopped automatically")
        print("   4. All protection happens INSIDE the agent decision loop")
        print("\n   This is ACTIVE CONTROL - not just passive tracking!")
        print("   Your agents are now as secure as a Visa terminal.\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user.\n")
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
