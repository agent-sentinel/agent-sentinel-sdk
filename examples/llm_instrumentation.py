"""
Example: LLM Instrumentation with AgentSentinel

This example demonstrates automatic cost tracking for OpenAI, Anthropic,
Grok, and Gemini LLM providers using transparent instrumentation.

Requirements:
    pip install agent-sentinel[remote]
    pip install openai  # For OpenAI/Grok
    pip install anthropic  # For Anthropic
    pip install google-generativeai  # For Gemini
"""
import os
from agent_sentinel.integrations.llm import (
    instrument_openai,
    instrument_anthropic,
    instrument_grok,
    instrument_gemini,
    get_token_costs,
)
from agent_sentinel.cost import CostTracker


def example_openai():
    """Example 1: OpenAI with automatic cost tracking."""
    print("\n=== Example 1: OpenAI Instrumentation ===\n")
    
    # Instrument OpenAI (patches the client)
    instrument_openai()
    
    import openai
    
    # Use OpenAI normally - costs tracked automatically!
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Making OpenAI API calls with automatic cost tracking...\n")
    
    # Test different models
    models = [
        ("gpt-3.5-turbo", "What is Python?"),
        ("gpt-4o-mini", "Explain quantum computing in one sentence."),
        ("gpt-4o", "What are the benefits of type hints?"),
    ]
    
    for model, prompt in models:
        print(f"Calling {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
        )
        
        print(f"  Response: {response.choices[0].message.content[:80]}...")
        print(f"  Tokens: {response.usage.total_tokens}")
        print()
    
    # Get cost summary
    costs = get_token_costs()
    print(f"\nCost Summary:")
    print(f"  Total: ${costs['total_usd']:.6f}")
    print(f"  By Provider: {costs['by_provider']}")
    print(f"  By Model:")
    for model, cost in costs['by_model'].items():
        print(f"    - {model}: ${cost:.6f}")


def example_anthropic():
    """Example 2: Anthropic with automatic cost tracking."""
    print("\n=== Example 2: Anthropic Instrumentation ===\n")
    
    # Reset for clean example
    CostTracker.reset_run()
    
    # Instrument Anthropic
    instrument_anthropic()
    
    import anthropic
    
    # Use Anthropic normally
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print("Making Anthropic API calls with automatic cost tracking...\n")
    
    # Test different Claude models
    models = [
        ("claude-3-haiku-20240307", "What is Rust?"),
        ("claude-3-5-sonnet-20241022", "Explain machine learning briefly."),
    ]
    
    for model, prompt in models:
        print(f"Calling {model}...")
        message = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print(f"  Response: {message.content[0].text[:80]}...")
        print(f"  Tokens: {message.usage.input_tokens + message.usage.output_tokens}")
        print()
    
    # Get cost summary
    costs = get_token_costs()
    print(f"\nCost Summary:")
    print(f"  Total: ${costs['total_usd']:.6f}")
    print(f"  By Model:")
    for model, cost in costs['by_model'].items():
        print(f"    - {model}: ${cost:.6f}")


def example_grok():
    """Example 3: Grok/xAI with automatic cost tracking."""
    print("\n=== Example 3: Grok Instrumentation ===\n")
    
    # Reset for clean example
    CostTracker.reset_run()
    
    # Instrument Grok (uses OpenAI-compatible API)
    instrument_grok()
    
    import openai
    
    # Configure for Grok/xAI
    client = openai.OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    
    print("Making Grok API call with automatic cost tracking...\n")
    
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "user", "content": "What makes Grok unique?"}
        ],
        max_tokens=100,
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    
    # Get cost summary
    costs = get_token_costs()
    print(f"\nCost Summary:")
    print(f"  Total: ${costs['total_usd']:.6f}")
    print(f"  By Provider: {costs['by_provider']}")


def example_gemini():
    """Example 4: Google Gemini with automatic cost tracking."""
    print("\n=== Example 4: Gemini Instrumentation ===\n")
    
    # Reset for clean example
    CostTracker.reset_run()
    
    # Instrument Gemini
    instrument_gemini()
    
    import google.generativeai as genai
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    print("Making Gemini API calls with automatic cost tracking...\n")
    
    # Test different Gemini models
    models = [
        ("gemini-1.5-flash", "What is JavaScript?"),
        ("gemini-1.5-pro", "Explain neural networks briefly."),
    ]
    
    for model_name, prompt in models:
        print(f"Calling {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        print(f"  Response: {response.text[:80]}...")
        if hasattr(response, 'usage_metadata'):
            print(f"  Tokens: {response.usage_metadata.total_token_count}")
        print()
    
    # Get cost summary
    costs = get_token_costs()
    print(f"\nCost Summary:")
    print(f"  Total: ${costs['total_usd']:.6f}")
    print(f"  By Provider: {costs['by_provider']}")
    print(f"  By Model:")
    for model, cost in costs['by_model'].items():
        print(f"    - {model}: ${cost:.6f}")


def example_multi_provider():
    """Example 5: Multiple providers in one application."""
    print("\n=== Example 5: Multi-Provider Cost Tracking ===\n")
    
    # Reset for clean example
    CostTracker.reset_run()
    
    # Instrument all providers
    instrument_openai()
    # instrument_anthropic()  # Uncomment if you have API key
    # instrument_gemini()  # Uncomment if you have API key
    
    print("Using multiple LLM providers with unified cost tracking...\n")
    
    # OpenAI call
    import openai
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("1. OpenAI GPT-4o-mini:")
    response1 = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is AI?"}],
        max_tokens=50,
    )
    print(f"   Response: {response1.choices[0].message.content[:60]}...")
    
    # Anthropic call (if available)
    # import anthropic
    # anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # 
    # print("\n2. Claude 3 Haiku:")
    # message = anthropic_client.messages.create(
    #     model="claude-3-haiku-20240307",
    #     max_tokens=50,
    #     messages=[{"role": "user", "content": "What is ML?"}]
    # )
    # print(f"   Response: {message.content[0].text[:60]}...")
    
    # Get unified cost summary
    costs = get_token_costs()
    print(f"\n\nUnified Cost Summary Across All Providers:")
    print(f"  Total Cost: ${costs['total_usd']:.6f}")
    print(f"\n  By Provider:")
    for provider, cost in costs['by_provider'].items():
        print(f"    - {provider}: ${cost:.6f}")
    print(f"\n  By Model:")
    for model, cost in costs['by_model'].items():
        print(f"    - {model}: ${cost:.6f}")
    print(f"\n  Total API Calls: {sum(costs['action_counts'].values())}")


def example_with_budgets():
    """Example 6: Cost tracking with budget enforcement."""
    print("\n=== Example 6: Budget Enforcement ===\n")
    
    from agent_sentinel import PolicyEngine, PolicyConfig, guarded_action
    
    # Reset for clean example
    CostTracker.reset_run()
    
    # Configure budget policy
    policy = PolicyConfig(
        budget_limits={
            "action": 0.01,  # Max $0.01 per action
            "run": 0.05,     # Max $0.05 per run
        }
    )
    
    PolicyEngine.set_policy(policy)
    
    # Instrument OpenAI
    instrument_openai()
    
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Wrap API calls with budget enforcement
    @guarded_action(name="gpt_call", cost_usd=0.0)  # Cost tracked automatically
    def call_gpt(prompt: str, model: str = "gpt-4o-mini"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return response.choices[0].message.content
    
    print("Making LLM calls with budget enforcement...\n")
    
    try:
        # First call - should succeed
        result1 = call_gpt("What is Python?")
        print(f"Call 1 succeeded: {result1[:60]}...")
        
        # Second call - should succeed
        result2 = call_gpt("What is JavaScript?")
        print(f"Call 2 succeeded: {result2[:60]}...")
        
        # More calls might exceed budget
        result3 = call_gpt("What is Rust?")
        print(f"Call 3 succeeded: {result3[:60]}...")
        
    except Exception as e:
        print(f"\nBudget exceeded! {e}")
    
    # Show final costs
    costs = get_token_costs()
    print(f"\n\nFinal Costs:")
    print(f"  Total: ${costs['total_usd']:.6f}")
    print(f"  Budget Limit: $0.05")


def example_cost_comparison():
    """Example 7: Compare costs across providers for the same task."""
    print("\n=== Example 7: Provider Cost Comparison ===\n")
    
    prompt = "Explain machine learning in 2 sentences."
    
    # Test configurations: (provider, model, description)
    configs = [
        ("openai", "gpt-3.5-turbo", "OpenAI GPT-3.5 Turbo"),
        ("openai", "gpt-4o-mini", "OpenAI GPT-4o Mini"),
        ("openai", "gpt-4o", "OpenAI GPT-4o"),
    ]
    
    # Instrument OpenAI
    instrument_openai()
    
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    results = []
    
    for provider, model, description in configs:
        CostTracker.reset_run()
        
        print(f"\nTesting {description}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        
        costs = get_token_costs()
        
        results.append({
            "description": description,
            "model": model,
            "cost": costs['total_usd'],
            "tokens": response.usage.total_tokens,
            "response": response.choices[0].message.content[:80],
        })
    
    # Display comparison
    print("\n\n" + "="*80)
    print("COST COMPARISON")
    print("="*80)
    
    for r in sorted(results, key=lambda x: x['cost']):
        print(f"\n{r['description']}:")
        print(f"  Model: {r['model']}")
        print(f"  Cost: ${r['cost']:.6f}")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Cost per 1k tokens: ${(r['cost'] / r['tokens'] * 1000):.6f}")
        print(f"  Response: {r['response']}...")


if __name__ == "__main__":
    print("AgentSentinel LLM Instrumentation Examples")
    print("=" * 80)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set (optional)")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nWarning: GOOGLE_API_KEY not set (optional)")
    
    if not os.getenv("XAI_API_KEY"):
        print("\nWarning: XAI_API_KEY not set (optional)")
    
    # Run examples
    try:
        if os.getenv("OPENAI_API_KEY"):
            example_openai()
            # example_multi_provider()  # Uncomment to run
            # example_with_budgets()  # Uncomment to run
            # example_cost_comparison()  # Uncomment to run
        
        # if os.getenv("ANTHROPIC_API_KEY"):
        #     example_anthropic()  # Uncomment to run
        
        # if os.getenv("XAI_API_KEY"):
        #     example_grok()  # Uncomment to run
        
        # if os.getenv("GOOGLE_API_KEY"):
        #     example_gemini()  # Uncomment to run
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


