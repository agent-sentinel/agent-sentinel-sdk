# Changelog

All notable changes to AgentSentinel SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-15

### Initial Release

**AgentSentinel is the operational circuit breaker for autonomous agents.**

This SDK provides runtime authority and control over autonomous AI agents, enabling developers and organizations to actively manage the financial, logical, and compliance risks of agents in production.

> **Core Value:** AgentSentinel exists to decide whether an autonomous system is allowed to act — before it does.

#### The Three Pillars

**1. Active Safety 🛡️**
Runtime enforcement that overrides agent intent. Every check happens BEFORE execution.

- `@guarded_action` decorator for action instrumentation
- Budget enforcement (session, run, and action-level hard caps)
- Action bans (deny list prevents execution of forbidden actions)
- Action allowlists (restrict agents to approved actions only)
- Rate limiting (time-windowed execution limits)
- Fail-open design (agents never fail-open without explicit authorization)
- High-precision time tracking (nanosecond resolution)

**2. Governance 📋**
Complete authority over what happened, when, and why.

- Immutable execution ledger (`.agent-sentinel/ledger.jsonl`)
- Cost tracking with per-action, per-run, per-session attribution
- Human-in-the-loop approval workflow
- Decision rationale logging
- Data lineage tracking
- Replay mode for deterministic testing
- Input divergence detection
- EU AI Act Article 14 compliance metadata

**3. Operations ⚙️**
The bridge between autonomous agents and human authority.

- Policy engine with YAML configuration (`callguard.yaml`)
- Remote policy synchronization from platform
- Background sync with batch uploads and retry logic
- Intervention tracking (records when Sentinel blocks actions)
- Approval client for managing approval requests
- Interruptible execution with resumable authority

#### Integrations

**LLM Providers**
- OpenAI with automatic cost tracking
- Anthropic Claude
- xAI/Grok
- Google Gemini
- Token-based cost calculation with pricing database

**AI Frameworks**
- LangChain callback handler for chain/agent tracing
- CrewAI integration for crew and task monitoring
- MCP (Model Context Protocol) client support

**Error Handling & Resilience**
- Structured exception hierarchy
- `with_retry` decorator with exponential backoff
- `CircuitBreaker` for fault tolerance
- Comprehensive error types: `BudgetExceededError`, `PolicyViolationError`, `ReplayDivergenceError`, `NetworkError`, `SyncError`, `TimeoutError`, `ConfigurationError`

#### Installation

```bash
# Basic installation
pip install agentsentinel-sdk

# With remote sync
pip install agentsentinel-sdk[remote]

# With LLM integrations
pip install agentsentinel-sdk[llm]

# With framework integrations
pip install agentsentinel-sdk[integrations]

# With everything
pip install agentsentinel-sdk[all]
```

#### Technical Highlights

- **Python 3.9+** support
- **Type-safe** with full type hints and `py.typed` marker
- **Minimal dependencies** (pydantic + pyyaml for core)
- **Local-first** architecture (works completely offline)
- **Async/sync** support throughout
- **Production-ready** with comprehensive test suite (4,737 lines)
- **MIT License**

#### Strategic Positioning

AgentSentinel is not:
- A logging tool (that's a camera)
- An analytics dashboard (that's a report)
- An observability platform (that's hindsight)

AgentSentinel is:
- The brakes on the car
- The circuit breaker in the system
- The authorization layer between intent and action

> "Commoditize observability. Monetize control."

[0.1.0]: https://github.com/agent-sentinel/agent-sentinel-sdk/releases/tag/v0.1.0
