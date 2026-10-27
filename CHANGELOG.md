# Changelog

All notable changes to AgentSentinel SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-10-27

No breaking changes for existing SDK or `agentsentinel-mcp` users.

### Added

- **`agentsentinel-runtime` approvals (t28b).** With `approvals.wait: true` in
  the runtime config, a `tools/call` that the platform answers with
  `require_approval` is held while the runtime polls the approval outbound with
  the same agent-bound key, then retried once if approved. Rejected, expired,
  cancelled and info-requested outcomes return a clear tool error; a wait that
  times out does not cancel the approval. Approver text is scrubbed of the key
  (and any 8+ character part of it) and of email addresses.
- **`agentsentinel-runtime doctor` proves the key is out of the model's reach
  (t28c).** New checks: key file outside the workspace (by path and by
  directory identity), key file and config file ownership and permissions,
  a bounded scan of the workspace for the key's bytes (files of any size,
  symlinks to outside files, an entry/byte/time budget), known harness config
  files, the environment, and upstream reachability (`--offline` skips it).
  doctor never passes what it cannot prove: it fails when the runtime runs as
  root or as the user that owns or can write the workspace, and when a scan
  limit is hit. `run_checks()` withholds any message holding part of the key.

### Security

- The runtime's config file must be owned by the current user, not writable by
  group or others, have no other hard links and sit outside the workspace; it
  is checked on the opened file and re-checked after reading.

### Changed

- Dependency floors raised for security advisories (PyJWT, starlette,
  python-multipart, langchain-core and others via the lockfile); `crewai`
  extra resolves to 1.15.x.

## [0.5.0] - 2026-10-27

No breaking changes: every existing call, including positional ones, keeps
working unchanged.

### Added

- **The platform URL defaults to the hosted AgentSentinel platform.**
  `platform_url` is now optional on every entry point that talks to the
  platform: `configure()`, `enable_remote_sync()`, `manual_flush()`,
  `SyncConfig`, `PolicyEngine.enable_remote_sync()` / `RemotePolicyConfig`,
  `KillSwitchClient.configure()` / `KillSwitchConfig`,
  `ApprovalClient.configure()` / `ApprovalConfig`, `RemoteCostBackend`,
  `MCPClient` / `mcp.set_default_client()` and
  `integrations.registry.auto_register_tools()`. A new shared resolver,
  `agent_sentinel._defaults.resolve_platform_url()`, applies one precedence
  everywhere: the explicit `platform_url` argument, then
  `AGENT_SENTINEL_PLATFORM_URL`, then `AGENT_SENTINEL_URL`, then the hosted
  platform. Whitespace and trailing
  slashes are stripped. Set either environment variable to point the SDK at a
  self-hosted or staging platform. The `agentsentinel-mcp` proxy shares the
  same default and keeps its existing `AGENT_SENTINEL_URL` handling and
  validation.
- **The API key falls back to `AGENT_SENTINEL_API_KEY`.** Where `api_token` /
  `api_key` followed `platform_url` positionally it now also has a default:
  when not passed it is read from `AGENT_SENTINEL_API_KEY`, and a
  `ConfigurationError` is raised if neither is set. `configure()` still takes
  `api_key` as its required first argument. `auto_register_tools()` still
  requires `agent_id` (raising `ConfigurationError` when it is missing).
- **`agentsentinel-runtime`** (`[mcp]` extra): a second console script that
  serves the `agentsentinel-mcp` stdio proxy from one validated
  `runtime.yaml` (`url`, `workspace`, `api_key_file`, `harness`,
  `approvals`), reading the key from a file outside the model's workspace.
  Outbound-only; subcommands `run`, `config` and `doctor`.

### Changed

- `AGENT_SENTINEL_COST_BACKEND=remote` no longer requires
  `AGENT_SENTINEL_PLATFORM_URL`; only `AGENT_SENTINEL_API_KEY` is required.

## [0.4.0] - 2026-09-10

One behaviour change, called out under Changed: the default enforcement mode
is now `fail_closed_writes`, not `fail_open`. An unconfigured or unreachable
SDK now blocks write-impact actions instead of silently permitting them.

### Added

- **Secrets are redacted from ledger inputs before write and upload.** New
  `agent_sentinel.redaction.redact_inputs(args, kwargs, *, max_bytes=8192)`
  replaces values whose key looks sensitive (`api_key`, `secret`, `token`,
  `password`, `passwd`, `authorization`, `credential`, `private_key`, ...) and
  string values matching common secret shapes (OpenAI/Stripe/AWS/GitHub/Slack
  keys, PEM private-key headers, JWTs) with `[redacted:<kind>]`. Oversized
  values are truncated instead (`...[truncated N bytes]`). `guarded_action`
  now applies this before every ledger write, so tool-call secrets never
  reach `.agent-sentinel/ledger.jsonl` or the platform upload. The ledger
  entry carries `inputs_redacted: true` when anything was replaced.
  `AGENT_SENTINEL_LEDGER_INPUTS=none` disables ledger input capture entirely.
- **Identity headers on every platform request, one auth scheme.** `configure()`
  now accepts `harness: str | None` and `principal: str | None` (also read
  from `AGENT_SENTINEL_HARNESS` / `AGENT_SENTINEL_PRINCIPAL`), stored in the
  `ExecutionContext` alongside `agent_id`/`run_id`. A single helper,
  `agent_sentinel.context.build_identity_headers()`, builds the headers for
  every platform call: `Authorization: Bearer <key>`, `User-Agent:
  agentsentinel-sdk/<version>`, and `X-Sentinel-Agent-Id` /
  `X-Sentinel-Run-Id` / `X-Sentinel-Harness` / `X-Sentinel-Principal`
  (omitted when unset). This lets the platform bind identity to the API key
  instead of trusting request-body strings. Request bodies still send
  `agent_id`/`run_id` for backward compatibility.
- **`fail_closed` now covers every fail-open path, not just "no policy loaded".**
  Under `fail_closed`: a remote policy sync that has never succeeded and has
  no local cache blocks with `PolicyNotConfiguredError` (unchanged from
  0.3.0, but now reliably reached — see Changed); a kill-switch check that
  cannot complete (platform unreachable, non-200 response, or httpx not
  installed) raises the new `KillSwitchUnavailableError` (an `AgentKilledError`
  subclass, reason code `KILL_SWITCH_UNAVAILABLE`) instead of assuming
  not-killed; and an unexpected exception raised from inside policy
  evaluation itself is wrapped in `PolicyViolationError` (reason code
  `EVALUATION_ERROR`) instead of propagating raw. Under `fail_open`, all
  three keep their existing behaviour exactly.
- **`fail_closed` on the public entry points.** `configure(..., fail_closed:
  bool | None = None)`, `PolicyEngine.configure(..., fail_closed=None)` and
  `guarded_action(..., fail_closed=None)` accept an explicit override.
  Precedence is explicit argument > `AGENT_SENTINEL_ENFORCEMENT` > default
  (`fail_closed_writes` as of this release — see below). `fail_closed=True`/
  `False` still resolve to a plain `"fail_closed"`/`"fail_open"`, not the new
  impact-aware mode. `PolicyEngine.set_enforcement_mode(mode, pin=True)` pins
  a mode so it survives `PolicyEngine.reset()`; pass `reset(clear_pin=True)`
  to also clear the pin (test isolation does this so a pinned mode cannot
  leak between tests).
- **`guarded_action(..., impact: Literal["read", "low", "medium", "high"] =
  "medium")`.** Declares how consequential an action is. Sent to the
  platform in the ledger/ingest entry (`entry["impact"]`) and consulted
  locally by the new default enforcement mode below. Read-only tools should
  declare `impact="read"` or `"low"` so they keep working without a policy
  loaded.
- **New default enforcement mode: `fail_closed_writes`.** See Changed for
  the full behaviour change. New `NoPolicyForWriteError` (a
  `PolicyViolationError` subclass, reason code `NO_POLICY_FOR_WRITE`),
  raised for `impact="medium"`/`"high"` actions blocked under this mode.
- New exports: `KillSwitchUnavailableError`, `NoPolicyForWriteError`.

### Changed

- **Breaking: the default enforcement mode is now `fail_closed_writes`, not
  `fail_open`.** Previously an unconfigured or unreachable `PolicyEngine`
  permitted every action. Now, whenever the policy layer cannot vouch for an
  action — unconfigured, a remote sync that has never produced a usable
  local cache, or policy evaluation itself raising an unexpected error —
  actions declaring `impact="medium"` or `"high"` (`guarded_action`'s own
  default) are blocked with `NoPolicyForWriteError`
  (`reason_code="NO_POLICY_FOR_WRITE"`), while `impact="read"`/`"low"`
  actions proceed with a one-time warning, same as before. **Escape hatch:**
  set `AGENT_SENTINEL_ENFORCEMENT=fail_open` (or
  `PolicyEngine.set_enforcement_mode("fail_open")`) to restore the exact
  pre-0.4 behaviour for one more minor version. `fail_closed` remains
  available to block every action regardless of declared impact.
  `PolicyEngine.reset()` keeps its own, independent contract of reverting an
  unpinned mode to `fail_open` (a test/ops-utility detail; a real process
  that never calls `reset()` sees `fail_closed_writes` from the start).
- **`sync.py`, `policy.py`, `kill_switch.py`, `mcp.py`, and `approval.py` now
  all authenticate with `Authorization: Bearer <key>`.** The `ApiKey <key>`
  scheme has been removed from these modules; the platform already accepted
  both schemes, so this is backward compatible.
- **`PolicyEngine.configure(fail_closed=True)` alone no longer installs an
  empty, permissive policy.** Previously, calling `configure()` at all —
  even with only `fail_closed` set and no budgets/deny list — created a
  `PolicyConfig()` and marked the engine "configured", so `is_configured()`
  returned `True` and every action was silently permitted (nothing in the
  empty config denies or budgets anything). `configure()` now only installs
  a `PolicyConfig` when at least one policy argument (budgets, deny/allow
  lists, rate limits, evidence rules, etc.) is supplied; `fail_closed` is
  applied via `set_enforcement_mode(pin=True)` regardless. The same guard
  applies to the top-level `configure()` in `_configure.py`.
- **A remote sync that succeeds with an empty policy list still leaves the
  engine unconfigured**, so `fail_closed`/`fail_closed_writes` continue to
  block actions even after a "successful" sync with nothing to enforce.
  This is a known gap, not this release's fix — a future release may
  distinguish empty-but-synced from never-synced.

## [0.3.0] - 2026-07-26

Hardening pass so the SDK can back a deployed control, not only a
single-process library. Rebased onto 0.2.0; the guardrails, group
attribution and idempotency work from that release is preserved and the
guardrail reason codes are wired into the new intervention dispatch.

One behaviour change, called out under Changed: budgets now actually bind.

### Added

- **Pluggable cost backends** (`agent_sentinel.cost_backend`). `CostTracker`
  now delegates storage to a `CostBackend`. `InMemoryCostBackend` is the
  default and behaves exactly as before. `FileCostBackend` shares counters
  across processes on one host using a lock file, with no new dependencies.
  Configure in code with `CostTracker.set_backend(...)`, or via
  `AGENT_SENTINEL_COST_BACKEND=file` and `AGENT_SENTINEL_COST_PATH=...`.
- **`PolicyEngine.check_and_reserve()`** — checks an action and reserves its
  cost against every budget inside a single lock hold. Use this instead of
  `check_action()` + `add_cost()` whenever the result gates execution: the
  two-call form is check-then-act, and concurrent workers can all pass the
  check before any records. Measured with 6 workers against a $1.00 cap: the
  two-step path spends $3.00, `check_and_reserve` spends $1.00.
- **`CostBackend.try_reserve()`** — the atomic compare-and-increment primitive
  behind the above. A custom backend without it falls back to the non-atomic
  path and logs at debug.
- **Enforcement modes.** `PolicyEngine.set_enforcement_mode("fail_closed")` or
  `AGENT_SENTINEL_ENFORCEMENT=fail_closed` makes an unconfigured engine block
  instead of permit. **The default remains `fail_open`**, unchanged, so no
  existing install starts blocking. New `PolicyNotConfiguredError` (a subclass
  of `PolicyViolationError`).
- **Remediation on every block type.** Deny lists, allowlist misses, rate
  limits and budget blocks now carry a `RemediationPayload` telling the agent
  what to do next, not just that it failed. Allowlist misses name the
  permitted actions; rate limits carry `retry_after_seconds`. Previously only
  evidence violations did this.
- New exports: `PolicyNotConfiguredError`, `CostBackend`, `CostSnapshot`,
  `InMemoryCostBackend`, `FileCostBackend`.

### Changed

- **Budgets now actually bind.** `@guarded_action` reserves cost atomically via
  `PolicyEngine.check_and_reserve()` instead of checking and then recording in
  a `finally`. Measured with 6 threads against a $1.00 run budget: the old path
  spent $3.00 with zero blocks, the new one holds at $1.00. **If your budgets
  were set low and your agents were silently overshooting, they will now raise
  `BudgetExceededError`.** Cost is still recorded when an action raises, as
  before. Do not call `check_and_reserve()` yourself on a decorated action; the
  decorator already did, and a second call double-counts.
- A corrupt or unreadable cost state file is now quarantined (renamed, logged
  at ERROR) rather than silently serving zeroed counters and persisting those
  zeros — which reset every budget permanently.
- Non-finite and negative spends are rejected at the counter boundary. A single
  NaN made every `>` comparison False and disabled every budget for good.
- Lock-wait timeouts raise `agent_sentinel.TimeoutError` (an
  `AgentSentinelError`) rather than the builtin, so they degrade to a blocked
  result instead of escaping every SDK handler.
- `AGENT_SENTINEL_ENFORCEMENT` typos raise on every call, not just the first.
  The flag marking the env as checked was set before the raise, so a typo threw
  once and then ran `fail_open` for the rest of the process.
- Compliance metadata is cleared on any policy-gate failure, not only the three
  anticipated policy exceptions, so an unexpected error cannot leak one action's
  approval metadata into the next action's ledger row.
- Argument-constraint violations report the canonical
  `ARGUMENT_CONSTRAINT_VIOLATED`; the decorator-level check previously emitted
  `...VIOLATION`, so the code differed by where the constraint was declared.
- The LangChain integration forwards `remediation_payload` on policy and budget
  interventions, matching `guard.py`.
- `ToolResult.remediation` now merges both shapes it previously had: the error
  fields (`message`, `error_code`, `details`, `recoverable`) and the guidance
  fields (`reason_code`, `retry_guidance`, `safe_alternatives`) appear side by
  side, plus a nested `remediation` key. Callers of either prior shape keep
  working.
- Rate-limit violations now set `recoverable=True` (they clear by waiting).
  Deny-list and budget violations remain `recoverable=False`.
- Intervention type is derived from `reason_code` rather than substring-matching
  the error message, so rewording a message no longer reclassifies a block.
- Policy and budget interventions now forward `remediation_payload` to the
  platform. Previously only evidence interventions did.
- `AGENT_SENTINEL_ENFORCEMENT` with an unrecognised value now raises
  `ValueError` instead of logging and falling back to `fail_open`. An operator
  who typed it was turning enforcement on; landing permissive is the wrong
  direction to fail.

### Fixed

- `FileCostBackend` no longer crashes on a state file containing valid JSON of
  the wrong shape (`[1,2,3]`, `null`); it degrades to zero.
- `FileCostBackend` honours its documented lock `timeout` instead of blocking
  forever behind a wedged holder.
- `PolicyEngine.reset()` clears enforcement state, so opting into
  `fail_closed` no longer leaks into subsequent code in the same process.
- `BudgetExceededError` tolerates a non-numeric limit instead of raising
  `TypeError` from inside the exception constructor.
- `RemediationPayload.to_dict()` omits `retry_after_seconds` when unset,
  keeping evidence payloads byte-identical to prior releases.

## [0.2.0] - 2026-04-16

### Added

#### Built-in Runtime Guardrails

- **PII Detection** (`PIIGuard`, `PIIRule`, `detect_pii`): stdlib-only regex detector for email, US SSN, Luhn-validated credit card, US phone, AWS access key, private-key blocks, and generic API-key-like tokens. Recursively walks action kwargs. Wired into `PolicyEngine.check_action` via `pii_rules` / `pii_default_enabled`.
- **Content Moderation** (`ModerationGuard`, `ModerationRule`, `KeywordModerator`): pluggable `Moderator` protocol with an offline keyword-based default. Strict / balanced / permissive strictness. `moderation_rules` / `moderation_default_enabled`.
- **Loop Protection** (`LoopGuard`, `LoopRule`): sliding-window detector on `(action_name, arg_hash)` with configurable threshold, window, and `arg_exclude` for nonces. `loop_rules` / `loop_default_enabled`.
- **Idempotency** (`IdempotencyCache`, `IdempotencyHit`): in-process keyed cache with TTL and `run_id` scoping. `@guarded_action(idempotency_key=..., idempotency_ttl_seconds=...)` wired in both async and sync execution paths; cache hit short-circuits before policy/approval checks.

#### Self-Repair Feedback Enrichment

- Every guardrail block now carries `retry_guidance`, `safe_alternatives`, `recoverable`, and `prior_attempts` (loop) in `PolicyViolationError.details` — enabling LLM agents to self-correct without operator intervention.

#### New Intervention Types

- `InterventionType.PII_BLOCKED`, `CONTENT_BLOCKED`, `LOOP_DETECTED`, `IDEMPOTENT_REPLAY` — blocks are categorized in the ledger by their guardrail type, not just `HARD_BLOCK`.

#### PolicyConfig Extensions

- `pii_rules`, `moderation_rules`, `loop_rules`: per-action rule maps.
- `pii_default_enabled`, `moderation_default_enabled`, `loop_default_enabled`: global toggle switches.

#### Examples

- `examples/langgraph_sentinel.py`: runnable 3-node LangGraph workflow demonstrating evidence-graph enforcement, groundedness checks, and structured remediation payloads flowing back as tool messages.

### Changed

- `PolicyEngine.check_action` now runs PII → moderation → loop checks between argument-constraint and groundedness steps (backward-compatible; new checks only fire when rules are configured).
- `_record_policy_intervention` in `guard.py` routes blocks by `reason_code` so the ledger categorizes PII/moderation/loop interventions correctly.

### Fixed

- `docs/README.md`: "zero-dependency" claim corrected to "minimal-dependency (Pydantic + PyYAML)".

---

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
