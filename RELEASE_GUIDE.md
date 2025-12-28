# AgentSentinel SDK v0.1.0 - Release Guide

Complete step-by-step guide for releasing AgentSentinel SDK to PyPI and GitHub.

---

## Quick Links

- **TestPyPI**: https://test.pypi.org/
- **Production PyPI**: https://pypi.org/
- **GitHub Releases**: https://github.com/agent-sentinel/agent-sentinel-sdk/releases

---

## Prerequisites

- [x] Package built successfully (agentsentinel_sdk-0.1.0.tar.gz and .whl)
- [x] Twine check passed
- [ ] PyPI account created
- [ ] PyPI API token generated

---

## Part 1: Test on TestPyPI (RECOMMENDED)

### 1.1 Create TestPyPI Account
- Go to https://test.pypi.org/account/register/
- Verify your email

### 1.2 Create API Token
- Go to https://test.pypi.org/manage/account/token/
- Create token with scope: "Entire account"
- Copy and save the token securely

### 1.3 Upload to TestPyPI

```bash
.venv/bin/twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: `<your-testpypi-token>`

### 1.4 Test Installation

```bash
# Create test environment
python3 -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    agentsentinel-sdk

# Verify it works
python3 -c "from agent_sentinel import guarded_action; print('✅ Success!')"

# Clean up
deactivate
rm -rf test-env
```

---

## Part 2: Publish to Production PyPI

### 2.1 Create PyPI Account
- Go to https://pypi.org/account/register/
- Verify your email

### 2.2 Create API Token
- Go to https://pypi.org/manage/account/token/
- Create token with scope: "Entire account"
- Copy and save the token securely

### 2.3 Upload to Production PyPI

```bash
.venv/bin/twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: `<your-pypi-token>`

### 2.4 Verify on PyPI
- Go to https://pypi.org/project/agentsentinel-sdk/
- Verify version 0.1.0 appears
- Check description and metadata

### 2.5 Test Installation

```bash
pip install agentsentinel-sdk
python3 -c "from agent_sentinel import guarded_action; print('✅ Success!')"
```

---

## Part 3: Commit and Push to GitHub

### 3.1 Review Changes

```bash
git status
git status --ignored  # Verify .gitignore works
```

### 3.2 Stage Changes

```bash
git add .
```

### 3.3 Review Staged Changes

```bash
git diff --cached --stat
git diff --cached  # Full diff (optional)
```

### 3.4 Commit

```bash
git commit -m "feat: release v0.1.0 - initial public release

AgentSentinel SDK - The operational circuit breaker for autonomous agents

Package Rebranding:
- Rename package to agentsentinel-sdk (PyPI)
- Update product positioning to 'runtime authority and control'
- Align with strategic manifesto: Active Safety, Governance, Operations

Core Changes:
- Complete feature set with all runtime authority capabilities
- Action instrumentation with @guarded_action decorator
- Budget enforcement (session, run, action-level hard caps)
- Policy engine with YAML configuration and remote sync
- Human-in-the-loop approval workflows
- Replay mode for deterministic testing
- EU AI Act Article 14 compliance metadata
- LLM integrations (OpenAI, Anthropic, Grok, Gemini)
- Framework support (LangChain, CrewAI, MCP)

Documentation:
- Add CHANGELOG.md with Three Pillars framework
- Update README.md with circuit breaker positioning
- Add CONTRIBUTING.md with development guidelines
- Production-ready build (Twine check PASSED)

Strategic Positioning:
- 'The operational circuit breaker for autonomous agents'
- Positioned as authorization layer, not observability tool

License: MIT | Python: >=3.9
This is the first public release of AgentSentinel SDK."
```

### 3.5 Push to GitHub

```bash
git push origin main
```

### 3.6 Create and Push Tag

```bash
git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release

AgentSentinel SDK - The operational circuit breaker for autonomous agents

Highlights:
- Active Safety: Runtime enforcement with budget caps, action bans
- Governance: Immutable ledger, replay mode, compliance metadata
- Operations: Human-in-the-loop approvals, policy engine
- Integrations: OpenAI, Anthropic, Grok, Gemini, LangChain, CrewAI
- Production-ready with 4,737 lines of tests

See CHANGELOG.md for full details."

git push origin v0.1.0
```

---

## Part 4: Create GitHub Release (Optional)

### 4.1 Navigate to Releases
- Go to https://github.com/agent-sentinel/agent-sentinel-sdk/releases/new

### 4.2 Configure Release
- **Tag**: v0.1.0
- **Title**: v0.1.0 - Initial Public Release
- **Description**: See full description in detailed guide above

### 4.3 Attach Files
Upload from `dist/`:
- agentsentinel_sdk-0.1.0-py3-none-any.whl
- agentsentinel_sdk-0.1.0.tar.gz

### 4.4 Publish
- Check "Set as the latest release"
- Click "Publish release"

---

## Post-Release Checklist

- [ ] Verify package on PyPI: https://pypi.org/project/agentsentinel-sdk/
- [ ] Test installation: `pip install agentsentinel-sdk`
- [ ] Verify GitHub commit appears
- [ ] Verify tag v0.1.0 shows up
- [ ] Check GitHub Release page
- [ ] Update agentsentinel.dev website
- [ ] Announce on social media
- [ ] Send email announcement

---

## Package Information

- **PyPI Name**: agentsentinel-sdk
- **GitHub**: agent-sentinel/agent-sentinel-sdk
- **Version**: 0.1.0
- **Description**: Runtime authority and control for autonomous AI agents
- **Tagline**: The operational circuit breaker for autonomous agents
- **License**: MIT
- **Python**: >=3.9

---

## Support

- **Issues**: https://github.com/agent-sentinel/agent-sentinel-sdk/issues
- **Email**: hello@agentsentinel.dev
- **Website**: https://agentsentinel.dev
