# Contributing to Agent Sentinel SDK

Thank you for your interest in contributing to Agent Sentinel SDK! We welcome contributions from the community.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/agent-sentinel-sdk.git
   cd agent-sentinel-sdk
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=agent_sentinel --cov-report=html

# Run specific test file
uv run pytest tests/test_guard_coverage.py -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking with mypy
uv run mypy agent_sentinel/
```

### Making Changes

1. **Write tests first** - We practice test-driven development
2. **Keep changes focused** - One feature/fix per pull request
3. **Update documentation** - Update README.md and docstrings as needed
4. **Add changelog entry** - Update CHANGELOG.md with your changes

## Pull Request Process

1. **Ensure tests pass** - All existing and new tests must pass
2. **Update documentation** - Include relevant documentation updates
3. **Follow code style** - Run ruff format and ruff check
4. **Write clear commit messages** - Use descriptive commit messages
5. **Submit PR** - Push to your fork and create a pull request

### Commit Message Format

```
type: brief description

Longer description if needed

- Bullet points for details
- Reference issues: Fixes #123
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Maximum line length: 88 characters (ruff default)
- Use descriptive variable names
- Add docstrings to all public functions

## Testing Guidelines

- Write unit tests for new features
- Aim for high test coverage (>80%)
- Test both sync and async code paths
- Test error conditions and edge cases
- Use descriptive test names

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Email: hello@agentsentinel.dev

## Code of Conduct

Please be respectful and constructive in all interactions. We are building a welcoming community.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
