"""
``agentsentinel-runtime``: the AgentSentinel stdio proxy packaged for an agent VM.

One YAML config file names the platform URL, the model's workspace and a key
file. The runtime serves ``agent_sentinel.mcp_proxy`` over stdio with it and
is outbound-only: it never binds or listens on a socket. The key is read from
the key file only and never printed.

Needs the optional extra: ``pip install "agentsentinel-sdk[mcp]"`` (Python 3.10+).
This package imports nothing on purpose, so ``import agent_sentinel`` never
needs ``mcp``; use ``agent_sentinel.runtime.cli`` / ``.config`` / ``.doctor``.
"""
