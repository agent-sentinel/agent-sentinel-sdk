"""
Tests for ``agentsentinel-runtime doctor``'s secret-isolation checks
(agent_sentinel.runtime.doctor), and for the fallback in ``config.py`` used
where ``/proc/self/fd`` does not exist (macOS).

Every check gets a passing and a failing case on a ``tmp_path`` workspace.
Every doctor run goes through ``_doctor``, which asserts that no output line
holds the sentinel key or any 8 characters of it in a row, and that every
line keeps the one-line ``PASS|FAIL|SKIP <check-id> <message>`` contract.
``upstream-reachable`` drives the real ``mcp_proxy.serve`` with only the
HTTP transport replaced by an ``httpx.MockTransport``. No network is used.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re

import pytest
import yaml

if importlib.util.find_spec("mcp") is not None:
    import anyio
    import httpx

    from agent_sentinel.runtime import cli, doctor
    from agent_sentinel.runtime import config as runtime_config
    from agent_sentinel.runtime.config import ConfigError


@pytest.fixture
def _require_mcp():
    pytest.importorskip("mcp")


pytestmark = pytest.mark.usefixtures("_require_mcp")

# No 8-character run of this appears in any check id, fixed message or path.
SENTINEL = "as_Q7vK2mZp9XwL4tRj8NcY3bHs6DfG1"
REAL_UID = os.getuid()

IDS = [
    "config-file",
    "config-paths",
    "config-file-location",
    "key-file-location",
    "key-file-permissions",
    "key-file-content",
    "config-url",
    "config-harness",
    "config-approvals",
    "config-no-key",
    "key-file-outside-workspace",
    "runtime-user-separate",
    "key-not-in-workspace",
    "key-not-in-harness-config",
    "key-not-in-environment",
    "upstream-reachable",
]

_LINE = re.compile(r"^(PASS|FAIL|SKIP) [a-z-]+ \S")


def _assert_no_key(text: str) -> None:
    for i in range(len(SENTINEL) - 7):
        assert SENTINEL[i : i + 8] not in text


class Setup:
    """A workspace, a 0600 key file outside it and a runtime.yaml writer."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.workspace = tmp_path / "work"
        self.workspace.mkdir()
        self.secrets = tmp_path / "secrets"
        self.secrets.mkdir()
        self.key_file = self.secrets / "key"
        self.key_file.write_text(SENTINEL + "\n")
        os.chmod(self.key_file, 0o600)
        self.path = tmp_path / "runtime.yaml"

    def write(self, **overrides) -> str:
        doc = {
            "url": "https://api.example.test",
            "workspace": str(self.workspace),
            "api_key_file": str(self.key_file),
            "harness": "cursor",
        }
        doc.update(overrides)
        self.path.write_text(yaml.safe_dump(doc))
        os.chmod(self.path, 0o600)
        return str(self.path)

    def plant(self, rel: str, content: bytes = SENTINEL.encode()) -> str:
        target = self.workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


RUNTIME_UID = REAL_UID + 1000


@pytest.fixture(autouse=True)
def separate_runtime_user(monkeypatch):
    """tmp_path and its files belong to the test's user. Pretend the runtime is
    another user (RUNTIME_UID) who cannot write the workspace, so the test's
    own files look like the harness's. Each same-user case undoes one piece."""
    if importlib.util.find_spec("mcp") is not None:
        monkeypatch.setattr(doctor, "_runtime_uid", lambda: RUNTIME_UID)
        monkeypatch.setattr(doctor, "_runtime_can_write", lambda ws: False)


def _doctor(setup, *, environ=None, offline=True, transport=None, path=None):
    """Run doctor; check the output contract and that the key never leaks."""
    assert offline or transport is not None, "tests never reach the network"
    out = io.StringIO()
    code = doctor.main(
        path or setup.write(),
        out,
        offline=offline,
        environ={} if environ is None else environ,
        http_transport=transport,
    )
    text = out.getvalue()
    _assert_no_key(text)
    lines = text.splitlines()
    assert text.count("\n") == len(lines)
    for line in lines:
        assert _LINE.match(line), line
    return code, lines


def _line(lines, check_id) -> str:
    found = [line for line in lines if line.split(" ")[1] == check_id]
    assert len(found) == 1, (check_id, lines)
    return found[0]


def _failed(lines) -> list[str]:
    return [line.split(" ")[1] for line in lines if line.startswith("FAIL ")]


# --- the whole run -----------------------------------------------------------


def test_every_check_passes_offline_and_exits_0(setup):
    code, lines = _doctor(setup)
    assert code == 0
    assert [line.split(" ")[1] for line in lines] == IDS
    assert all(line.startswith("PASS ") for line in lines[:-1])
    assert lines[-1] == "SKIP upstream-reachable skipped (--offline)"


def test_any_fail_exits_1(setup):
    code, lines = _doctor(setup, environ={"AGENT_SENTINEL_API_KEY": "x"})
    assert code == 1
    assert _failed(lines) == ["key-not-in-environment"]


@pytest.mark.parametrize("argv", [["doctor", "--bogus"], ["doctor", "--config"]])
def test_unusable_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2


def test_without_the_key_the_isolation_checks_do_not_run(setup):
    os.chmod(setup.key_file, 0o640)
    code, lines = _doctor(setup)
    assert code == 1
    ids = [line.split(" ")[1] for line in lines]
    assert "key-not-in-workspace" not in ids
    assert "upstream-reachable" not in ids


# --- key-file-outside-workspace ----------------------------------------------


def test_key_file_outside_workspace_passes(setup):
    _, lines = _doctor(setup)
    assert _line(lines, "key-file-outside-workspace") == (
        "PASS key-file-outside-workspace api_key_file resolves (realpath) "
        "outside workspace"
    )


def test_key_file_retargeted_into_workspace_fails(setup, monkeypatch):
    # api_key_file is a symlink outside the workspace; the model retargets it
    # into the workspace after the key was read. The realpath re-check sees it.
    link = setup.tmp / "key-link"
    link.symlink_to(setup.key_file)
    inner = setup.plant("copied-key")
    os.chmod(inner, 0o600)
    real_parse = runtime_config.parse_key

    def parse_then_retarget(raw):
        value = real_parse(raw)
        staged = setup.tmp / "key-link.new"
        staged.symlink_to(inner)
        os.replace(staged, link)
        return value

    monkeypatch.setattr(runtime_config, "parse_key", parse_then_retarget)
    code, lines = _doctor(setup, path=setup.write(api_key_file=str(link)))
    assert code == 1
    assert _line(lines, "key-file-location").startswith("PASS ")
    assert _line(lines, "key-file-outside-workspace") == (
        "FAIL key-file-outside-workspace api_key_file must not be inside workspace"
    )


# --- key-file-permissions ----------------------------------------------------


def test_key_file_permissions_pass_at_0600(setup):
    _, lines = _doctor(setup)
    assert _line(lines, "key-file-permissions").startswith("PASS ")


@pytest.mark.parametrize("mode", [0o640, 0o604])
def test_key_file_with_group_or_other_bits_fails(setup, mode):
    os.chmod(setup.key_file, mode)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-file-permissions") == (
        "FAIL key-file-permissions api_key_file must not be readable or writable "
        "by group or others (chmod 600)"
    )


def test_key_file_owned_by_another_user_fails_with_layout_guidance(setup, monkeypatch):
    real_read = runtime_config.read_key_file

    def read_as_another_user(*args):
        # The config file is checked as the real user; the key file is not.
        monkeypatch.setattr(os, "getuid", lambda: REAL_UID + 1)
        return real_read(*args)

    monkeypatch.setattr(runtime_config, "read_key_file", read_as_another_user)
    code, lines = _doctor(setup)
    assert code == 1
    line = _line(lines, "key-file-permissions")
    assert line.startswith(
        "FAIL key-file-permissions api_key_file must be owned by the current user; "
        "the runtime's own user must own it with mode 600"
    )
    assert "a root-owned key file fails this check" in line
    assert "/etc/agentsentinel/key" in line


def test_run_and_config_give_key_owner_guidance(setup, monkeypatch, capsys):
    real_read = runtime_config.read_key_file

    def read_as_another_user(*args):
        monkeypatch.setattr(os, "getuid", lambda: REAL_UID + 1)
        try:
            return real_read(*args)
        finally:
            monkeypatch.setattr(os, "getuid", lambda: REAL_UID)

    monkeypatch.setattr(runtime_config, "read_key_file", read_as_another_user)
    path = setup.write()
    assert cli.main(["config", "--config", path]) == 2
    err = capsys.readouterr().err
    assert "api_key_file must be owned by the current user" in err
    assert "chown" in err and "a root-owned key file fails this check" in err
    assert str(setup.key_file) not in err
    _assert_no_key(err)


# --- runtime-user-separate (the same-uid limit) ------------------------------


def test_runtime_user_separate_passes_and_states_the_limit(setup):
    _, lines = _doctor(setup)
    line = _line(lines, "runtime-user-separate")
    assert line.startswith("PASS ")
    assert "cannot prove the model never runs as the runtime's user" in line


def test_workspace_owned_by_the_runtime_user_fails(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_workspace_owner", lambda ws: RUNTIME_UID)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "runtime-user-separate") == (
        "FAIL runtime-user-separate workspace is owned by the runtime's user; a "
        "model running as this user can read api_key_file wherever it is; run "
        "the harness as a different user from the runtime"
    )


def test_workspace_writable_by_the_runtime_user_fails(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_runtime_can_write", lambda ws: True)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "runtime-user-separate").startswith(
        "FAIL runtime-user-separate workspace is writable by the runtime's user "
        "(group or other permissions)"
    )


@pytest.mark.parametrize("mode", [0o1777, 0o757])
def test_world_writable_workspace_counts_as_writable(setup, monkeypatch, mode):
    # A root-owned 1777 /workspace is the usual container layout: os.access
    # is patched off so only the mode bits can catch it.
    monkeypatch.undo()
    monkeypatch.setattr(os, "access", lambda path, how: False)
    os.chmod(setup.workspace, mode)
    try:
        assert doctor._runtime_can_write(str(setup.workspace)) is True
    finally:
        os.chmod(setup.workspace, 0o755)


def test_group_writable_workspace_in_the_runtimes_group_counts_as_writable(
    setup, monkeypatch
):
    monkeypatch.undo()
    monkeypatch.setattr(os, "access", lambda path, how: False)
    os.chmod(setup.workspace, 0o775)  # the test's own group, which it is in
    try:
        assert doctor._runtime_can_write(str(setup.workspace)) is True
        os.chmod(setup.workspace, 0o755)
        assert doctor._runtime_can_write(str(setup.workspace)) is False
    finally:
        os.chmod(setup.workspace, 0o755)


def test_entries_owned_by_the_runtime_user_fail(setup, monkeypatch):
    # The workspace is someone else's and not writable, but files in it were
    # written by the runtime's user: the model may run as that user.
    monkeypatch.setattr(doctor, "_runtime_uid", lambda: REAL_UID)
    monkeypatch.setattr(doctor, "_workspace_owner", lambda ws: REAL_UID + 5000)
    setup.plant("written-by-the-model.txt", b"clean")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "runtime-user-separate") == (
        "FAIL runtime-user-separate 1 entries under workspace are owned by the "
        "runtime's user, so the model may run as it; a model running as this user "
        "can read api_key_file wherever it is; run the harness as a different "
        "user from the runtime"
    )


def test_runtime_running_as_root_fails_with_an_accurate_reason(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_runtime_uid", lambda: 0)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "runtime-user-separate") == (
        "FAIL runtime-user-separate the runtime runs as root; root bypasses file "
        "permissions, so doctor cannot prove the model is separated; run the "
        "runtime as a dedicated non-root user that owns api_key_file"
    )


def test_workspace_owner_unknown_fails(setup, monkeypatch):
    def cannot_stat(ws):
        raise PermissionError("denied")

    monkeypatch.setattr(doctor, "_workspace_owner", cannot_stat)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "runtime-user-separate").startswith(
        "FAIL runtime-user-separate cannot stat workspace"
    )


# --- key-not-in-workspace ----------------------------------------------------


def test_clean_workspace_passes(setup):
    setup.plant("src/app.py", b"print('hello')\n")
    setup.plant("README.md", b"# readme\n")
    os.mkfifo(setup.workspace / "a-fifo")  # never opened, so it cannot hang
    _, lines = _doctor(setup)
    assert _line(lines, "key-not-in-workspace") == (
        "PASS key-not-in-workspace no file under workspace contains the API key "
        "(2 files checked, 0 links out of workspace to anything but a readable "
        "file not followed, top-level .git/objects not scanned; encoded copies of "
        "the key are not detected)"
    )


def test_key_in_a_file_over_5_mib_fails(setup):
    # The key sits past the old 5 MiB cut-off: large files are scanned now.
    setup.plant("big.bin", b"\0" * (6 * 1024 * 1024) + SENTINEL.encode())
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace big.bin contains the API key"
    )


def test_reading_more_than_the_byte_budget_fails(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_MAX_SCAN_BYTES", 1000)
    setup.plant("clean.bin", b"x" * 2000)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace workspace too large to prove (more than 1000 "
        "bytes to read); narrow workspace"
    )


def test_a_scan_over_the_time_budget_fails(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_MAX_SCAN_SECONDS", -1.0)
    setup.plant("clean.txt", b"clean")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace workspace too large to prove (more than -1 s "
        "to scan, checked between reads); narrow workspace"
    )


def test_git_objects_are_not_scanned(setup):
    setup.plant(".git/objects/ab/cdef")
    setup.plant(".git/HEAD", b"ref: refs/heads/main\n")
    _, lines = _doctor(setup)
    assert _line(lines, "key-not-in-workspace").startswith(
        "PASS key-not-in-workspace no file under workspace contains the API key "
        "(1 files checked"
    )


def test_nested_git_objects_are_scanned(setup):
    # Only the top-level .git/objects is skipped; a nested one is not a
    # place to hide the key.
    setup.plant("vendor/.git/objects/ab/cdef")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace vendor/.git/objects/ab/cdef contains the API key"
    )


def test_symlink_to_an_outside_file_with_the_key_fails(setup):
    # The model reads through the link, so the target is scanned.
    outside = setup.tmp / "outside.txt"
    outside.write_text(SENTINEL)
    (setup.workspace / "out-link").symlink_to(outside)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace out-link is a symlink to a file outside "
        "workspace that contains the API key"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0000 file")
def test_symlink_to_an_unreadable_outside_file_fails(setup):
    hidden = setup.tmp / "hidden.txt"
    hidden.write_text("unknown")
    os.chmod(hidden, 0)
    (setup.workspace / "to-hidden").symlink_to(hidden)
    try:
        code, lines = _doctor(setup)
    finally:
        os.chmod(hidden, 0o600)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace to-hidden is a symlink to a file outside "
        "workspace that doctor could not read"
    )


def test_links_to_one_outside_file_are_read_once(setup):
    target = setup.tmp / "big-clean.bin"
    target.write_bytes(b"x" * 100_000)
    for name in ("a", "b", "c"):
        (setup.workspace / name).symlink_to(target)
    scan = doctor._workspace_scan(str(setup.workspace), str(setup.key_file), SENTINEL)
    assert scan.findings == []
    assert scan.bytes_read == 100_000


def test_symlinks_out_to_a_directory_or_dangling_are_counted(setup):
    clean = setup.tmp / "clean.txt"
    clean.write_text("clean")
    (setup.tmp / "a-dir").mkdir()
    (setup.workspace / "to-clean").symlink_to(clean)
    (setup.workspace / "to-dir").symlink_to(setup.tmp / "a-dir")
    (setup.workspace / "dangling").symlink_to(setup.tmp / "absent")
    code, lines = _doctor(setup)
    assert code == 0
    assert "1 links out of workspace to anything but a readable file not followed" in (
        _line(lines, "key-not-in-workspace")
    )


def test_key_in_a_workspace_file_fails(setup):
    setup.plant("src/notes.txt", b"token = " + SENTINEL.encode() + b"\n")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace src/notes.txt contains the API key"
    )


def test_every_finding_is_counted(setup):
    setup.plant("a.txt")
    setup.plant("b/c.txt")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace").endswith("(2 findings in total)")


def test_key_across_a_read_chunk_boundary_fails(setup):
    padding = b"x" * (doctor._READ_CHUNK - 5)
    setup.plant("straddle.txt", padding + SENTINEL.encode())
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace straddle.txt contains the API key"
    )


def test_symlink_to_the_key_file_fails(setup):
    (setup.workspace / "nested").mkdir()
    (setup.workspace / "nested" / "k").symlink_to(setup.key_file)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace nested/k is a symlink to api_key_file"
    )


def test_symlink_to_the_key_directory_fails(setup):
    (setup.workspace / "secrets").symlink_to(setup.secrets)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace secrets is a symlink to a directory that "
        "contains api_key_file"
    )


def test_too_many_files_fails(setup, monkeypatch):
    monkeypatch.setattr(doctor, "_MAX_SCAN_FILES", 10)
    for i in range(10):
        setup.plant(f"f{i}.txt", b"clean")
    _, lines = _doctor(setup)
    assert _line(lines, "key-not-in-workspace").startswith("PASS ")

    setup.plant("one-more.txt", b"clean")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace workspace too large to prove (more than 10 "
        "files and directories); narrow workspace"
    )


def test_many_empty_directories_hit_the_entry_cap(setup, monkeypatch):
    # Directories count toward the cap: a tree of empty directories cannot
    # make the walk unbounded.
    monkeypatch.setattr(doctor, "_MAX_SCAN_FILES", 10)
    for i in range(11):
        (setup.workspace / f"d{i}").mkdir()
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace workspace too large to prove (more than 10 "
        "files and directories); narrow workspace"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0000 file")
def test_unreadable_file_fails(setup):
    os.chmod(setup.plant("locked.txt", b"clean"), 0)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace 1 files or directories under workspace could "
        "not be read; cannot prove the key is absent"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0000 directory")
def test_unreadable_directory_fails(setup):
    locked = setup.workspace / "locked"
    locked.mkdir()
    os.chmod(locked, 0)
    try:
        code, lines = _doctor(setup)
    finally:
        os.chmod(locked, 0o700)
    assert code == 1
    assert "could not be read" in _line(lines, "key-not-in-workspace")


def test_a_name_holding_the_key_fails_without_naming_it(setup):
    setup.plant(SENTINEL, b"clean")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace a file or directory name under workspace "
        "contains the API key"
    )


def test_a_path_holding_part_of_the_key_is_withheld(setup):
    setup.plant(f"notes-{SENTINEL[3:15]}.txt")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace (message withheld: it contained part of the API key)"
    )


def test_a_newline_in_a_file_name_keeps_one_line(setup):
    setup.plant("a\nb.txt")
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-workspace") == (
        "FAIL key-not-in-workspace a\\x0ab.txt contains the API key"
    )


# --- key-not-in-harness-config -----------------------------------------------


HARNESS_CONFIGS = (
    ".cursor/mcp.json",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".vscode/mcp.json",
)


def test_every_harness_config_from_the_brief_is_checked():
    assert doctor.HARNESS_CONFIGS == HARNESS_CONFIGS


@pytest.mark.parametrize("rel", HARNESS_CONFIGS)
def test_harness_config_without_the_key_passes(setup, rel):
    setup.plant(rel, json.dumps({"mcpServers": {}}).encode())
    code, lines = _doctor(setup)
    assert code == 0
    assert _line(lines, "key-not-in-harness-config").startswith(
        "PASS key-not-in-harness-config none of .cursor/mcp.json, .mcp.json, "
    )


@pytest.mark.parametrize("rel", HARNESS_CONFIGS)
def test_key_planted_in_a_harness_config_fails(setup, rel):
    config = {"mcpServers": {"s": {"env": {"AGENT_SENTINEL_API_KEY": SENTINEL}}}}
    setup.plant(rel, json.dumps(config).encode())
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-harness-config") == (
        f"FAIL key-not-in-harness-config {rel} contains the API key; remove it, "
        "the runtime reads the key from api_key_file"
    )
    # Reported separately from the general scan, which fails too.
    assert _failed(lines) == ["key-not-in-workspace", "key-not-in-harness-config"]


def test_harness_config_symlinked_to_the_key_file_fails(setup):
    (setup.workspace / ".cursor").mkdir()
    (setup.workspace / ".cursor" / "mcp.json").symlink_to(setup.key_file)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-harness-config") == (
        "FAIL key-not-in-harness-config .cursor/mcp.json leads to api_key_file"
    )


def test_harness_config_symlinked_out_to_a_file_with_the_key_fails(setup):
    # The harness follows the link, so this check reads through it.
    outside = setup.tmp / "mcp.json"
    outside.write_text(json.dumps({"key": SENTINEL}))
    (setup.workspace / ".mcp.json").symlink_to(outside)
    code, lines = _doctor(setup)
    assert code == 1
    assert _line(lines, "key-not-in-harness-config").startswith(
        "FAIL key-not-in-harness-config .mcp.json contains the API key"
    )


# --- key-not-in-environment --------------------------------------------------


def test_clean_environment_passes(setup):
    _, lines = _doctor(setup, environ={"PATH": "/usr/bin", "HOME": "/home/x"})
    assert _line(lines, "key-not-in-environment").startswith("PASS ")


def test_env_api_key_variable_fails(setup):
    code, lines = _doctor(setup, environ={"AGENT_SENTINEL_API_KEY": "x"})
    assert code == 1
    assert _line(lines, "key-not-in-environment") == (
        "FAIL key-not-in-environment AGENT_SENTINEL_API_KEY is set; unset it, the "
        "runtime reads the key from api_key_file"
    )


def test_env_value_holding_the_key_fails(setup):
    code, lines = _doctor(setup, environ={"OTHER": SENTINEL})
    assert code == 1
    assert _line(lines, "key-not-in-environment") == (
        "FAIL key-not-in-environment environment variable OTHER holds the API key"
    )


def test_env_api_key_holding_the_key_fails_on_both_counts(setup):
    code, lines = _doctor(setup, environ={"AGENT_SENTINEL_API_KEY": SENTINEL})
    assert code == 1
    line = _line(lines, "key-not-in-environment")
    assert "AGENT_SENTINEL_API_KEY is set" in line
    assert "environment variable AGENT_SENTINEL_API_KEY holds the API key" in line


def test_doctor_reads_its_own_environment_by_default(setup, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_SENTINEL_API_KEY", SENTINEL)
    assert doctor.main(setup.write(), offline=True) == 1
    out, _ = capsys.readouterr()
    _assert_no_key(out)
    assert _line(out.splitlines(), "key-not-in-environment").startswith("FAIL ")


# --- upstream-reachable ------------------------------------------------------


class FakeUpstream:
    """A minimal stand-in for the platform's stateless JSON ``/mcp``."""

    def __init__(self, status: int = 200, body=None) -> None:
        self.status = status
        self.body = body
        self.requests: list[tuple[str, dict, dict]] = []

    def methods(self) -> list[str]:
        return [body.get("method") for _, _, body in self.requests]

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            self.requests.append((str(request.url), dict(request.headers), body))
            if self.status != 200:
                return httpx.Response(self.status, json=self.body)
            if "id" not in body:
                return httpx.Response(202)
            if body["method"] == "initialize":
                result = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "agent-sentinel", "version": "test"},
                }
            else:
                result = {"tools": []}
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}
            )

        return httpx.MockTransport(handler)


def test_upstream_200_passes_and_exits_0(setup):
    fake = FakeUpstream()
    code, lines = _doctor(setup, offline=False, transport=fake.transport())
    assert code == 0
    assert [line.split(" ")[1] for line in lines] == IDS
    assert all(line.startswith("PASS ") for line in lines)
    assert lines[-1] == (
        "PASS upstream-reachable tools/list through the proxy to url succeeded"
    )
    assert "tools/list" in fake.methods()
    for url, headers, _ in fake.requests:
        assert url == "https://api.example.test/mcp"
        assert headers["authorization"] == f"ApiKey {SENTINEL}"


def test_upstream_401_fails_with_the_reason_code(setup):
    fake = FakeUpstream(
        401,
        {"detail": {"reason_code": "invalid_api_key", "approver": "a@example.com"}},
    )
    code, lines = _doctor(setup, offline=False, transport=fake.transport())
    assert code == 1
    assert lines[-1] == (
        "FAIL upstream-reachable tools/list through the proxy failed "
        "(HTTP 401, invalid_api_key)"
    )
    assert "example.com" not in "\n".join(lines)


def test_upstream_redirect_is_not_followed(setup):
    seen = []

    def redirect(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(302, headers={"location": "https://evil.example/mcp"})

    code, lines = _doctor(setup, offline=False, transport=httpx.MockTransport(redirect))
    assert code == 1
    assert lines[-1].startswith("FAIL upstream-reachable ")
    assert seen, "the upstream was called"
    assert all(url == "https://api.example.test/mcp" for url, _ in seen)
    assert not any("evil.example" in url for url, _ in seen)


def test_upstream_connection_error_fails(setup):
    def refuse(request):
        raise httpx.ConnectError("connection refused to secret-host")

    code, lines = _doctor(setup, offline=False, transport=httpx.MockTransport(refuse))
    assert code == 1
    assert lines[-1] == "FAIL upstream-reachable url is unreachable (ConnectError)"


def test_upstream_timeout_fails(setup, monkeypatch):
    class Hang(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await anyio.sleep(30)
            raise AssertionError("not reached")

    monkeypatch.setattr(doctor, "_UPSTREAM_TIMEOUT_SECONDS", 0.2)
    code, lines = _doctor(setup, offline=False, transport=Hang())
    assert code == 1
    assert lines[-1] == (
        "FAIL upstream-reachable tools/list through the proxy timed out after 0.2s"
    )


def test_upstream_is_not_attempted_after_a_failure(setup):
    fake = FakeUpstream()
    setup.plant("leak.txt")
    code, lines = _doctor(setup, offline=False, transport=fake.transport())
    assert code == 1
    assert lines[-1] == (
        "SKIP upstream-reachable not attempted: an earlier check failed"
    )
    assert fake.requests == []


def test_offline_skips_upstream_and_exits_0(setup):
    fake = FakeUpstream()
    code, lines = _doctor(setup, offline=True, transport=fake.transport())
    assert code == 0
    assert lines[-1] == "SKIP upstream-reachable skipped (--offline)"
    assert fake.requests == []


def test_run_checks_returns_structured_results(setup):
    results = doctor.run_checks(setup.write(), offline=True, environ={})
    by_id = {r.check_id: r for r in results}
    assert [r.check_id for r in results] == IDS
    assert by_id["upstream-reachable"].ok is True
    assert by_id["upstream-reachable"].skipped is True
    assert by_id["key-not-in-workspace"].ok is True


@pytest.mark.parametrize(
    "rel",
    [
        f"notes-{SENTINEL[3:15]}.txt",
        f"{SENTINEL[10:18]}/deep/notes.txt",
        f"a\x01{SENTINEL[20:28]}\x7f.txt",
    ],
)
def test_run_checks_withholds_messages_holding_part_of_the_key(setup, rel):
    setup.plant(rel)
    results = doctor.run_checks(setup.write(), offline=True, environ={})
    by_id = {r.check_id: r for r in results}
    assert [r.check_id for r in results] == IDS
    assert by_id["key-not-in-workspace"].ok is False
    assert by_id["key-not-in-workspace"].message == doctor._WITHHELD
    assert by_id["upstream-reachable"].skipped is True
    for r in results:
        _assert_no_key(r.check_id)
        _assert_no_key(r.message)


# --- config.py: the fallback without /proc/self/fd (macOS) -------------------


@pytest.fixture
def no_proc_fd(monkeypatch):
    """Make ``/proc/self/fd`` unreadable, as on macOS; record that it was tried."""
    calls: list[str] = []
    real_readlink = os.readlink

    def readlink(path, *args, **kwargs):
        if str(path).startswith("/proc/self/fd/"):
            calls.append(str(path))
            raise OSError("no /proc here")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(runtime_config.os, "readlink", readlink)
    return calls


def _retarget_on_open(monkeypatch, link, new_target):
    """After ``link`` is opened, atomically point it at ``new_target``."""
    real_open = os.open

    def open_then_retarget(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == str(link):
            staged = str(link) + ".new"
            os.symlink(str(new_target), staged)
            os.replace(staged, str(link))
        return fd

    monkeypatch.setattr(runtime_config.os, "open", open_then_retarget)


def test_fallback_reads_a_valid_key_file(setup, no_proc_fd):
    raw = runtime_config.read_key_file(str(setup.workspace), str(setup.key_file))
    assert raw == (SENTINEL + "\n").encode()
    assert no_proc_fd  # the fallback path ran


def test_fallback_refuses_a_key_symlink_swapped_after_open(
    setup, no_proc_fd, monkeypatch
):
    link = setup.tmp / "key-link"
    link.symlink_to(setup.key_file)
    other = setup.secrets / "other"
    other.write_text(SENTINEL)
    os.chmod(other, 0o600)
    _retarget_on_open(monkeypatch, link, other)
    with pytest.raises(ConfigError) as caught:
        runtime_config.read_key_file(str(setup.workspace), str(link))
    assert (
        str(caught.value) == "api_key_file changed while it was being read; run again"
    )
    assert no_proc_fd
    _assert_no_key(str(caught.value))


def test_fallback_refuses_a_key_symlink_swapped_into_the_workspace(
    setup, no_proc_fd, monkeypatch
):
    link = setup.tmp / "key-link"
    link.symlink_to(setup.key_file)
    inner = setup.plant("key")
    os.chmod(inner, 0o600)
    _retarget_on_open(monkeypatch, link, inner)
    with pytest.raises(ConfigError) as caught:
        runtime_config.read_key_file(str(setup.workspace), str(link))
    assert str(caught.value) == "api_key_file must not be inside workspace"
    assert no_proc_fd
