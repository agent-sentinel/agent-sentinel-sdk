"""
Tests for ``agentsentinel-runtime``'s config file and the ``config`` / ``doctor``
subcommands (agent_sentinel.runtime).

The runtime builds on the stdio proxy, which needs the optional ``mcp`` extra
(Python 3.10+), so every test here skips without it, as in test_mcp_proxy.py.

Every error path is checked against a sentinel key: no ConfigError message,
stdout or stderr line may contain it, or a distinctive fragment of it.
"""

from __future__ import annotations

import importlib.util
import os
import stat

import pytest
import yaml

if importlib.util.find_spec("mcp") is not None:
    from agent_sentinel.mcp_proxy import ConfigError
    from agent_sentinel.runtime import cli, doctor
    from agent_sentinel.runtime import config as runtime_config


@pytest.fixture
def _require_mcp():
    pytest.importorskip("mcp")


pytestmark = pytest.mark.usefixtures("_require_mcp")

SENTINEL = "as_sentinel_KEYLEAK_7d41b84a6c"
FRAGMENT = "KEYLEAK"


def _assert_no_key(text: str) -> None:
    assert SENTINEL not in text
    assert FRAGMENT not in text


class Setup:
    """A workspace, a 0600 key file outside it and a runtime.yaml writer."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.workspace = tmp_path / "work"
        self.workspace.mkdir()
        secrets = tmp_path / "secrets"
        secrets.mkdir()
        self.key_file = secrets / "key"
        self.key_file.write_text(SENTINEL + "\n")
        os.chmod(self.key_file, 0o600)
        self.path = tmp_path / "runtime.yaml"

    def doc(self, **overrides):
        doc = {
            "url": "https://api.example.test",
            "workspace": str(self.workspace),
            "api_key_file": str(self.key_file),
            "harness": "cursor",
        }
        doc.update(overrides)
        return {k: v for k, v in doc.items() if v is not _DROP}

    def write(self, doc=None, **overrides) -> str:
        return self.write_raw(
            yaml.safe_dump(doc if doc is not None else self.doc(**overrides))
        )

    def write_raw(self, text: str) -> str:
        self.path.write_text(text)
        os.chmod(self.path, 0o600)
        return str(self.path)


_DROP = object()


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


def _refused(path: str, capsys, match: str) -> str:
    """The path is refused by the loader and by ``config``, without the key."""
    with pytest.raises(ConfigError) as caught:
        runtime_config.load_runtime_config(path)
    message = str(caught.value)
    assert match in message
    assert "\n" not in message
    _assert_no_key(message)

    assert cli.main(["config", "--config", path]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert match in err
    _assert_no_key(err)
    return message


# --- happy path --------------------------------------------------------------


def test_valid_config_loads_the_key_from_the_file(setup):
    cfg = runtime_config.load_runtime_config(setup.write())

    assert cfg.api_key == SENTINEL  # newline stripped
    assert cfg.url == "https://api.example.test"
    assert cfg.workspace == str(setup.workspace)
    assert cfg.api_key_file == str(setup.key_file)
    assert cfg.harness == "cursor"
    assert cfg.approvals == runtime_config.ApprovalsConfig(
        wait=True, timeout_seconds=900, poll_interval_seconds=5
    )
    _assert_no_key(repr(cfg))

    proxy = cfg.proxy_config()
    assert proxy.url == "https://api.example.test/mcp"
    assert proxy.api_key == SENTINEL
    assert proxy.harness == "cursor"
    _assert_no_key(repr(proxy))


def test_optional_fields_default(setup):
    cfg = runtime_config.load_runtime_config(setup.write(url=_DROP, harness=_DROP))
    assert cfg.url == "https://api.agentsentinel.dev"
    assert cfg.harness is None
    assert cfg.proxy_config().url == "https://api.agentsentinel.dev/mcp"


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_key_file_modes_accepted(setup, mode):
    os.chmod(setup.key_file, mode)
    assert runtime_config.load_runtime_config(setup.write()).api_key == SENTINEL


def test_local_http_url_accepted(setup):
    cfg = runtime_config.load_runtime_config(setup.write(url="http://127.0.0.1:8000/"))
    assert cfg.url == "http://127.0.0.1:8000"


def test_approvals_values_are_stored(setup):
    cfg = runtime_config.load_runtime_config(
        setup.write(
            approvals={
                "wait": False,
                "timeout_seconds": 86400,
                "poll_interval_seconds": 300,
            }
        )
    )
    assert cfg.approvals == runtime_config.ApprovalsConfig(
        wait=False, timeout_seconds=86400, poll_interval_seconds=300
    )

    cfg = runtime_config.load_runtime_config(
        setup.write(approvals={"timeout_seconds": 1})
    )
    assert cfg.approvals == runtime_config.ApprovalsConfig(timeout_seconds=1)


def test_config_command_prints_resolved_config_without_the_key(setup, capsys):
    path = setup.write(approvals={"wait": False})

    assert cli.main(["config", "--config", path]) == 0

    out, err = capsys.readouterr()
    assert err == ""
    _assert_no_key(out)
    assert yaml.safe_load(out) == {
        "url": "https://api.example.test",
        "workspace": str(setup.workspace),
        "api_key_file": str(setup.key_file),
        "harness": "cursor",
        "approvals": {
            "wait": False,
            "timeout_seconds": 900,
            "poll_interval_seconds": 5,
        },
    }


# --- discovery ---------------------------------------------------------------


def test_config_path_discovery_order(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_SENTINEL_RUNTIME_CONFIG", "/from/env.yaml")

    assert runtime_config.resolve_config_path("/from/arg.yaml", os.environ) == (
        "/from/arg.yaml"
    )
    assert runtime_config.resolve_config_path(None, os.environ) == "/from/env.yaml"

    monkeypatch.delenv("AGENT_SENTINEL_RUNTIME_CONFIG")
    assert runtime_config.resolve_config_path(None, os.environ) == str(
        tmp_path / ".config" / "agentsentinel" / "runtime.yaml"
    )


def test_config_command_uses_the_env_var_path(setup, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_SENTINEL_RUNTIME_CONFIG", setup.write())
    assert cli.main(["config"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["harness"] == "cursor"


# --- top-level keys ----------------------------------------------------------


@pytest.mark.parametrize("value", ["something", SENTINEL])
def test_api_key_in_config_refused(setup, capsys, value):
    path = setup.write(setup.doc(api_key=value))
    message = _refused(path, capsys, "api_key is not allowed")
    assert "api_key_file" in message


@pytest.mark.parametrize(
    "extra",
    [{"port": 8080}, {"token": SENTINEL}, {SENTINEL: "x"}, {"apikey": SENTINEL}],
)
def test_unknown_key_refused(setup, capsys, extra):
    doc = setup.doc()
    doc.update(extra)
    _refused(setup.write(doc), capsys, "unknown key")


@pytest.mark.parametrize("missing", ["workspace", "api_key_file"])
def test_required_keys(setup, capsys, missing):
    _refused(setup.write(**{missing: _DROP}), capsys, f"{missing} is required")


def test_invalid_yaml_refused_without_echoing_source(setup, capsys):
    path = setup.write_raw(f"workspace: /w\napi_key_file: [{SENTINEL}\n")
    _refused(path, capsys, "not valid YAML")


@pytest.mark.parametrize("text", ["", "- a\n- b\n", f"{SENTINEL}\n"])
def test_non_mapping_document_refused(setup, capsys, text):
    _refused(setup.write_raw(text), capsys, "must be a YAML mapping")


def test_missing_config_file_refused(setup, capsys):
    _refused(str(setup.tmp / "nope.yaml"), capsys, "cannot read config file")


@pytest.mark.parametrize("path", [SENTINEL, f"/tmp/{SENTINEL}", f"{SENTINEL}\nx"])
def test_config_path_holding_a_key_is_not_echoed(setup, capsys, monkeypatch, path):
    _refused(path, capsys, "cannot read config file")

    monkeypatch.setenv(runtime_config.CONFIG_ENV_VAR, path)
    assert cli.main(["run"]) == 2
    _, err = capsys.readouterr()
    assert err.count("\n") == 1
    _assert_no_key(err)

    assert cli.main(["doctor", "--config", path]) == 1
    out, _ = capsys.readouterr()
    assert out.count("\n") == 1
    assert out.startswith("FAIL config-file cannot read config file")
    _assert_no_key(out)


@pytest.mark.parametrize(
    "text",
    [
        "harness: 2026-99-99\nworkspace: /w\napi_key_file: /k\n",
        'workspace: /w\napi_key_file: "/k\\0x"\n',
        'workspace: "/w\\0x"\napi_key_file: /k\n',
    ],
)
def test_loader_value_errors_are_config_errors(setup, capsys, text):
    path = setup.write_raw(text)
    with pytest.raises(ConfigError):
        runtime_config.load_runtime_config(path)
    for command in ("config", "run"):
        assert cli.main([command, "--config", path]) == 2
        out, err = capsys.readouterr()
        assert out == ""
        assert err.count("\n") == 1
        assert "Traceback" not in err
    assert cli.main(["doctor", "--config", path]) == 1
    out, err = capsys.readouterr()
    assert err == ""
    assert "FAIL " in out
    assert all(line.startswith(("PASS ", "FAIL ")) for line in out.splitlines())


# --- paths -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["workspace", "api_key_file"])
@pytest.mark.parametrize("value", ["relative/path", "", "./key"])
def test_relative_paths_refused(setup, capsys, name, value):
    _refused(setup.write(**{name: value}), capsys, f"{name} must be an absolute path")


@pytest.mark.parametrize("name", ["workspace", "api_key_file"])
def test_non_string_paths_refused(setup, capsys, name):
    _refused(setup.write(**{name: [SENTINEL]}), capsys, f"{name} must be a string")


def test_key_file_inside_workspace_refused(setup, capsys):
    inner = setup.workspace / "nested" / "key"
    inner.parent.mkdir()
    inner.write_text(SENTINEL)
    os.chmod(inner, 0o600)
    _refused(
        setup.write(api_key_file=str(inner)), capsys, "must not be inside workspace"
    )


def test_key_file_equal_to_workspace_refused(setup, capsys):
    _refused(
        setup.write(api_key_file=str(setup.workspace)),
        capsys,
        "must not be inside workspace",
    )


def test_symlink_outside_pointing_into_workspace_refused(setup, capsys):
    inner = setup.workspace / "key"
    inner.write_text(SENTINEL)
    os.chmod(inner, 0o600)
    link = setup.tmp / "secrets" / "link"
    link.symlink_to(inner)
    _refused(
        setup.write(api_key_file=str(link)), capsys, "must not be inside workspace"
    )


def test_symlinked_workspace_containing_the_key_refused(setup, capsys):
    # The workspace is a symlink to the directory that holds the key file.
    link = setup.tmp / "work-link"
    link.symlink_to(setup.key_file.parent)
    _refused(setup.write(workspace=str(link)), capsys, "must not be inside workspace")


def test_symlink_inside_workspace_pointing_out_refused(setup, capsys):
    # The model can read the key through a link in its workspace.
    link = setup.workspace / "key-link"
    link.symlink_to(setup.key_file)
    _refused(
        setup.write(api_key_file=str(link)), capsys, "must not be inside workspace"
    )


def test_symlink_retargeted_into_workspace_after_check_refused(
    setup, capsys, monkeypatch
):
    # An outside symlink passes the location check, then is pointed at an
    # owner-only file inside the workspace just before open().
    inner = setup.workspace / "key"
    inner.write_text(SENTINEL)
    os.chmod(inner, 0o600)
    link = setup.tmp / "secrets" / "link"
    link.symlink_to(setup.key_file)
    real_open = os.open
    swapped = []

    def swapping_open(path, *args, **kwargs):
        if str(path) == str(link):
            link.unlink()
            link.symlink_to(inner)
            swapped.append(True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(runtime_config.os, "open", swapping_open)
    with pytest.raises(ConfigError) as caught:
        runtime_config.load_runtime_config(setup.write(api_key_file=str(link)))
    assert swapped == [True]  # refused after open(), not by the first check
    assert str(caught.value) == "api_key_file must not be inside workspace"


def test_hard_linked_key_file_refused(setup, capsys):
    os.link(setup.key_file, setup.workspace / "key")
    _refused(setup.write(), capsys, "must not have other hard links")


def test_workspace_prefix_sibling_is_not_inside(setup):
    # /tmp/x/work-other is not inside /tmp/x/work.
    sibling = setup.tmp / "work-other"
    sibling.mkdir()
    key = sibling / "key"
    key.write_text(SENTINEL)
    os.chmod(key, 0o600)
    cfg = runtime_config.load_runtime_config(setup.write(api_key_file=str(key)))
    assert cfg.api_key == SENTINEL


# --- key file ----------------------------------------------------------------


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o620, 0o602, 0o610])
def test_group_or_world_accessible_key_file_refused(setup, capsys, mode):
    os.chmod(setup.key_file, mode)
    _refused(
        setup.write(), capsys, "must not be readable or writable by group or others"
    )


def test_key_file_directory_refused(setup, capsys):
    directory = setup.tmp / "secrets" / "dir"
    directory.mkdir(mode=0o700)
    _refused(setup.write(api_key_file=str(directory)), capsys, "must be a regular file")


def test_key_file_fifo_refused_without_blocking(setup, capsys):
    fifo = setup.tmp / "secrets" / "fifo"
    os.mkfifo(fifo, 0o600)
    _refused(setup.write(api_key_file=str(fifo)), capsys, "must be a regular file")


def test_missing_key_file_refused(setup, capsys):
    _refused(
        setup.write(api_key_file=str(setup.tmp / "secrets" / "absent")),
        capsys,
        "cannot read api_key_file",
    )


def test_key_file_owned_by_another_user_refused(setup, capsys, monkeypatch):
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    _refused(setup.write(), capsys, "must be owned by the current user")


@pytest.mark.parametrize(
    "content",
    [
        b"as_sentinel KEYLEAK_7d41b84a6c\n",
        "as_sentinel_KEYLEAK_7d41b84a6cé\n".encode(),
        b"as_sentinel_KEYLEAK_\xff7d41b84a6c",
        b"as_sentinel_KEYLEAK\x007d41b84a6c",
    ],
)
def test_unsendable_key_refused(setup, capsys, content):
    setup.key_file.write_bytes(content)
    _refused(setup.write(), capsys, "api_key_file contains characters")


@pytest.mark.parametrize("content", [b"", b"  \n\t\n"])
def test_empty_key_file_refused(setup, capsys, content):
    setup.key_file.write_bytes(content)
    _refused(setup.write(), capsys, "api_key_file is empty")


def test_oversized_key_file_refused(setup, capsys):
    setup.key_file.write_bytes((SENTINEL * 200).encode())
    _refused(setup.write(), capsys, "larger than 4096 bytes")


# --- url and harness ---------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("http://api.agentsentinel.dev", "url must use https"),
        ("http://10.0.0.5:8000", "url must use https"),
        ("http://localhost.evil.example", "url must use https"),
        ("ftp://localhost", "url must use https"),
        (f"https://{SENTINEL}@api.example.test", "url must not contain credentials"),
        (SENTINEL, "url must be an absolute http(s) URL"),
        ("https://api.example.test/?k=v", "url must not contain a query or fragment"),
        ("https://api.example.test:99999", "url is not a valid URL"),
        (8080, "url must be a string"),
    ],
)
def test_bad_url_refused(setup, capsys, url, match):
    _refused(setup.write(url=url), capsys, match)


@pytest.mark.parametrize("harness", ["cur\x07sor", "cur\nsor", "cursör", 7])
def test_bad_harness_refused(setup, capsys, harness):
    message = _refused(setup.write(harness=harness), capsys, "harness")
    assert "cursor" not in message


def test_key_pasted_as_harness_refused(setup, capsys):
    _refused(setup.write(harness=SENTINEL), capsys, "harness")
    assert cli.main(["run", "--config", setup.write(harness=SENTINEL)]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    _assert_no_key(err)


@pytest.mark.parametrize("harness", ["as_cursor", "as_other_key_123"])
def test_harness_with_key_like_prefix_is_accepted(setup, harness):
    # Only the real key is refused; a header-safe name that merely starts
    # like a key is a valid harness under mcp_proxy's rules.
    cfg = runtime_config.load_runtime_config(setup.write(harness=harness))
    assert cfg.harness == harness


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("url", f"https://api.example.test/{SENTINEL}"),
        ("harness", f"cursor-{SENTINEL}"),
    ],
)
def test_key_in_a_displayed_setting_refused(setup, capsys, name, value):
    _refused(
        setup.write(**{name: value}), capsys, f"{name} must not contain the API key"
    )

    assert cli.main(["doctor", "--config", setup.write(**{name: value})]) == 1
    out, _ = capsys.readouterr()
    assert f"FAIL config-no-key {name} must not contain the API key" in out.splitlines()
    _assert_no_key(out)


def test_key_in_key_file_path_refused(setup, capsys):
    named = setup.tmp / "secrets" / SENTINEL
    named.write_text(SENTINEL)
    os.chmod(named, 0o600)
    _refused(
        setup.write(api_key_file=str(named)),
        capsys,
        "api_key_file must not contain the API key",
    )


# --- approvals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("approvals", "match"),
    [
        ({"wait": "yes"}, "approvals.wait must be true or false"),
        ({"wait": 1}, "approvals.wait must be true or false"),
        ({"timeout_seconds": 0}, "approvals.timeout_seconds must be an integer"),
        ({"timeout_seconds": 86401}, "approvals.timeout_seconds must be an integer"),
        ({"timeout_seconds": True}, "approvals.timeout_seconds must be an integer"),
        ({"timeout_seconds": "900"}, "approvals.timeout_seconds must be an integer"),
        ({"timeout_seconds": 1.5}, "approvals.timeout_seconds must be an integer"),
        ({"poll_interval_seconds": 0}, "approvals.poll_interval_seconds must be"),
        ({"poll_interval_seconds": 301}, "approvals.poll_interval_seconds must be"),
        ({"poll_interval_seconds": False}, "approvals.poll_interval_seconds must be"),
        ({"notify": True}, "approvals has an unknown key"),
        ({SENTINEL: 1}, "approvals has an unknown key"),
        ([SENTINEL], "approvals must be a mapping"),
        (SENTINEL, "approvals must be a mapping"),
    ],
)
def test_bad_approvals_refused(setup, capsys, approvals, match):
    _refused(setup.write(approvals=approvals), capsys, match)


# --- doctor ------------------------------------------------------------------

DOCTOR_IDS = [
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
]


def test_doctor_valid_config_passes(setup, capsys):
    assert cli.main(["doctor", "--config", setup.write()]) == 0

    out, err = capsys.readouterr()
    assert err == ""
    _assert_no_key(out)
    lines = out.splitlines()
    assert all(line.startswith("PASS ") for line in lines)
    assert [line.split(" ")[1] for line in lines] == DOCTOR_IDS


def test_doctor_world_readable_key_file_fails(setup, capsys):
    os.chmod(setup.key_file, 0o644)

    assert cli.main(["doctor", "--config", setup.write()]) == 1

    out, _ = capsys.readouterr()
    _assert_no_key(out)
    lines = out.splitlines()
    failed = [line for line in lines if line.startswith("FAIL ")]
    assert failed == [
        "FAIL key-file-permissions api_key_file must not be readable or writable "
        "by group or others (chmod 600)"
    ]
    assert all(line.startswith(("PASS ", "FAIL ")) for line in lines)
    # Content is not read from a file that failed the permission check.
    assert "key-file-content" not in out


def test_doctor_key_inside_workspace_fails(setup, capsys):
    link = setup.workspace / "key-link"
    link.symlink_to(setup.key_file)
    assert cli.main(["doctor", "--config", setup.write(api_key_file=str(link))]) == 1
    out, _ = capsys.readouterr()
    assert "FAIL key-file-location api_key_file must not be inside workspace" in (
        out.splitlines()
    )
    _assert_no_key(out)


def test_doctor_missing_config_fails(setup, capsys):
    assert cli.main(["doctor", "--config", str(setup.tmp / "absent.yaml")]) == 1
    out, _ = capsys.readouterr()
    lines = out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("FAIL config-file cannot read config file")


def test_doctor_api_key_in_config_fails_without_the_key(setup, capsys):
    path = setup.write(setup.doc(api_key=SENTINEL))
    assert cli.main(["doctor", "--config", path]) == 1
    out, _ = capsys.readouterr()
    assert out.startswith("FAIL config-file api_key is not allowed")
    _assert_no_key(out)


def test_doctor_bad_path_still_checks_the_rest(setup, capsys):
    path = setup.write(api_key_file="relative", url="ftp://x")
    assert cli.main(["doctor", "--config", path]) == 1
    out, _ = capsys.readouterr()
    ids = [(line.split(" ")[0], line.split(" ")[1]) for line in out.splitlines()]
    assert ids == [
        ("PASS", "config-file"),
        ("FAIL", "config-paths"),
        ("FAIL", "config-url"),
        ("PASS", "config-harness"),
        ("PASS", "config-approvals"),
    ]


def test_doctor_check_results_are_structured(setup):
    os.chmod(setup.key_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    results = doctor.run_checks(setup.write())
    by_id = {r.check_id: r for r in results}
    assert by_id["key-file-permissions"].ok is False
    assert doctor.format_line(by_id["config-file"]).startswith("PASS config-file ")


@pytest.mark.parametrize(
    "argv", [["doctor", "--bogus"], ["doctor", "--config"], [], ["nope"]]
)
def test_unusable_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--api-key", SENTINEL],
        ["run", SENTINEL],
        ["config", f"--{SENTINEL}"],
        ["doctor", "--config", "x", SENTINEL],
        [SENTINEL],
        [f"--api-key={SENTINEL}"],
    ],
)
def test_argument_errors_never_print_the_key(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2
    out, err = capsys.readouterr()
    _assert_no_key(out + err)
    assert err.strip() == cli.USAGE_ERROR


# --- the config file itself --------------------------------------------------
# A model that can edit runtime.yaml could point url at another host and have
# the key sent there, so the file gets the key file's rules.


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o666])
def test_config_file_writable_by_group_or_others_refused(setup, capsys, mode):
    path = setup.write()
    os.chmod(path, mode)
    _refused(path, capsys, "config file must not be writable by group or others")


def test_redirected_url_in_a_writable_config_never_runs(setup, monkeypatch, capsys):
    path = setup.write(url="https://attacker.example.test")
    os.chmod(path, 0o664)
    started = []

    async def fake_run_stdio(cfg):
        started.append(cfg)

    monkeypatch.setattr(cli, "_run_stdio", fake_run_stdio)
    assert cli.main(["run", "--config", path]) == 2
    assert started == []
    out, err = capsys.readouterr()
    assert "attacker" not in out + err
    _assert_no_key(out + err)


def test_config_file_inside_workspace_refused(setup, capsys):
    path = setup.workspace / "runtime.yaml"
    path.write_text(yaml.safe_dump(setup.doc()))
    os.chmod(path, 0o600)
    _refused(str(path), capsys, "config file must not be inside workspace")


def test_config_file_symlink_into_workspace_refused(setup, capsys):
    inner = setup.workspace / "runtime.yaml"
    inner.write_text(yaml.safe_dump(setup.doc()))
    os.chmod(inner, 0o600)
    outside = setup.tmp / "outside.yaml"
    outside.symlink_to(inner)
    _refused(str(outside), capsys, "config file must not be inside workspace")


def test_config_symlink_inside_workspace_pointing_out_refused(setup, capsys):
    # The model can retarget a symlink it can write, so the path as given must
    # be outside the workspace too, not only the file it resolves to.
    real = setup.write()
    link = setup.workspace / "runtime.yaml"
    link.symlink_to(real)
    _refused(str(link), capsys, "config file must not be inside workspace")


def test_config_file_made_writable_during_the_read_refused(setup, monkeypatch, capsys):
    path = setup.write()
    real_read = os.read

    def read_then_chmod(fd, n):
        data = real_read(fd, n)
        os.chmod(path, 0o620)
        return data

    monkeypatch.setattr(os, "read", read_then_chmod)
    _refused(path, capsys, "config file must not be writable by group or others")


def test_config_file_with_another_hard_link_refused(setup, capsys):
    path = setup.write()
    os.link(path, setup.workspace / "linked.yaml")
    _refused(path, capsys, "config file must not have other hard links")


def test_config_file_must_be_a_regular_file(setup, capsys):
    directory = setup.tmp / "a-directory"
    directory.mkdir()
    _refused(str(directory), capsys, "config file must be a regular file")


def test_config_file_owned_by_another_user_refused(setup, monkeypatch, capsys):
    path = setup.write()
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    _refused(path, capsys, "config file must be owned by the current user")


def test_config_file_swapped_after_reading_refused(setup):
    path = setup.write()
    doc = runtime_config.read_document(path)
    replacement = setup.tmp / "replacement.yaml"
    replacement.write_text(yaml.safe_dump(setup.doc()))
    os.chmod(replacement, 0o600)
    os.replace(replacement, path)
    with pytest.raises(
        ConfigError, match="config file changed while it was being read"
    ):
        runtime_config.check_config_location(doc, str(setup.workspace))


def test_config_file_location_not_bound_to_the_open_file_refused(
    setup, monkeypatch, capsys
):
    path = setup.write()
    monkeypatch.setattr(runtime_config, "_opened_location", lambda fd, p, st: None)
    _refused(path, capsys, "config file changed while it was being read")


def test_doctor_config_file_inside_workspace_fails(setup, capsys):
    path = setup.workspace / "runtime.yaml"
    path.write_text(yaml.safe_dump(setup.doc()))
    os.chmod(path, 0o600)
    assert cli.main(["doctor", "--config", str(path)]) == 1
    out, _ = capsys.readouterr()
    assert "FAIL config-file-location config file must not be inside workspace" in out
    _assert_no_key(out)


def test_doctor_writable_config_file_fails(setup, capsys):
    path = setup.write()
    os.chmod(path, 0o666)
    assert cli.main(["doctor", "--config", path]) == 1
    out, _ = capsys.readouterr()
    assert out.startswith("FAIL config-file config file must not be writable")
    _assert_no_key(out)


def test_oversized_config_file_refused(setup, capsys):
    padding = "#" + "x" * (70 * 1024) + "\n"
    path = setup.write_raw(padding + yaml.safe_dump(setup.doc()))
    _refused(path, capsys, "config file is larger than 65536 bytes")


def test_yaml_python_tags_are_refused(setup, capsys):
    marker = setup.tmp / "tag-ran"
    path = setup.write_raw(
        yaml.safe_dump(setup.doc())
        + f"harness: !!python/object/apply:os.system ['touch {marker}']\n"
    )
    _refused(path, capsys, "config file is not valid YAML")
    assert not marker.exists()


# --- workspace ---------------------------------------------------------------


def test_missing_workspace_refused(setup, capsys):
    path = setup.write(workspace=str(setup.tmp / "does-not-exist"))
    _refused(path, capsys, "workspace must be an existing directory")


def test_workspace_that_is_a_file_refused(setup, capsys):
    a_file = setup.tmp / "not-a-dir"
    a_file.write_text("")
    path = setup.write(workspace=str(a_file))
    _refused(path, capsys, "workspace must be an existing directory")


def test_key_inside_workspace_detected_by_identity_not_spelling(
    setup, monkeypatch, capsys
):
    # A second spelling of the workspace (as on a case-insensitive file system)
    # that realpath() does not normalise: only the directory identity matches.
    alias = setup.tmp / "WORK"
    alias.symlink_to(setup.workspace)
    key = setup.workspace / "key"
    key.write_text(SENTINEL)
    os.chmod(key, 0o600)
    path = setup.write(api_key_file=str(alias / "key"))
    monkeypatch.setattr(
        runtime_config.os.path, "realpath", lambda p, **kw: os.path.normpath(p)
    )
    # No /proc/self/fd, as on macOS, so only the location check itself can catch it.
    real_readlink = os.readlink

    def no_proc(path, *args, **kwargs):
        if str(path).startswith("/proc/self/fd/"):
            raise OSError("no /proc")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(runtime_config.os, "readlink", no_proc)
    _refused(path, capsys, "api_key_file must not be inside workspace")
