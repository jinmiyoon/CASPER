import json

from casper.user_config import _deep_merge, _load_user_config, _resolve_path, get


def test_resolve_path_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CASPER_TEST_PATH", raising=False)
    default = tmp_path
    assert _resolve_path("CASPER_TEST_PATH", default) == str(default)


def test_resolve_path_env_set_to_existing_dir(tmp_path, monkeypatch):
    target = tmp_path / "exists"
    target.mkdir()
    monkeypatch.setenv("CASPER_TEST_PATH", str(target))
    assert _resolve_path("CASPER_TEST_PATH", tmp_path) == str(target)


def test_resolve_path_env_set_to_missing_dir_falls_back(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setenv("CASPER_TEST_PATH", str(missing))
    result = _resolve_path("CASPER_TEST_PATH", tmp_path)
    assert result == str(tmp_path)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_deep_merge_overrides_nested_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}, "c": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    override = {"a": {"x": 2}}
    _deep_merge(base, override)
    assert base == {"a": {"x": 1}}


def test_load_user_config_missing_file_returns_defaults(tmp_path):
    config = _load_user_config(config_path=tmp_path / "does_not_exist.json")
    assert config["io_paths"]["output_file_name"] == "sample"


def test_load_user_config_invalid_json_returns_defaults(tmp_path):
    bad_file = tmp_path / "user_config.json"
    bad_file.write_text("{not valid json")
    config = _load_user_config(config_path=bad_file)
    assert config["io_paths"]["output_file_name"] == "sample"


def test_load_user_config_merges_valid_json(tmp_path):
    custom_file = tmp_path / "user_config.json"
    custom_file.write_text(json.dumps({"io_paths": {"output_file_name": "custom_name"}}))
    config = _load_user_config(config_path=custom_file)
    assert config["io_paths"]["output_file_name"] == "custom_name"
    # Unspecified keys still come from DEFAULTS
    assert config["io_paths"]["plot"] is True


def test_get_simple_dot_path_lookup():
    assert get("io_paths.plot") is True


def test_get_missing_path_returns_default():
    assert get("does.not.exist", default="fallback") == "fallback"


def test_get_empty_path_returns_full_config():
    assert isinstance(get(""), dict)
