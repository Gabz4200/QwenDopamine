"""Behavioural tests for the CLI dispatcher and entrypoint.

The CLI is a thin Hydra wrapper around :class:`HFIntegration`. We do not
launch the real CLI here (it would require a Hub download) — instead we
exercise the config-resolution helper, the package ``main`` dispatcher,
and the script entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from qwendopamine import DEFAULT_QWEN35_REPO
from qwendopamine.cli.train import _get_cfg, main


def test_when_get_cfg_with_single_key_then_returns_value() -> None:
    cfg = OmegaConf.create({"a": 1})
    assert _get_cfg(cfg, "a", default=99) == 1


def test_when_get_cfg_with_multiple_keys_then_first_match_wins() -> None:
    cfg = OmegaConf.create({"a": 1, "b": 2})
    assert _get_cfg(cfg, "a", "b", default=99) == 1
    assert _get_cfg(cfg, "missing", "b", default=99) == 2


def test_when_get_cfg_with_no_keys_match_then_returns_default() -> None:
    cfg = OmegaConf.create({"a": 1})
    assert _get_cfg(cfg, "missing", default="default-value") == "default-value"


def test_when_get_cfg_with_dotted_path_then_traverses() -> None:
    cfg = OmegaConf.create({"train": {"device": "cuda", "lr": 1e-3}})
    assert _get_cfg(cfg, "train.device") == "cuda"
    assert _get_cfg(cfg, "train.lr") == 1e-3
    assert _get_cfg(cfg, "train.missing", default="cpu") == "cpu"


def test_when_package_main_called_then_dispatches_to_cli_train(monkeypatch) -> None:
    """``qwendopamine.main`` (the console-script entrypoint) must import and
    call ``qwendopamine.cli.train.main``."""
    import qwendopamine

    called = {"n": 0}

    def _fake() -> None:
        called["n"] += 1

    monkeypatch.setattr("qwendopamine.cli.train.main", _fake)
    qwendopamine.main()
    assert called["n"] == 1


def test_when_main_runs_with_empty_config_then_loads_default_model(monkeypatch) -> None:
    """``main`` must read the base model and device from the config (with
    defaults), then call :func:`HFIntegration.load_model` with them."""
    captured: dict = {}

    def _fake_load_model(name, **kwargs):  # pyrefly: ignore[unannotated-return]
        captured["name"] = name
        captured["kwargs"] = kwargs
        return mock.MagicMock(__class__=type("FakeModel", (), {}))

    monkeypatch.setattr(
        "qwendopamine.cli.train.HFIntegration.load_model", _fake_load_model
    )
    monkeypatch.setattr(
        "qwendopamine.cli.train.HFIntegration.make_quantization_config",
        lambda method: mock.MagicMock(name="qconfig"),  # pyrefly: ignore[implicit-any-lambda]
    )

    cfg = OmegaConf.create({})  # no overrides
    main.__wrapped__(cfg) if hasattr(main, "__wrapped__") else main(cfg)
    assert captured["name"] == DEFAULT_QWEN35_REPO
    assert captured["kwargs"]["device_map"] == "cpu"
    assert captured["kwargs"]["quantization_config"] is None


def test_when_main_runs_with_quantization_enabled_then_config_is_built(
    monkeypatch,
) -> None:
    captured: dict = {}

    def _fake_load_model(name, **kwargs):  # pyrefly: ignore[unannotated-return]
        captured["name"] = name
        captured["kwargs"] = kwargs
        return mock.MagicMock(__class__=type("FakeModel", (), {}))

    monkeypatch.setattr(
        "qwendopamine.cli.train.HFIntegration.load_model", _fake_load_model
    )
    monkeypatch.setattr(
        "qwendopamine.cli.train.HFIntegration.make_quantization_config",
        lambda method: {"method": method},  # pyrefly: ignore[implicit-any-lambda]
    )

    cfg = OmegaConf.create({"quantization": {"enabled": True, "method": "int4"}})
    main.__wrapped__(cfg) if hasattr(main, "__wrapped__") else main(cfg)
    assert captured["kwargs"]["quantization_config"] == {"method": "int4"}


def test_when_main_runs_with_custom_base_model_then_uses_override(monkeypatch) -> None:
    captured: dict = {}

    def _fake_load_model(name, **kwargs):  # pyrefly: ignore[unannotated-return]
        captured["name"] = name
        captured["kwargs"] = kwargs
        return mock.MagicMock(__class__=type("FakeModel", (), {}))

    monkeypatch.setattr(
        "qwendopamine.cli.train.HFIntegration.load_model", _fake_load_model
    )
    cfg = OmegaConf.create(
        {"model": {"base_model": "my-org/my-model", "device": "cuda"}}
    )
    main.__wrapped__(cfg) if hasattr(main, "__wrapped__") else main(cfg)
    assert captured["name"] == "my-org/my-model"
    assert captured["kwargs"]["device_map"] == "cuda"


def test_when_main_script_entry_exists_then_runs() -> None:
    """Sanity: the ``__main__`` block is reachable (it just calls ``main``)."""
    src = Path("src/qwendopamine/cli/train.py").read_text()
    assert 'if __name__ == "__main__"' in src
    assert "main()" in src
