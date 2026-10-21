from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import model_config


def test_model_path_defaults_to_project_data(monkeypatch, tmp_path):
    monkeypatch.delenv("WORD_VECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(model_config, "ROOT", tmp_path)
    assert model_config.model_path() == tmp_path / "data" / model_config.FILENAME


def test_missing_models_explains_full_release(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "model_path", lambda: tmp_path / "missing.bin")
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--check"])
    with pytest.raises(SystemExit, match="Releases"):
        bootstrap.main()


def test_incomplete_environment_reinstalls_dependencies(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    model.touch()
    Path(str(model) + ".vectors.npy").touch()
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path / "requirements.txt").write_text("streamlit==1.66.0\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "model_path", lambda: model)
    monkeypatch.setattr(bootstrap, "requirements_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--check"])

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "--dependencies-ready" in command else 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert bootstrap.main() == 0
    assert any(command[1:5] == ["-m", "pip", "install", "-r"] for command in calls)
