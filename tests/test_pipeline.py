"""
test_pipeline.py — Testes mínimos do pipeline (rodam sem GPU/modelos).

Os testes exercitam o caminho MOCK do app.py e a configuração, garantindo que
a interface e o contrato das funções continuem válidos mesmo antes do
treinamento do LoRA. Rode com:  pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config  # noqa: E402


def test_config_paths_exist():
    """Diretórios essenciais são criados na importação do config."""
    assert config.DATASET_DIR.exists()
    assert config.OUTPUTS_DIR.exists()
    assert config.LORA_DIR.exists()


def test_style_trigger_default():
    """O trigger token padrão é o definido pela equipe."""
    assert config.STYLE_TRIGGER == "estilo_pixel_art"


def test_two_train_presets_for_etapa2():
    """A Etapa 2 exige comparar pelo menos duas configurações."""
    assert len(config.TRAIN_PRESETS) >= 2


def test_app_runs_in_mock(monkeypatch):
    """run_pipeline deve retornar imagem e áudio mesmo sem modelos reais."""
    # Força o modo mock impedindo a importação do pipeline real.
    monkeypatch.setitem(sys.modules, "scripts.generate_image", None)
    import importlib
    import app
    importlib.reload(app)

    prompt, image, audio, status = app.run_pipeline("uma vila medieval")
    assert image is not None
    assert audio is not None
    assert config.STYLE_TRIGGER in prompt
    assert "✅" in status


def test_empty_theme_is_handled():
    import app
    prompt, image, audio, status = app.run_pipeline("   ")
    assert image is None
    assert "⚠️" in status
