"""
config.py — Configuração central do Ateliê Generativo.

DECISÃO ARQUITETURAL
--------------------
Centralizamos TODOS os caminhos e hiperparâmetros aqui para:
  * Reprodutibilidade (um único ponto de verdade).
  * Trocar facilmente entre ambientes (local / Colab / HF Spaces).
  * Permitir que scripts e app.py importem a mesma configuração.

Segredos (tokens da Hugging Face, etc.) NUNCA ficam neste arquivo, pois o
repositório é PÚBLICO. Eles são lidos de variáveis de ambiente / Secrets do
Hugging Face Spaces. Expor chaves no código público zera o critério de
segurança da sistematização.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
FONTES_CSV = DATASET_DIR / "fontes.csv"

MODELS_DIR = BASE_DIR / "models"
LORA_DIR = MODELS_DIR / "lora"          # pesos LoRA treinados pela equipe

OUTPUTS_DIR = BASE_DIR / "outputs"      # imagens/áudios gerados em runtime

for _d in (DATASET_IMAGES_DIR, LORA_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelo base e LoRA
# ---------------------------------------------------------------------------
# Modelo base de difusão. SD 1.5 é leve e treina bem com LoRA na GPU T4 (Colab).
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "runwayml/stable-diffusion-v1-5")

# Onde carregar o LoRA: ou de um repo do Hub, ou do diretório local models/lora.
# Se LORA_REPO_ID estiver definido, tem prioridade (útil no HF Spaces).
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "")          # ex.: "equipe/atelie-pixelart-lora"
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")
LORA_LOCAL_PATH = LORA_DIR                            # fallback local

# Palavra-gatilho de estilo (trigger token). Definida pela equipe seguindo o
# padrão sugerido pelo professor (ex.: "estilo_cordel, ..."). TODA caption do
# dataset começa com este token e ele é injetado no prompt em inferência.
STYLE_TRIGGER = os.getenv("STYLE_TRIGGER", "estilo_pixel_art")

# ---------------------------------------------------------------------------
# Modelo de linguagem (expansão de prompt)
# ---------------------------------------------------------------------------
# LLM leve para enriquecer o tema do usuário. Pode ser trocado por um modelo
# instruct maior se houver GPU disponível.
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "google/flan-t5-base")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "80"))

# ---------------------------------------------------------------------------
# TTS (texto -> áudio)
# ---------------------------------------------------------------------------
# Backends suportados em generate_audio.py: "gtts" (leve, online),
# "speecht5" (transformers, offline) ou "coqui" (TTS).
TTS_BACKEND = os.getenv("TTS_BACKEND", "gtts")
TTS_LANG = os.getenv("TTS_LANG", "pt")

# ---------------------------------------------------------------------------
# Parâmetros de TREINAMENTO (LoRA)
# ---------------------------------------------------------------------------
# A Etapa 2 da sistematização exige testar PELO MENOS DUAS configurações
# distintas e justificar os hiperparâmetros. Mantemos um dict de presets.
TRAIN_CONFIG = {
    "resolution": 512,            # imagens >= 512x512 (exigência do trabalho)
    "train_batch_size": 1,        # cabe na T4 (16GB); aumentar se houver VRAM
    "gradient_accumulation_steps": 4,
    "max_train_steps": 1500,
    "learning_rate": 1e-4,        # típico para LoRA em SD
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 50,
    "lora_rank": 16,              # rank do adaptador (capacidade vs. tamanho)
    "lora_alpha": 16,
    "seed": 42,                   # reprodutibilidade
    "mixed_precision": "fp16",
    "caption_column": "description",
}

# Presets para a comparação exigida na Etapa 2 (rank baixo x rank alto, etc.).
TRAIN_PRESETS = {
    "config_a_rank8": {**TRAIN_CONFIG, "lora_rank": 8, "lora_alpha": 8,
                       "max_train_steps": 1000, "learning_rate": 1e-4},
    "config_b_rank16": {**TRAIN_CONFIG, "lora_rank": 16, "lora_alpha": 16,
                        "max_train_steps": 1500, "learning_rate": 1e-4},
}

# ---------------------------------------------------------------------------
# Parâmetros de INFERÊNCIA
# ---------------------------------------------------------------------------
INFER_CONFIG = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "lora_scale": 0.8,            # peso do LoRA na geração
    # Negative prompt para evitar artefatos comuns e manter estética 16-bit.
    "negative_prompt": (
        "blurry, low quality, jpeg artifacts, photorealistic, 3d render, "
        "watermark, text, signature, deformed"
    ),
}

# ---------------------------------------------------------------------------
# Segredos / tokens (lidos do ambiente — NUNCA hardcode)
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")  # configurar em Spaces > Settings > Secrets


def device() -> str:
    """Retorna o device disponível ('cuda' ou 'cpu'). Importa torch tardiamente."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
