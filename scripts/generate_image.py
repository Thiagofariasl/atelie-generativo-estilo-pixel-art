"""
generate_image.py — Expansão de prompt (LLM) + geração de imagem (SD + LoRA).

Responsável por DUAS etapas do pipeline:
  1. expand_prompt(theme): usa um LLM para transformar um tema curto em um
     prompt rico, sempre prefixado pelo token de estilo (config.STYLE_TRIGGER).
  2. generate_image(prompt): carrega Stable Diffusion + LoRA e gera a imagem.

DECISÃO ARQUITETURAL
--------------------
* Os pipelines (LLM e difusão) são carregados de forma LAZY e cacheados em
  variáveis de módulo, para não pagar o custo de carregar pesos a cada chamada
  (importante na latência da UI Gradio).
* Pode ser usado tanto importado (pelo app.py) quanto via CLI:
      python -m scripts.generate_image --theme "uma vila medieval"
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import config

# Caches de módulo (preenchidos sob demanda).
_llm = None
_sd_pipe = None


# ---------------------------------------------------------------------------
# 1) Expansão de prompt via LLM
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_llm():
    """Carrega o pipeline de geração de texto (text2text)."""
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model=config.LLM_MODEL_ID,
        device=0 if config.device() == "cuda" else -1,
    )


def expand_prompt(theme: str) -> str:
    """Expande um tema curto em um prompt detalhado de pixel art 16-bit.

    O token de estilo (estilo_pixel_art) é SEMPRE adicionado ao início, pois é
    a palavra-gatilho que ativa o LoRA treinado pela equipe.
    """
    instruction = (
        "Expand the following theme into a short, vivid English image-generation "
        "prompt for a 16-bit pixel art game scene. Describe setting, colors and "
        f"mood. Theme: {theme}"
    )
    try:
        llm = _load_llm()
        generated = llm(instruction, max_new_tokens=config.LLM_MAX_NEW_TOKENS)[0]
        body = generated["generated_text"].strip()
    except Exception as exc:  # noqa: BLE001
        # Fallback determinístico se o LLM não estiver disponível.
        print(f"[generate_image] LLM indisponível ({exc}); usando template.")
        body = (
            f"{theme}, detailed 16-bit pixel art, vibrant retro game scene, "
            f"crisp pixels, limited color palette, side-scroller background"
        )

    # Garante o trigger token no início e remove duplicações triviais.
    prompt = f"{config.STYLE_TRIGGER}, {body}"
    return prompt


# ---------------------------------------------------------------------------
# 2) Geração de imagem via Stable Diffusion + LoRA
# ---------------------------------------------------------------------------
def _load_sd_pipeline():
    """Carrega (uma vez) o StableDiffusionPipeline com os pesos LoRA aplicados."""
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if config.device() == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.BASE_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,  # desabilitado: dataset é controlado/curado
    )
    pipe = pipe.to(config.device())

    # Carrega os pesos LoRA: prioridade para o repo do Hub (HF Spaces),
    # senão usa o diretório local models/lora.
    try:
        if config.LORA_REPO_ID:
            pipe.load_lora_weights(
                config.LORA_REPO_ID, weight_name=config.LORA_WEIGHT_NAME
            )
            print(f"[generate_image] LoRA carregado do Hub: {config.LORA_REPO_ID}")
        elif any(config.LORA_LOCAL_PATH.glob("*.safetensors")):
            pipe.load_lora_weights(str(config.LORA_LOCAL_PATH))
            print(f"[generate_image] LoRA carregado de {config.LORA_LOCAL_PATH}")
        else:
            print("[generate_image] AVISO: nenhum LoRA encontrado; usando SD base.")
    except Exception as exc:  # noqa: BLE001
        print(f"[generate_image] Falha ao carregar LoRA ({exc}); usando SD base.")

    _sd_pipe = pipe
    return _sd_pipe


def generate_image(prompt: str):
    """Gera uma imagem PIL a partir do prompt expandido."""
    import torch

    pipe = _load_sd_pipeline()
    cfg = config.INFER_CONFIG
    generator = torch.Generator(device=config.device()).manual_seed(
        config.TRAIN_CONFIG["seed"]
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=cfg["negative_prompt"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        width=cfg["width"],
        height=cfg["height"],
        generator=generator,
        cross_attention_kwargs={"scale": cfg["lora_scale"]},
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(description="Gera pixel art a partir de um tema.")
    parser.add_argument("--theme", required=True, help="Tema curto.")
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "out.png"))
    args = parser.parse_args()

    prompt = expand_prompt(args.theme)
    print(f"Prompt expandido: {prompt}")
    image = generate_image(prompt)
    image.save(args.out)
    print(f"Imagem salva em: {args.out}")


if __name__ == "__main__":
    _main()
