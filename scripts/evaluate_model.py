"""
evaluate_model.py — Avaliação do modelo (Etapa 3 da sistematização).

MÉTRICAS
--------
1. CLIPScore: mede o alinhamento texto-imagem usando CLIP. Quanto maior, mais a
   imagem corresponde ao prompt. Comparamos SD base x SD+LoRA para evidenciar o
   efeito do fine-tuning.
2. Comparação entre modelos/configs: gera as MESMAS cenas para cada config
   (ex.: rank 8 x rank 16) e tabula os CLIPScores.
3. Avaliação humana: exporta um CSV/painel para que avaliadores pontuem
   fidelidade ao estilo, qualidade e aderência ao prompt (1–5).

DECISÃO ARQUITETURAL
--------------------
* CLIPScore é calculado localmente via transformers (CLIPModel), evitando
  dependências extras e mantendo o cálculo reprodutível.
* Os prompts de avaliação ficam em uma lista fixa (seed fixa) para que a
  comparação entre configs seja justa (mesmas cenas, mesma seed).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import config

# Conjunto fixo de prompts de avaliação (cenas representativas do domínio).
EVAL_PROMPTS = [
    "a cozy medieval village at sunset",
    "a spaceship flying over a neon city",
    "an enchanted forest with glowing mushrooms",
    "a sunday market in a retro town",
    "a snowy mountain cabin at night",
]


def clip_score(images: list, prompts: list[str]) -> float:
    """Calcula o CLIPScore médio entre imagens e prompts."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(config.device())
    processor = CLIPProcessor.from_pretrained(model_id)

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(
            text=[prompt], images=img, return_tensors="pt", padding=True
        ).to(config.device())
        with torch.no_grad():
            out = model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        # CLIPScore = 100 * max(cos_sim, 0), seguindo Hessel et al. (2021).
        scores.append(float(100 * max((img_emb * txt_emb).sum().item(), 0)))
    return sum(scores) / len(scores) if scores else 0.0


def generate_eval_images(use_lora: bool) -> list:
    """Gera as imagens de avaliação para os EVAL_PROMPTS."""
    from scripts.generate_image import generate_image, _load_sd_pipeline  # noqa

    # Para comparar base x LoRA, alternamos LORA_REPO_ID via config em runtime.
    images = []
    for prompt in EVAL_PROMPTS:
        full_prompt = f"{config.STYLE_TRIGGER}, {prompt}"
        images.append(generate_image(full_prompt))
    return images


def export_human_eval_template(out_csv: Path) -> None:
    """Cria um CSV em branco para a avaliação humana (1–5 por critério)."""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "avaliador", "prompt", "modelo",
            "fidelidade_estilo_1a5", "qualidade_1a5",
            "aderencia_prompt_1a5", "comentarios",
        ])
        for prompt in EVAL_PROMPTS:
            for modelo in ("sd_base", "sd_lora"):
                writer.writerow(["", prompt, modelo, "", "", "", ""])
    print(f"[evaluate] Template de avaliação humana em {out_csv}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o modelo (CLIPScore + humana).")
    parser.add_argument("--clipscore", action="store_true",
                        help="Calcula CLIPScore das imagens geradas com LoRA.")
    parser.add_argument("--human-template", action="store_true",
                        help="Gera o CSV de avaliação humana.")
    args = parser.parse_args()

    if args.human_template:
        export_human_eval_template(config.OUTPUTS_DIR / "avaliacao_humana.csv")

    if args.clipscore:
        images = generate_eval_images(use_lora=True)
        score = clip_score(images, EVAL_PROMPTS)
        print(f"[evaluate] CLIPScore médio (SD+LoRA): {score:.2f}")
        # Salva resultado para o relatório.
        (config.OUTPUTS_DIR / "clipscore.txt").write_text(
            f"CLIPScore medio (SD+LoRA): {score:.2f}\n", encoding="utf-8"
        )

    if not (args.human_template or args.clipscore):
        parser.print_help()


if __name__ == "__main__":
    _main()
