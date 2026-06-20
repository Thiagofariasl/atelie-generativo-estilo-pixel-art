"""
train_lora.py — Fine-tuning de Stable Diffusion com LoRA.

OBJETIVO (Etapa 2 da sistematização)
-------------------------------------
Treinar um adaptador LoRA sobre o modelo base de difusão usando o dataset de
pixel art. A sistematização exige TESTAR PELO MENOS DUAS CONFIGURAÇÕES
distintas e justificar os hiperparâmetros — por isso recebemos um --preset que
seleciona um dos config.TRAIN_PRESETS (ex.: rank 8 x rank 16).

DECISÃO ARQUITETURAL
--------------------
* Treino real é tipicamente executado no GOOGLE COLAB (GPU T4). A forma mais
  robusta e mantida é o script oficial do diffusers
  `train_text_to_image_lora.py`. Este módulo:
    (a) monta e imprime o comando correspondente (modo padrão), e
    (b) expõe os hiperparâmetros vindos de config.py para manter um único
        ponto de verdade.
  Assim evitamos reimplementar o loop de treino (propenso a bugs) e mantemos
  reprodutibilidade. O notebook em notebooks/ chama este script.

USO
---
    python -m scripts.train_lora --preset config_b_rank16 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import config


def build_command(preset_name: str, train_data_dir: str, output_dir: str) -> list[str]:
    """Monta o comando do script oficial de treino LoRA do diffusers."""
    if preset_name not in config.TRAIN_PRESETS:
        raise ValueError(
            f"Preset '{preset_name}' inexistente. "
            f"Opções: {list(config.TRAIN_PRESETS)}"
        )
    p = config.TRAIN_PRESETS[preset_name]

    cmd = [
        "accelerate", "launch", "train_text_to_image_lora.py",
        f"--pretrained_model_name_or_path={config.BASE_MODEL_ID}",
        f"--train_data_dir={train_data_dir}",
        "--caption_column=text",
        f"--resolution={p['resolution']}",
        "--random_flip",
        f"--train_batch_size={p['train_batch_size']}",
        f"--gradient_accumulation_steps={p['gradient_accumulation_steps']}",
        f"--max_train_steps={p['max_train_steps']}",
        f"--learning_rate={p['learning_rate']}",
        f"--lr_scheduler={p['lr_scheduler']}",
        f"--lr_warmup_steps={p['lr_warmup_steps']}",
        f"--rank={p['lora_rank']}",
        f"--seed={p['seed']}",
        f"--mixed_precision={p['mixed_precision']}",
        f"--output_dir={output_dir}",
        "--checkpointing_steps=500",
        "--validation_prompt="
        f"{config.STYLE_TRIGGER}, a cozy village at sunset",
    ]
    return cmd


def _main() -> None:
    parser = argparse.ArgumentParser(description="Treina LoRA sobre Stable Diffusion.")
    parser.add_argument("--preset", default="config_b_rank16",
                        choices=list(config.TRAIN_PRESETS))
    parser.add_argument("--train-data-dir", default=str(config.DATASET_IMAGES_DIR))
    parser.add_argument("--output-dir", default=str(config.LORA_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas imprime o comando, sem executar.")
    args = parser.parse_args()

    cmd = build_command(args.preset, args.train_data_dir, args.output_dir)
    print("Preset:", args.preset)
    print("Hiperparâmetros:", config.TRAIN_PRESETS[args.preset])
    print("\nComando de treino:\n  " + " ".join(cmd) + "\n")

    if args.dry_run:
        print("[train_lora] dry-run: comando NÃO executado.")
        return

    # Requer que train_text_to_image_lora.py esteja no diretório de trabalho
    # (baixe-o do repositório diffusers/examples/text_to_image — ver docs/treinamento.md).
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[train_lora] ERRO: 'accelerate' ou o script de treino não encontrado.",
              file=sys.stderr)
        print("Veja docs/treinamento.md para o setup no Colab.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
