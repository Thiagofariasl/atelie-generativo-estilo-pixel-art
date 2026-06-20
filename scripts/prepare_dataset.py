"""
prepare_dataset.py — Preparação e validação do dataset de Pixel Art 16 bits.

O QUE FAZ
---------
1. Lê dataset/fontes.csv (proveniência: image_url, source_site, license, description).
2. (Opcional) baixa as imagens listadas para dataset/images/.
3. Valida a resolução mínima (>= 512x512, exigência da sistematização).
4. Normaliza para 512x512 (center-crop + resize) preservando a estética pixelada
   (resampling NEAREST para não borrar os pixels).
5. Gera metadata.jsonl no formato esperado pelos scripts de treino de LoRA do
   diffusers: {"file_name": "...", "text": "estilo_pixel_art, <descrição>"}.

DECISÃO ARQUITETURAL
--------------------
* A caption usada no treino é prefixada com config.STYLE_TRIGGER, seguindo o
  padrão do professor ("estilo_cordel, ...") -> aqui "estilo_pixel_art, ...".
* Usamos resampling NEAREST no resize para manter as bordas duras do pixel art;
  BILINEAR/BICUBIC borrariam o estilo que queremos ensinar ao LoRA.
* As imagens NÃO são versionadas no git (ver .gitignore); apenas fontes.csv e
  metadata.jsonl, garantindo proveniência sem inflar o repositório.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config


def load_sources() -> pd.DataFrame:
    """Carrega e valida o fontes.csv."""
    df = pd.read_csv(config.FONTES_CSV)
    required = {"image_url", "source_site", "license", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fontes.csv sem colunas obrigatórias: {missing}")
    # Regra do trabalho: dataset entre 20 e 40 imagens.
    if not (20 <= len(df) <= 40):
        print(f"[prepare] AVISO: {len(df)} linhas (esperado entre 20 e 40).")
    return df


def download_images(df: pd.DataFrame) -> None:
    """Baixa as imagens de image_url para dataset/images/ (quando for URL direta)."""
    import requests

    for i, row in df.iterrows():
        url = str(row["image_url"])
        if not url.lower().startswith("http"):
            print(f"[prepare] linha {i}: image_url não é URL direta, pulando download.")
            continue
        ext = Path(url.split("?")[0]).suffix or ".png"
        dest = config.DATASET_IMAGES_DIR / f"img_{i:03d}{ext}"
        if dest.exists():
            continue
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"[prepare] baixada {dest.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[prepare] FALHA ao baixar {url}: {exc}")


def validate_and_normalize() -> list[dict]:
    """Valida resolução e normaliza imagens para 512x512 preservando pixels."""
    from PIL import Image

    res = config.TRAIN_CONFIG["resolution"]
    records: list[dict] = []

    for path in sorted(config.DATASET_IMAGES_DIR.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < res or h < res:
                print(f"[prepare] {path.name}: {w}x{h} < {res}px — REVISAR/descartar.")
            # Center-crop quadrado + resize NEAREST (mantém estética pixelada).
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((res, res), Image.NEAREST)
            im.save(path)
        records.append({"file_name": path.name})
    return records


def build_metadata(df: pd.DataFrame, records: list[dict]) -> None:
    """Escreve metadata.jsonl com captions prefixadas pelo trigger token."""
    out = config.DATASET_IMAGES_DIR / "metadata.jsonl"
    descriptions = df["description"].fillna("").tolist()

    with out.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            desc = descriptions[i] if i < len(descriptions) else "pixel art scene"
            caption = f"{config.STYLE_TRIGGER}, {desc}"
            f.write(json.dumps({"file_name": rec["file_name"], "text": caption},
                               ensure_ascii=False) + "\n")
    print(f"[prepare] metadata.jsonl escrito em {out} ({len(records)} itens).")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset de pixel art.")
    parser.add_argument("--download", action="store_true",
                        help="Baixa as imagens listadas em fontes.csv.")
    parser.add_argument("--skip-normalize", action="store_true")
    args = parser.parse_args()

    df = load_sources()
    print(f"[prepare] {len(df)} fontes carregadas de fontes.csv.")

    if args.download:
        download_images(df)

    records = [] if args.skip_normalize else validate_and_normalize()
    if records:
        build_metadata(df, records)
    print("[prepare] Concluído.")


if __name__ == "__main__":
    _main()
