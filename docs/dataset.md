# 📚 Dataset — Pixel Art 16 bits

## Objetivo
Construir um conjunto de **20 a 40 imagens** de pixel art 16 bits, com resolução
mínima de **512×512**, proveniência registrada e legendas (captions) revisadas.

## Fontes utilizadas
A equipe coletou imagens priorizando licenças compatíveis com treino de modelos:

- **itch.io** — assets de jogos disponibilizados gratuitamente por artistas
  (fonte principal). Selecionamos **poucas imagens por artista** para garantir
  diversidade e evitar enviesar o modelo ao estilo de um único criador.
- **Pixabay** — Pixabay Content License.
- **Pexels** — Pexels License.
- **Kaggle** (pixel-art dataset) — usado como referência; atenção à resolução.

> Critério declarado ao professor no fórum: usamos imagens cujas licenças
> permitem o uso pretendido e sem restrição a treinamento de IA, coletando
> poucas imagens de cada artista e buscando diversidade.

## Proveniência — `fontes.csv`
Cada linha registra a origem de uma imagem. Colunas:

| Coluna        | Descrição                                                |
|---------------|----------------------------------------------------------|
| `image_url`   | URL direta da imagem (ou da página de origem, quando o asset está zipado) |
| `source_site` | Site de origem (itch.io, pixabay, pexels, ...)           |
| `license`     | Licença declarada da imagem                              |
| `description` | Legenda no formato `estilo_pixel_art, <descrição da cena>` |

## Padrão de legenda (caption)
Seguindo o exemplo do professor (`estilo_cordel, uma rendeira...`), toda caption
começa com o **trigger token** do estilo:

```
estilo_pixel_art, cena com rio, árvores, casa e personagem segurando uma espada
```

Esse token (`config.STYLE_TRIGGER`) é o que o LoRA aprende a associar ao estilo
e o que injetamos no prompt durante a inferência.

## Pré-processamento
Execute:

```bash
python -m scripts.prepare_dataset --download
```

O script:
1. Valida as colunas obrigatórias de `fontes.csv`.
2. Baixa as imagens (quando a URL é direta) para `dataset/images/`.
3. Valida a resolução mínima (≥ 512×512) e avisa as que precisam de revisão.
4. Normaliza para 512×512 com **center-crop + resize NEAREST** (mantém as bordas
   duras do pixel art — bilinear/bicubic borrariam o estilo).
5. Gera `metadata.jsonl` com as captions prefixadas pelo trigger token.

## Versionamento
As **imagens não são versionadas** no Git (ver `.gitignore`) para não inflar o
repositório. Versionamos a **proveniência** (`fontes.csv`) e as **legendas**
(`metadata.jsonl`). Os binários ficam no Google Drive / Hugging Face Datasets.
