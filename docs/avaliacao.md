# 📊 Avaliação (Etapa 3)

A avaliação combina uma métrica **quantitativa** (CLIPScore), **avaliação
humana** qualitativa e uma **comparação entre modelos/configurações**.

## 1. CLIPScore
Mede o alinhamento texto↔imagem usando o CLIP. Quanto maior, mais a imagem
corresponde ao prompt. Calculado localmente em `scripts/evaluate_model.py`
(modelo `openai/clip-vit-base-patch32`).

```bash
python -m scripts.evaluate_model --clipscore
```

Comparamos **SD base x SD+LoRA** com os **mesmos prompts** e a **mesma seed**,
para isolar o efeito do fine-tuning. Resultado salvo em `outputs/clipscore.txt`.

> Limitação: o CLIPScore mede aderência semântica ao prompt, **não** fidelidade
> ao estilo pixel art. Por isso a avaliação humana é complementar e essencial.

## 2. Avaliação humana
Gera um template CSV para avaliadores pontuarem (escala 1–5):

```bash
python -m scripts.evaluate_model --human-template
# -> outputs/avaliacao_humana.csv
```

Critérios:
| Critério               | O que mede                                         |
|------------------------|----------------------------------------------------|
| `fidelidade_estilo`    | Quão "16-bit pixel art" a imagem parece            |
| `qualidade`            | Ausência de artefatos, coerência visual            |
| `aderencia_prompt`     | A imagem corresponde ao tema pedido?               |

Recomendação: ≥ 3 avaliadores, média e desvio por critério/modelo no relatório.

## 3. Comparação entre modelos/configs
Gerar as **mesmas cenas** (`EVAL_PROMPTS`) para:
- SD base (sem LoRA)
- SD + LoRA `config_a_rank8`
- SD + LoRA `config_b_rank16`

e tabular CLIPScore + médias humanas. Tabela sugerida para o relatório:

| Modelo               | CLIPScore | Fidelidade | Qualidade | Aderência |
|----------------------|-----------|------------|-----------|-----------|
| SD base              |           |            |           |           |
| SD+LoRA rank8        |           |            |           |           |
| SD+LoRA rank16       |           |            |           |           |

## 4. Discussão
Relacionar os números às escolhas de hiperparâmetros (ver
[treinamento.md](treinamento.md)) e justificar o modelo escolhido para o deploy.
