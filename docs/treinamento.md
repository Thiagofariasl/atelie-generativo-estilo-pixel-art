# 🏋️ Treinamento — Fine-tuning com LoRA (Etapa 2)

## Por que LoRA?
**LoRA (Low-Rank Adaptation)** injeta matrizes de baixo rank nas camadas de
atenção do modelo, treinando **poucos parâmetros** em vez do modelo inteiro.
Vantagens para este projeto:
- Cabe na **GPU T4 gratuita** do Colab.
- Pesos finais pequenos (~MBs), fáceis de publicar no Hub e versionar.
- Preserva o conhecimento do modelo base, especializando só o estilo.

## Ambiente
Treino executado no **Google Colab (GPU T4)**. Veja o notebook
[`notebooks/01_train_lora_colab.ipynb`](../notebooks/01_train_lora_colab.ipynb).

Usamos o script oficial mantido pelo `diffusers`
(`train_text_to_image_lora.py`) em vez de reimplementar o loop de treino —
mais robusto e reprodutível. O `scripts/train_lora.py` monta o comando a partir
dos hiperparâmetros centralizados em `config.py`.

## Duas configurações (exigência da Etapa 2)
A sistematização exige **testar ao menos duas configurações e justificar**.
Definimos em `config.TRAIN_PRESETS`:

| Preset             | rank | alpha | steps | lr     | Hipótese                                  |
|--------------------|------|-------|-------|--------|-------------------------------------------|
| `config_a_rank8`   | 8    | 8     | 1000  | 1e-4   | Menor capacidade, menos overfitting, mais rápido |
| `config_b_rank16`  | 16   | 16    | 1500  | 1e-4   | Maior capacidade p/ capturar detalhes do estilo |

**Justificativa dos hiperparâmetros:**
- **rank**: controla a capacidade do adaptador. Rank baixo (8) generaliza com
  dataset pequeno; rank maior (16) captura mais nuances mas arrisca overfitting.
- **learning_rate 1e-4**: valor estável e usual para LoRA em SD.
- **lr_scheduler cosine + warmup**: evita instabilidade no início e refina no fim.
- **resolution 512**: casa com o modelo base SD 1.5 e a exigência mínima do dataset.
- **seed 42**: reprodutibilidade e comparação justa entre as configs.

## Como rodar
```bash
# Imprime o comando (revisar antes de executar):
python -m scripts.train_lora --preset config_b_rank16 --dry-run

# Executa de fato (no Colab, com o script de treino baixado):
python -m scripts.train_lora --preset config_b_rank16 --output-dir models/lora_rank16
```

## Saída
Pesos `pytorch_lora_weights.safetensors` em `models/lora_<preset>/`. Publique no
Hugging Face Hub e aponte `LORA_REPO_ID` (em `config.py` ou Secrets do Space).

## Próximos passos
Comparar os dois presets na [avaliação](avaliacao.md) (CLIPScore + humana) e
escolher o melhor para o deploy.
