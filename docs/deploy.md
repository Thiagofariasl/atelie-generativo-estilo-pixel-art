# 🚀 Deploy — Hugging Face Spaces (Etapa 4)

## Visão geral
A interface Gradio (`app.py`) é publicada em um **Hugging Face Space** (SDK
Gradio). O Space lê `app.py` na raiz e instala `requirements.txt`.

## Passo a passo
1. Crie um Space: **New Space → SDK: Gradio**.
2. Conecte o GitHub **ou** faça push direto para o repositório do Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<usuario>/atelie-generativo
   git push space main
   ```
3. Cabeçalho YAML sugerido (no topo do `README.md` do Space):
   ```yaml
   ---
   title: Ateliê Generativo
   emoji: 🎨
   colorFrom: indigo
   colorTo: purple
   sdk: gradio
   sdk_version: "4.44.0"
   app_file: app.py
   pinned: false
   ---
   ```
4. **Hardware**: para geração real de imagem, selecione um **GPU** em
   *Settings → Hardware*. Sem GPU, o app roda em **modo MOCK** (útil para
   validar a UI sem custo).

## 🔐 Segredos (CRÍTICO — critério de segurança)
> A exposição de chaves/tokens no código de um repositório **público** zera o
> critério de segurança da sistematização.

- **Nunca** coloque tokens em `config.py` ou em qualquer arquivo versionado.
- Configure em **Settings → Variables and secrets** do Space:
  - `HF_TOKEN` — token de acesso ao Hub (se o LoRA estiver em repo privado).
  - `LORA_REPO_ID` — id do repo com os pesos LoRA (ex.: `equipe/atelie-pixelart-lora`).
  - Demais variáveis opcionais (`TTS_BACKEND`, `LLM_MODEL_ID`, ...).
- `config.py` lê tudo via `os.getenv(...)`; localmente use um arquivo `.env`
  (ignorado pelo git) ou variáveis de ambiente.

## Pesos do LoRA no Space
Duas opções:
1. **Recomendado**: publicar os pesos no **HF Hub** e definir `LORA_REPO_ID`.
   O app baixa em runtime — mantém o repositório de código leve.
2. Incluir `*.safetensors` em `models/lora/` (atenção ao limite de tamanho e ao
   `.gitignore`, que ignora esses binários por padrão).

## Verificação pós-deploy
- O Space builda sem erros (ver aba *Logs*).
- O fluxo tema → prompt → imagem → áudio funciona na UI.
- Conferir que nenhum segredo aparece nos logs ou no código público.
