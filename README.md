# 🎨 Ateliê Generativo

Pipeline multimodal de geração de **Pixel Art 16 bits** que combina expansão de prompt
via LLM, geração de imagem com **Stable Diffusion + LoRA**, síntese de voz (TTS) e uma
interface **Gradio** pronta para deploy no **Hugging Face Spaces**.

> Projeto acadêmico. O nome do diretório usa `atelie-generativo` (sem acento) para garantir
> compatibilidade com Git, URLs e ferramentas cross-platform. O nome de exibição do projeto
> continua sendo **Ateliê Generativo**.

---

## 📌 Descrição do projeto

O **Ateliê Generativo** é um sistema de IA generativa multimodal. A partir de um tema
curto digitado pelo usuário (ex.: *"um castelo ao pôr do sol"*), o sistema:

1. **Expande** o tema em um prompt rico e estilizado usando um LLM.
2. **Gera** uma imagem em estilo Pixel Art 16 bits usando Stable Diffusion com um adaptador
   LoRA treinado pela equipe.
3. **Converte** a descrição textual em **áudio** (TTS), narrando a cena gerada.
4. **Exibe** imagem + áudio em uma interface Gradio.

O modelo de imagem é especializado via **fine-tuning LoRA** sobre um dataset autoral/curado
de Pixel Art, respeitando restrições éticas e de direitos autorais.

---

## 🎯 Objetivos

- Demonstrar fine-tuning eficiente de Stable Diffusion com **LoRA** (baixo custo de GPU).
- Construir um **pipeline multimodal** texto → texto → imagem → áudio.
- Entregar uma aplicação **reproduzível** e publicada no Hugging Face Spaces.
- Aplicar boas práticas de **curadoria de dataset** e **considerações éticas**.
- Avaliar o modelo de forma quantitativa (**CLIPScore**) e qualitativa (**avaliação humana**).

---

## 🏗️ Arquitetura

```
┌──────────────┐
│ Usuário      │  digita um TEMA curto
│ (Gradio UI)  │
└──────┬───────┘
       │ texto
       ▼
┌──────────────────────┐
│ Expansão de Prompt   │  LLM transforma o tema em prompt detalhado
│ (scripts/...LLM)     │  + injeta tokens de estilo "pixel art 16-bit"
└──────┬───────────────┘
       │ prompt expandido
       ▼
┌──────────────────────┐      ┌─────────────────────┐
│ Geração de Imagem    │◄─────│ Stable Diffusion     │
│ generate_image.py    │      │ + LoRA (PEFT)        │
└──────┬───────────────┘      └─────────────────────┘
       │ imagem (PIL)              ▲
       │                          │ pesos LoRA (models/)
       ▼
┌──────────────────────┐
│ Geração de Áudio     │  TTS narra a descrição da cena
│ generate_audio.py    │
└──────┬───────────────┘
       │ imagem + áudio
       ▼
┌──────────────┐
│ Gradio UI    │  exibe resultado multimodal
└──────────────┘
```

**Decisões arquiteturais (resumo):**

- **Separação de camadas**: a UI (`app.py`) só orquestra; toda lógica de domínio vive em
  `scripts/` para permitir testes, reuso em notebooks e execução headless.
- **`config.py` centralizado**: caminhos, hiperparâmetros e parâmetros de inferência em um
  único lugar → reprodutibilidade e fácil ajuste para Colab vs. Spaces vs. local.
- **Fallback mock**: o app funciona **mesmo sem GPU/modelo treinado** (funções mock), para
  que a interface possa ser desenvolvida e testada em paralelo ao treinamento.
- **LoRA em vez de full fine-tuning**: viável em GPU gratuita do Colab, pesos pequenos
  (~MBs) versionáveis e fáceis de publicar no Hub.

---

## 🧰 Tecnologias utilizadas

| Camada              | Tecnologia                                   |
|---------------------|----------------------------------------------|
| Linguagem           | Python 3.10+                                  |
| Treinamento         | Google Colab (GPU T4)                         |
| Modelo base         | Stable Diffusion (`diffusers`)                |
| Fine-tuning         | LoRA via `peft` / `diffusers` training scripts|
| Expansão de prompt  | LLM via `transformers`                        |
| Áudio (TTS)         | `gTTS` (default) / Coqui `TTS` / SpeechT5     |
| Interface           | `gradio`                                      |
| Aceleração          | `torch`, `accelerate`                         |
| Dados               | `pandas`, `Pillow`                            |
| Hospedagem          | Hugging Face Spaces + GitHub                  |

---

## ▶️ Como executar localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/<sua-equipe>/atelie-generativo.git
cd atelie-generativo

# 2. Criar ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. (Opcional) colocar pesos LoRA em models/lora/
#    Sem eles o app roda em modo MOCK automaticamente.

# 5. Rodar a aplicação
python app.py
```

A interface abrirá em `http://127.0.0.1:7860`.

> **Modo MOCK**: se `torch`/modelos não estiverem disponíveis, o app gera uma imagem
> placeholder e um áudio sintético simples, permitindo validar o fluxo end-to-end.

---

## ☁️ Como executar no Hugging Face Spaces

1. Crie um novo **Space** (SDK: **Gradio**).
2. Faça push deste repositório para o Space (ou conecte o GitHub).
3. Garanta que `app.py` esteja na raiz e `requirements.txt` liste as dependências.
4. Para usar o LoRA, publique os pesos no Hub e ajuste `LORA_REPO_ID` em `config.py`,
   **ou** inclua os pesos em `models/lora/` (atenção ao limite de tamanho do Space).
5. Para geração real de imagem, selecione um hardware com **GPU** nas configurações do Space.

```yaml
# Cabeçalho sugerido para o README do Space (metadata YAML):
title: Ateliê Generativo
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
```

---

## 📁 Estrutura de pastas

```
atelie-generativo/
│
├── app/                  # Componentes de UI reutilizáveis (futuro)
├── dataset/              # Dados de treino + fontes.csv (proveniência)
│   ├── images/           # Imagens 512x512+ (NÃO versionar imagens grandes)
│   └── fontes.csv        # Origem, licença e descrição de cada imagem
├── models/               # Pesos do modelo base e LoRA (ignorados no git)
│   └── lora/
├── notebooks/            # Notebooks Colab (treino, EDA, avaliação)
├── scripts/              # Lógica de domínio (CLI + importável)
│   ├── prepare_dataset.py
│   ├── train_lora.py
│   ├── evaluate_model.py
│   ├── generate_image.py
│   └── generate_audio.py
├── docs/                 # Documentação detalhada por etapa
│   ├── dataset.md
│   ├── treinamento.md
│   ├── avaliacao.md
│   ├── deploy.md
│   └── etica.md
├── outputs/              # Artefatos gerados (imagens/áudios) - ignorado
├── tests/                # Testes do pipeline
│   └── test_pipeline.py
├── config.py             # Configuração central (caminhos + hiperparâmetros)
├── requirements.txt
├── README.md
├── .gitignore
└── app.py                # Interface Gradio (orquestra o pipeline)
```

---

## 🗓️ Cronograma (etapas oficiais da sistematização)

| Etapa | Descrição (roteiro do professor)                                                        | Entregável                          | Status |
|-------|------------------------------------------------------------------------------------------|-------------------------------------|--------|
| 0     | Organização da equipe, repositório público no GitHub e proposta do estilo visual         | Repo + estilo definido (pixel art)  | ✅     |
| 1     | Dataset de 20–40 imagens (≥512×512) + `fontes.csv` (proveniência/licenças) e legendas    | `dataset/` + `fontes.csv`           | ✅     |
| 2     | Fine-tuning com LoRA testando **≥ 2 configurações** e justificando os hiperparâmetros    | Pesos LoRA + comparação             | ⬜     |
| 3     | Avaliação: comparação de resultados, **CLIPScore** e **avaliação humana**                | `docs/avaliacao.md` + métricas      | ⬜     |
| 4     | Integração texto + imagem + áudio via Gradio publicado no **Hugging Face Spaces**        | Space público funcional             | ⬜     |
| 5     | Relatório final em PDF (com **reflexão ética**) + **Demo Day** (12 min + 5 de arguição)  | PDF + apresentação                  | ⬜     |

---

## 👥 Equipe

| Nome            | Função (sugerida)              | Contato                        |
|-----------------|--------------------------------|--------------------------------|
| Matheus Loiola  | Curadoria de dataset           | matheusloiola@sempreceub.com   |
| Thiago Farias   | Repositório / Interface / Deploy | thiago.famaral@sempreceub.com|
| Rodério Kunz| Treinamento LoRA / Avaliação   | roderiok@sempreceub.com                       |
| Sabrina Santos Sousa | Treinamento LoRA / Avaliação | sabrina.santos@sempreceub.com | 
| Rafael Moreira Ferreira      | Treinamento LoRA / Avaliação                    | rfmoreira2@gmail.com                        |

**Professor:** Romes Heriberto.

> Complete com os integrantes restantes e ajuste as funções conforme a divisão real.

---

## ⚖️ Considerações éticas

Este projeto adota as seguintes diretrizes (detalhadas em [docs/etica.md](docs/etica.md)):

- **Sem personagens protegidos por copyright** (ex.: mascotes de jogos/franquias).
- **Sem imitação de artistas vivos identificáveis** — não usamos nomes próprios de artistas
  contemporâneos em prompts nem treinamos para reproduzir seu estilo pessoal.
- **Apenas estilos genéricos e históricos** (ex.: "estética 16-bit dos anos 90", "pixel art
  isométrica") — domínio público ou estilo, não obra específica.
- **Proveniência registrada** em [`dataset/fontes.csv`](dataset/fontes.csv): URL, site de
  origem, licença e descrição de cada imagem.
- **Licenças compatíveis**: priorizar CC0 / domínio público / criação própria da equipe.
- **Transparência**: o conteúdo é declaradamente gerado por IA.

---

## 📜 Licença

Defina a licença do projeto (sugestão: MIT para o código; o dataset segue as licenças
individuais listadas em `fontes.csv`).
