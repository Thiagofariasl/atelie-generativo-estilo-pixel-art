"""
app.py — Interface Gradio do Ateliê Generativo.

FLUXO MULTIMODAL
----------------
    Usuário digita um TEMA curto
        ↓  expand_prompt()         (LLM enriquece o tema)
    Prompt expandido
        ↓  generate_image()        (Stable Diffusion + LoRA)
    Imagem (PIL)
        ↓  generate_audio()        (TTS narra a descrição)
    Imagem + Áudio exibidos na UI

DECISÃO ARQUITETURAL
--------------------
* app.py SÓ orquestra. A lógica pesada vive em scripts/ (testável, reutilizável
  em notebooks e em execução headless).
* MODO MOCK: se torch/modelos não estiverem disponíveis (ex.: máquina sem GPU,
  LoRA ainda não treinado), o app cai automaticamente em funções mock para que
  a interface possa ser desenvolvida e demonstrada em paralelo ao treinamento.
  Assim a Etapa 4 (integração) não fica bloqueada pela Etapa 2 (treino).
"""

from __future__ import annotations

import io
import traceback

import gradio as gr

import config

# ---------------------------------------------------------------------------
# Importação tolerante a falhas: tentamos usar os módulos reais; se algo não
# estiver instalado/treinado, usamos os fallbacks mock definidos abaixo.
# ---------------------------------------------------------------------------
try:
    from scripts.generate_image import expand_prompt, generate_image
    from scripts.generate_audio import generate_audio
    REAL_PIPELINE = True
except Exception as exc:  # noqa: BLE001
    print(f"[app] Pipeline real indisponível, usando MOCK. Motivo: {exc}")
    REAL_PIPELINE = False

    from PIL import Image, ImageDraw

    def expand_prompt(theme: str) -> str:
        """MOCK: simula a expansão de prompt feita pelo LLM."""
        return (
            f"{config.STYLE_TRIGGER}, {theme}, detailed 16-bit pixel art, "
            f"vibrant retro game scene, crisp pixels, limited color palette"
        )

    def generate_image(prompt: str):
        """MOCK: gera uma imagem placeholder 512x512 com o prompt escrito."""
        img = Image.new("RGB", (512, 512), color=(34, 28, 52))
        draw = ImageDraw.Draw(img)
        draw.rectangle([16, 16, 496, 496], outline=(120, 200, 255), width=4)
        draw.text((28, 28), "MOCK PIXEL ART", fill=(255, 220, 120))
        # Quebra o prompt em linhas para caber na imagem.
        words, line, y = prompt.split(), "", 70
        for w in words:
            if len(line) + len(w) > 38:
                draw.text((28, y), line, fill=(200, 200, 200))
                line, y = "", y + 18
            line += w + " "
        draw.text((28, y), line, fill=(200, 200, 200))
        return img

    def generate_audio(text: str, out_path: str | None = None) -> str:
        """MOCK: gera um WAV curto (bip) representando o áudio TTS."""
        import math
        import struct
        import wave

        out_path = out_path or str(config.OUTPUTS_DIR / "mock_audio.wav")
        framerate = 22050
        with wave.open(out_path, "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(framerate)
            for i in range(int(framerate * 1.0)):
                value = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / framerate))
                wav.writeframes(struct.pack("<h", value))
        return out_path


# ---------------------------------------------------------------------------
# Função principal de orquestração (chamada pelo botão da UI)
# ---------------------------------------------------------------------------
def run_pipeline(theme: str):
    """Executa o pipeline completo e retorna (prompt, imagem, áudio, status)."""
    if not theme or not theme.strip():
        return "", None, None, "⚠️ Digite um tema para começar."

    try:
        prompt = expand_prompt(theme.strip())
        image = generate_image(prompt)
        # Narramos a descrição/prompt em áudio.
        narration = f"Cena gerada: {theme.strip()}."
        audio_path = generate_audio(narration)

        mode = "REAL" if REAL_PIPELINE else "MOCK"
        status = f"✅ Concluído (modo {mode}).\n\n**Prompt expandido:** {prompt}"
        return prompt, image, audio_path, status
    except Exception as exc:  # noqa: BLE001
        return "", None, None, f"❌ Erro: {exc}\n\n```\n{traceback.format_exc()}\n```"


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Ateliê Generativo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎨 Ateliê Generativo\n"
            "Gere **Pixel Art 16 bits** a partir de um tema curto. "
            "O sistema expande seu tema com um LLM, gera a imagem com "
            "Stable Diffusion + LoRA e narra a cena em áudio."
        )

        if not REAL_PIPELINE:
            gr.Markdown(
                "> ⚙️ **Modo MOCK ativo** — modelos reais não carregados. "
                "A interface está funcional para desenvolvimento e demonstração."
            )

        with gr.Row():
            with gr.Column(scale=1):
                theme_in = gr.Textbox(
                    label="Tema",
                    placeholder='Ex.: "uma vila medieval ao pôr do sol"',
                    lines=2,
                )
                run_btn = gr.Button("Gerar 🎨", variant="primary")
                gr.Examples(
                    examples=[
                        "uma vila medieval ao pôr do sol",
                        "nave espacial sobrevoando uma cidade neon",
                        "floresta encantada com cogumelos brilhantes",
                        "feira de domingo em uma cidade retrô",
                    ],
                    inputs=theme_in,
                )
            with gr.Column(scale=1):
                prompt_out = gr.Textbox(label="Prompt expandido (LLM)", lines=3)
                image_out = gr.Image(label="Imagem gerada", type="pil")
                audio_out = gr.Audio(label="Narração (TTS)", type="filepath")

        status_out = gr.Markdown()

        run_btn.click(
            fn=run_pipeline,
            inputs=theme_in,
            outputs=[prompt_out, image_out, audio_out, status_out],
        )

        gr.Markdown(
            "---\n*Conteúdo gerado por IA. Estilo: pixel art 16-bit genérico/histórico. "
            "Sem personagens protegidos por copyright nem imitação de artistas vivos. "
            "Proveniência do dataset em `dataset/fontes.csv`.*"
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
