# Criação do app.py (O Pipeline Multimodal)
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import os

# Configuração dinâmica: Usa GPU se disponível no Spaces (ex: ZeroGPU), senão cai para CPU (Inference via software)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Inicializando modelo no dispositivo: {device}")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype).to(device)

# Carrega o LoRA usando a Inference API
pipe.load_lora_weights("roderiok/pixelart-16bit-lora")

def gerar_experiencia_multimodal(prompt_usuario):
    # Modalidade 1: Orquestração do Texto
    prompt_orquestrado = f"estilo_pixel_art, {prompt_usuario}, cena em pixel art de um lago, coqueiros e um barco."

    # Modalidade 2: Visão Computacional (Geração da Imagem)
    # 30 steps garantem rapidez no ambiente sem GPU dedicada (como o Spaces gratuito)
    imagem = pipe(prompt_orquestrado, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.7}).images[0]

    # Modalidade 3: Síntese de Áudio
    texto_narracao = f"Gerando arte em pixel art para a sua solicitação: {prompt_usuario}."
    tts = gTTS(texto_narracao, lang='pt')
    audio_path = "narracao.mp3"
    tts.save(audio_path)

    return prompt_orquestrado, imagem, audio_path

# Interface UI/UX
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 👾 Ateliê Generativo: Multimodal Pixel Art (16-bits)")
    gr.Markdown("Transforme suas ideias em arte retrô narrada. Pipeline Integrado: Texto ➔ Imagem ➔ Áudio.")

    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Textbox(label="O que você deseja criar?", placeholder="Ex: Cena em pixel art de um lago, coqueiros e um barco.")
            btn = gr.Button("Sintetizar", variant="primary")
            saida_prompt = gr.Textbox(label="Texto Orquestrado", interactive=False)
            saida_audio = gr.Audio(label="Voz do Sistema")

        with gr.Column(scale=2):
            saida_imagem = gr.Image(label="Arte Gerada")

    btn.click(fn=gerar_experiencia_multimodal, inputs=entrada, outputs=[saida_prompt, saida_imagem, saida_audio])

demo.launch()