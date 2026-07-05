"""
Aplicação para o Gradio.

Criação do aplicativo multimodal que integra geração de imagens em pixel art com narração em áudio, 
utilizando o modelo Stable Diffusion e LoRA para estilo específico. A interface é construída com Gradio,
permitindo ao usuário inserir prompts de texto, ajustar parâmetros de inferência e receber como saída a 
imagem gerada e a narração correspondente.
"""
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from gtts import gTTS
from transformers import pipeline

# Usa GPU se disponível no Spaces (ex: ZeroGPU), senão cai para CPU (Inference via software)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Obtém o modelo base dos pesos
print(f"Inicializando modelo no dispositivo {device}")
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch_dtype).to(device)

# Carrega o LoRA usando a Inference API
pipe.load_lora_weights("loioladev/lora-estilo-pixel-art")

# Modelo leve de instrução usado para expandir o prompt do usuário com mais detalhes visuais
expansor_prompt = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch_dtype,
    device=0 if device == "cuda" else -1,
)

NEGATIVE_PROMPT_PADRAO = "blurry, low quality, jpeg artifacts, photorealistic, 3d render, watermark, text, signature, deformed"

def expandir_prompt(prompt_usuario):
    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente que melhora a descrição de prompts de descrições visuais. Seu objetivo é tornar eles mais ricos e detalhadas para geração de imagens em pixel art. Responda apenas com o prompt melhorado, em uma única frase, sem explicações adicionais. Adicione detalhes de cores, iluminação, perspectiva e elementos visuais que possam enriquecer a cena, mantendo o estilo pixel art e o que foi solicitado pelo usuário."
            ),
        },
        {"role": "user", "content": prompt_usuario},
    ]
    resposta = expansor_prompt(mensagens, max_new_tokens=80, do_sample=True, temperature=0.7)
    return resposta[0]["generated_text"][-1]["content"].strip()

def gerar_experiencia_multimodal(prompt_usuario, negative_prompt, num_inference_steps, guidance_scale, lora_scale):
    # Modalidade 1: Orquestração do Texto (expande o prompt do usuário via LLM e prefixa o estilo do LoRA)
    prompt_expandido = expandir_prompt(prompt_usuario)
    prompt_orquestrado = f"estilo_pixel_art, {prompt_expandido}"

    # Visão Computacional
    imagem = pipe(
        prompt_orquestrado,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]

    # Síntese de Áudio
    texto_narracao = f"Gerando arte em pixel art para a sua solicitação: {prompt_usuario}."
    tts = gTTS(texto_narracao, lang='pt')
    audio_path = "narracao.mp3"
    tts.save(audio_path)

    return prompt_orquestrado, imagem, audio_path

# Interface UI/UX
with gr.Blocks(theme=gr.Theme.from_hub("hmb/midnight")) as demo:
    gr.Markdown("# 👾 Ateliê Generativo: Multimodal Pixel Art (16-bits)")
    gr.Markdown("Transforme suas ideias em arte retrô narrada. Pipeline Integrado: Texto ➔ Imagem ➔ Áudio.")

    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Textbox(label="O que você deseja criar?", placeholder="Ex: Cena em pixel art de um lago, coqueiros e um barco.")
            negative_prompt = gr.Textbox(label="Prompt Negativo", value=NEGATIVE_PROMPT_PADRAO)

            with gr.Accordion("Parâmetros de Inferência", open=False):
                num_inference_steps = gr.Slider(label="Passos de Inferência", minimum=1, maximum=100, value=30, step=1)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                lora_scale = gr.Slider(label="Intensidade do LoRA (Estilo Pixel Art)", minimum=0.0, maximum=1.0, value=0.7, step=0.05)

            btn = gr.Button("Sintetizar", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                saida_prompt = gr.Textbox(label="Texto Orquestrado", interactive=False)
                saida_audio = gr.Audio(label="Voz do Sistema")

            saida_imagem = gr.Image(label="Arte Gerada")

    btn.click(
        fn=gerar_experiencia_multimodal,
        inputs=[entrada, negative_prompt, num_inference_steps, guidance_scale, lora_scale],
        outputs=[saida_prompt, saida_imagem, saida_audio],
    )

demo.launch()
