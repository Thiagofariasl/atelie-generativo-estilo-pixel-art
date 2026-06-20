"""
generate_audio.py — Síntese de voz (TTS) para o pipeline multimodal.

Converte um texto (a descrição/narração da cena) em um arquivo de áudio.

DECISÃO ARQUITETURAL
--------------------
Suportamos múltiplos backends, selecionáveis por config.TTS_BACKEND, porque
cada ambiente tem restrições diferentes:
  * "gtts"     -> leve, depende de internet (bom default para o Spaces gratuito).
  * "speecht5" -> offline via transformers (microsoft/speecht5_tts).
  * "coqui"    -> Coqui TTS, alta qualidade porém pesado.

Todos retornam o caminho do arquivo de áudio gerado (compatível com
gr.Audio(type="filepath")).
"""

from __future__ import annotations

import argparse

import config


def _tts_gtts(text: str, out_path: str) -> str:
    """Backend leve baseado em gTTS (Google Translate TTS). Requer internet."""
    from gtts import gTTS

    out_path = out_path.replace(".wav", ".mp3")
    gTTS(text=text, lang=config.TTS_LANG).save(out_path)
    return out_path


def _tts_speecht5(text: str, out_path: str) -> str:
    """Backend offline via transformers (SpeechT5). Idioma: inglês."""
    import torch
    import soundfile as sf
    from datasets import load_dataset
    from transformers import (
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
    )

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    embeddings = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(embeddings[7306]["xvector"]).unsqueeze(0)

    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"], speaker_embedding, vocoder=vocoder
    )
    sf.write(out_path, speech.numpy(), samplerate=16000)
    return out_path


def _tts_coqui(text: str, out_path: str) -> str:
    """Backend Coqui TTS (offline, alta qualidade, pesado)."""
    from TTS.api import TTS

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, file_path=out_path, language=config.TTS_LANG)
    return out_path


def generate_audio(text: str, out_path: str | None = None) -> str:
    """Gera áudio a partir de texto usando o backend configurado.

    Em caso de falha do backend escolhido, registra o erro e re-lança, para que
    o app.py possa decidir cair no MOCK.
    """
    out_path = out_path or str(config.OUTPUTS_DIR / "narration.wav")
    backend = config.TTS_BACKEND.lower()

    dispatch = {
        "gtts": _tts_gtts,
        "speecht5": _tts_speecht5,
        "coqui": _tts_coqui,
    }
    if backend not in dispatch:
        raise ValueError(f"TTS_BACKEND desconhecido: {backend}")

    return dispatch[backend](text, out_path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Converte texto em áudio (TTS).")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "narration.wav"))
    args = parser.parse_args()

    path = generate_audio(args.text, args.out)
    print(f"Áudio salvo em: {path}")


if __name__ == "__main__":
    _main()
