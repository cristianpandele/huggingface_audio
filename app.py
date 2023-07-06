import gradio as gr
import numpy as np
import torch
from datasets import load_dataset

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load speech translation checkpoint
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# load text-to-speech checkpoint and speaker embeddings
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def translate(audio):
    outputs = asr_pipe(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    return outputs["text"]


def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder)
    return speech.cpu()


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * 32767).astype(np.int16)
    return 16000, synthesised_speech


demo = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    examples=[["./example.wav"]],
)
demo.launch()
