import sys, os
# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from pretrain.whispa_data import preprocess_audio

processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Example usage
audio_path = "/cronus_data/rrao/samples/P209_segment.wav"

# Preprocess the audio
waveform = preprocess_audio(audio_path)

# Process the audio to get input features
input_features = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

generated_ids = model.generate(input_features=input_features, attention_mask=torch.ones_like(input_features))

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)