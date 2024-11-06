import os
import pandas as pd
import torch, torchaudio


HITOP_AUDIO_DIR = '/cronus_data/hitop/iHiTOP_transcripts/HiTOP/Audio_Segments'
WTC_AUDIO_DIR = '/cronus_data/wtc_clinic/Clinic_Audio_Segments/'


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, processor):
        self.hitop_segments_messages = pd.read_csv('/cronus_data/rrao/hitop/segment_table.csv')
        self.hitop_segments_messages = self.hitop_segments_messages.dropna().reset_index(drop=True)
        self.wtc_segments_messages = pd.read_csv('/cronus_data/rrao/wtc_clinic/segment_table.csv')
        self.wtc_segments_messages = self.wtc_segments_messages.dropna().reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.hitop_segments_messages) + len(self.wtc_segments_messages)
    
    def __getitem__(self, idx):
        audio_dir = HITOP_AUDIO_DIR if idx < len(self.hitop_segments_messages) else WTC_AUDIO_DIR
        i = idx if idx < len(self.hitop_segments_messages) else idx - len(self.hitop_segments_messages)
        df = self.hitop_segments_messages if idx < len(self.hitop_segments_messages) else self.wtc_segments_messages
        
        audio_path = os.path.join(audio_dir, df.iloc[i]['segment_filename'])
        return preprocess_audio(self.processor, audio_path), df.iloc[i]['segment_message']


def preprocess_audio(processor, audio_path):
    # Whisper Audio Pre-Processing
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert stereo (or multi-channel) to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")['input_features']


def collate(batch):
    return {
        'audio_inputs': torch.cat([a for a, _ in batch], dim=0),
        'text': [t for _, t  in batch]
    }
