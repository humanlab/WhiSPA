import os
import numpy as np
import pandas as pd
import torch, torchaudio


HITOP_AUDIO_DIR = '/cronus_data/hitop/iHiTOP_transcripts/HiTOP/Audio_Segments'
WTC_AUDIO_DIR = '/cronus_data/wtc_clinic/Clinic_Audio_Segments/'


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, processor, mode='train'):
        self.hitop_segments_df = pd.read_csv('/cronus_data/rrao/hitop/segment_outcomes.csv').dropna()
        self.wtc_segments_df = pd.read_csv('/cronus_data/rrao/wtc_clinic/segment_outcomes.csv').dropna()
        self.processor = processor
        self.mode = mode

        if mode == 'train':
            for feat in ['valence', 'arousal', 'ope', 'agr', 'ext', 'con', 'neu']:
                wtc_data = np.concatenate([self.wtc_segments_df[feat].to_numpy(), self.wtc_segments_df[feat].to_numpy()])
                wtc_min, wtc_max = wtc_data.min(), wtc_data.max()
                self.wtc_segments_df[feat] = 2 * ((self.wtc_segments_df[feat] - wtc_min) / (wtc_max - wtc_min)) - 1
                hitop_data = np.concatenate([self.hitop_segments_df[feat].to_numpy(), self.hitop_segments_df[feat].to_numpy()])
                hitop_min, hitop_max = hitop_data.min(), hitop_data.max()
                self.hitop_segments_df[feat] = 2 * ((self.hitop_segments_df[feat] - hitop_min) / (hitop_max - hitop_min)) - 1

    def __len__(self):
        return len(self.hitop_segments_df) + len(self.wtc_segments_df)
    
    def __getitem__(self, idx):
        audio_dir = HITOP_AUDIO_DIR if idx < len(self.hitop_segments_df) else WTC_AUDIO_DIR
        if idx < len(self.hitop_segments_df):
            i = idx
            df = self.hitop_segments_df
            dataset_name = 'hitop'
        else:
            i = idx - len(self.hitop_segments_df)
            df = self.wtc_segments_df
            dataset_name = 'wtc'
        
        audio_path = os.path.join(audio_dir, df.iloc[i]['segment_filename'])
        audio_inputs = preprocess_audio(self.processor, audio_path)
        message = df.iloc[i]['segment_message']
        if self.mode == 'train':
            return audio_inputs, message, torch.from_numpy(df.iloc[0][4:].to_numpy(dtype=np.float32)).unsqueeze(0)
        elif self.mode == 'inference':
            return dataset_name, df.iloc[i]['segment_id'], audio_inputs, message
        else:
            return None


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


def collate_train(batch):
    return {
        'audio_inputs': torch.cat([a for a, _, _ in batch], dim=0),
        'message': [m for _, m, _  in batch],
        'outcomes': torch.cat([o for _, _, o in batch], dim=0)
    }


def collate_inference(batch):
    return {
        'dataset_name': [d for d, _, _, _ in batch],
        'segment_id': [i for _, i, _, _ in batch],
        'audio_inputs': torch.cat([a for _, _, a, _ in batch], dim=0),
        'message': [m for _, _, _, m  in batch]
    }