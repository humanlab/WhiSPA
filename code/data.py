import os
import numpy as np
import pandas as pd
import torch, torchaudio
from config import SBERT_384_DIM_INDECES


HITOP_AUDIO_DIR = '/cronus_data/hitop/iHiTOP_transcripts/HiTOP/Audio_Segments'
WTC_AUDIO_DIR = '/cronus_data/wtc_clinic/Clinic_Audio_Segments/'


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, processor, mode='train'):
        self.hitop_segments_df = pd.read_csv('/cronus_data/rrao/hitop/seg_persona.csv')
        self.wtc_segments_df = pd.read_csv('/cronus_data/rrao/wtc_clinic/seg_persona.csv')
        self.processor = processor
        self.mode = mode

        if mode == 'train':
            # Load SBERT Embeddings (mean and std)
            if config.sbert_model_id == 'sentence-transformers/all-MiniLM-L12-v2':
                sbert_mean = np.load('/cronus_data/rrao/WhiSBERT/embeddings/all-MiniLM-L12-v2/mean_emb.npy')[SBERT_384_DIM_INDECES]
                sbert_std = np.load('/cronus_data/rrao/WhiSBERT/embeddings/all-MiniLM-L12-v2/std_emb.npy')[SBERT_384_DIM_INDECES]

            for i, feat in enumerate(['valence', 'arousal', 'ope', 'agr', 'ext', 'con', 'neu', 'ang_norm', 'anx_norm', 'dep_norm']):
                psych_feats = np.concatenate([self.wtc_segments_df[feat].to_numpy(), self.hitop_segments_df[feat].to_numpy()])
                
                # Z-Score Normalization for the psychological features
                psych_mean, psych_std = psych_feats.mean(), psych_feats.std()
                self.wtc_segments_df[feat] = (self.wtc_segments_df[feat] - psych_mean) / psych_std
                self.hitop_segments_df[feat] = (self.hitop_segments_df[feat] - psych_mean) / psych_std

                # Rescale to match SBERT's Dimensional Distribution
                self.wtc_segments_df[feat] = self.wtc_segments_df[feat] * sbert_std[i] + sbert_mean[i]
                self.hitop_segments_df[feat] = self.hitop_segments_df[feat] * sbert_std[i] + sbert_mean[i]
    
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
        
        audio_path = os.path.join(audio_dir, df.iloc[i]['filename'])
        audio_inputs = preprocess_audio(self.processor, audio_path)
        message = df.iloc[i]['message']
        if self.mode == 'train':
            return audio_inputs, message, torch.from_numpy(df.iloc[0][4:].to_numpy(dtype=np.float32)).unsqueeze(0)
        elif self.mode == 'inference':
            return dataset_name, df.iloc[i]['message_id'], audio_inputs, message
        else:
            return None
    
    def __len__(self):
        return len(self.hitop_segments_df) + len(self.wtc_segments_df)


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
        'message_id': [i for _, i, _, _ in batch],
        'audio_inputs': torch.cat([a for _, _, a, _ in batch], dim=0),
        'message': [m for _, _, _, m  in batch]
    }