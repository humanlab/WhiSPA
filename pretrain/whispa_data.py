import os
import logging
import numpy as np
import pandas as pd
import torch, torchaudio
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv

load_dotenv()


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, processors, dtype=torch.bfloat16, mode='train'):
        self.config = config
        self.processors = processors
        self.dtype = dtype
        self.mode = mode

        self.hitop_segments_df = pd.read_csv(f'{os.getenv("HITOP_DATA_DIR")}/whispa_dataset.csv').sort_values(by='message_id')
        self.wtc_segments_df = pd.read_csv(f'{os.getenv("WTC_DATA_DIR")}/whispa_dataset.csv').sort_values(by='message_id')
        
        if mode == 'train':
            linguistic_embs_path = os.path.join(
                os.getenv('EMBEDDINGS_DIR'),
                config.linguistic_teacher_id[config.linguistic_teacher_id.find('/') + 1:]
            )
            acoustic_embs_path = os.path.join(
                os.getenv('EMBEDDINGS_DIR'),
                'whisper_enc_1024' if config.acoustic_teacher_id == 'openai/whisper-medium' \
                    else config.acoustic_teacher_id[config.acoustic_teacher_id.find('/') + 1:]
            )

            if config.use_teacher_cache:
                logging.info('Caching teacher embeddings...')
                self.linguistic_embs_hitop_df = pd.read_csv(
                    os.path.join(linguistic_embs_path, 'hitop_embeddings.csv')
                ).drop_duplicates(subset=['message_id']).sort_values(by='message_id')
                self.linguistic_embs_wtc_df = pd.read_csv(
                    os.path.join(linguistic_embs_path, 'wtc_embeddings.csv')
                ).drop_duplicates(subset=['message_id']).sort_values(by='message_id')
                logging.info('  Linguistic embeddings loaded.')
                
                self.acoustic_embs_hitop_df = pd.read_csv(
                    os.path.join(acoustic_embs_path, 'hitop_embeddings.csv')
                ).drop_duplicates(subset=['message_id']).sort_values(by='message_id')
                self.acoustic_embs_wtc_df = pd.read_csv(
                    os.path.join(acoustic_embs_path, 'wtc_embeddings.csv')
                ).drop_duplicates(subset=['message_id']).sort_values(by='message_id')
                logging.info('  Acoustic embeddings loaded.')

            if self.config.n_new_dims:
                # Load SBERT Mean and Standard Dimensional Distribution
                linguistic_embs_mean = np.load(os.path.join(linguistic_embs_path, 'mean_emb.npy')).mean()
                linguistic_embs_std = np.load(os.path.join(linguistic_embs_path, 'std_emb.npy')).mean()

                for feat in ['ope', 'agr', 'ext', 'con', 'neu', 'valence', 'arousal', 'ang', 'anx', 'dep']:
                    # Z-Score Normalization for the psychological features
                    self.hitop_segments_df[feat] = (self.hitop_segments_df[feat] - self.hitop_segments_df[feat].mean()) / self.hitop_segments_df[feat].std()
                    self.wtc_segments_df[feat] = (self.wtc_segments_df[feat] - self.wtc_segments_df[feat].mean()) / self.wtc_segments_df[feat].std()

                    # Rescale to match SBERT's Dimensional Distribution
                    self.hitop_segments_df[feat] = self.hitop_segments_df[feat] * linguistic_embs_std + linguistic_embs_mean
                    self.wtc_segments_df[feat] = self.wtc_segments_df[feat] * linguistic_embs_std + linguistic_embs_mean
    
    def __getitem__(self, idx):
        audio_dir = os.getenv('HITOP_AUDIO_DIR') if idx < len(self.hitop_segments_df) else os.getenv('WTC_AUDIO_DIR')
        if idx < len(self.hitop_segments_df):
            dataset_name = 'hitop'
            i = idx
            df = self.hitop_segments_df
            if self.mode == 'train' and self.config.use_teacher_cache:
                linguistic_embs_df = self.linguistic_embs_hitop_df
                acoustic_embs_df = self.acoustic_embs_hitop_df
                if len(self.processors) > 1:
                    self.processors[-1] = None
        else:
            dataset_name = 'wtc'
            i = idx - len(self.hitop_segments_df)
            df = self.wtc_segments_df
            if self.mode == 'train' and self.config.use_teacher_cache:
                linguistic_embs_df = self.linguistic_embs_wtc_df
                acoustic_embs_df = self.acoustic_embs_wtc_df
                if len(self.processors) > 1:
                    self.processors[-1] = None
        
        audio_inputs = []
        for processor in self.processors:
            if processor is None:
                audio_input = None
            else:
                waveform = preprocess_audio(os.path.join(audio_dir, df.iloc[i]['filename']))
                audio_input = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
                try:
                    audio_input = audio_input['input_features']
                except:
                    audio_input = audio_input['input_values']
            audio_inputs.append(audio_input)
        
        if self.mode == 'train':
            return (
                audio_inputs[0].to(self.dtype),
                df.iloc[i]['message'],
                audio_inputs[1].to(self.dtype) if audio_inputs[1] is not None else None,
                torch.from_numpy(linguistic_embs_df.iloc[i][1:].to_numpy(np.float32)).unsqueeze(0) if self.config.use_teacher_cache else None,
                torch.from_numpy(acoustic_embs_df.iloc[i][1:].to_numpy(np.float32)).unsqueeze(0) if self.config.use_teacher_cache else None,
                torch.from_numpy(df.iloc[i][4:].to_numpy(np.float32)).unsqueeze(0) if self.config.n_new_dims else None
            )
        
        elif self.mode == 'inference':
            return (
                dataset_name,
                df.iloc[i]['message_id'],
                audio_inputs[0].to(self.dtype),
                df.iloc[i]['message']
            )
        else:
            return None
    
    def __len__(self):
        return len(self.hitop_segments_df) + len(self.wtc_segments_df)


def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert stereo (or multi-channel) to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform


def collate_train(batch):
    audio_inputs = None
    try:
        # Batch padding
        audio_inputs = torch.cat([w for w, _, _, _, _, _ in batch], dim=0)
    except Exception:
        audio_inputs = pad_sequence(
            [w.squeeze(0) for w, _, _, _, _, _ in batch],
            batch_first=True,
            padding_value=0.0
        )
    
    acoustic_inputs = None
    # Check if acoustic inputs are preprocessed
    if isinstance(batch[0][2], torch.Tensor):
        # Truncate preprocessed audio to max length
        MAX_LENGTH = 400000
        acoustic_inputs = [a[:, :MAX_LENGTH] for _, _, a, _, _, _ in batch]
        # Batch padding
        try:
            acoustic_inputs = torch.cat(acoustic_inputs, dim=0)
        except Exception:
            acoustic_inputs = pad_sequence(
                [a.squeeze(0) for a in acoustic_inputs],
                batch_first=True,
                padding_value=0.0
            )

    return {
        'audio_inputs': audio_inputs,
        'message': [m for _, m, _, _, _, _  in batch],
        'acoustic_inputs': acoustic_inputs,
        'linguistic_embs': torch.cat([l for _, _, _, l, _, _ in batch], dim=0) if isinstance(batch[0][-3], torch.Tensor) else None,
        'acoustic_embs': torch.cat([a for _, _, _, _, a, _ in batch], dim=0) if isinstance(batch[0][-2], torch.Tensor) else None,
        'psych_embs': torch.cat([o for _, _, _, _, _, o in batch], dim=0) if isinstance(batch[0][-1], torch.Tensor) else None
    }


def collate_inference(batch):
    try:
        audio_inputs = torch.cat([a for _, _, a, _ in batch], dim=0) if isinstance(batch[0][2], torch.Tensor) else None
    except Exception:
        audio_inputs = pad_sequence(
            [a.squeeze(0) for _, _, a, _ in batch],
            batch_first=True,
            padding_value=0.0
        )
    return {
        'dataset_name': [d for d, _, _, _ in batch],
        'message_id': [i for _, i, _, _ in batch],
        'audio_inputs': audio_inputs,
        'message': [m for _, _, _, m  in batch]
    }