import os
import numpy as np
import pandas as pd
import torch, torchaudio
from torch.nn.utils.rnn import pad_sequence
import transformers
from dotenv import load_dotenv

load_dotenv()


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, processors, use_psych=False, mode='train'):
        self.config = config
        self.hitop_segments_df = pd.read_csv(f'{os.getenv("HITOP_DATA_DIR")}/whispa_dataset.csv')
        self.wtc_segments_df = pd.read_csv(f'{os.getenv("WTC_DATA_DIR")}/whispa_dataset.csv')
        self.processors = processors
        self.use_psych = use_psych
        self.mode = mode

        if mode == 'train' and self.use_psych:
            # Load SBERT Mean and Standard Dimensional Distribution
            sbert_emb_path = os.path.join(os.getenv('EMBEDDINGS_DIR'), config.sbert_model_id.replace('sentence-transformers/', ''))
            sbert_mean = np.load(os.path.join(sbert_emb_path, 'mean_emb.npy')).mean()
            sbert_std = np.load(os.path.join(sbert_emb_path, 'std_emb.npy')).mean() # THIS IS NOT A TYPO

            for feat in ['ope', 'agr', 'ext', 'con', 'neu', 'valence', 'arousal', 'ang', 'anx', 'dep']:
                # Z-Score Normalization for the psychological features
                self.hitop_segments_df[feat] = (self.hitop_segments_df[feat] - self.hitop_segments_df[feat].mean()) / self.hitop_segments_df[feat].std()
                self.wtc_segments_df[feat] = (self.wtc_segments_df[feat] - self.wtc_segments_df[feat].mean()) / self.wtc_segments_df[feat].std()

                # Rescale to match SBERT's Dimensional Distribution
                self.hitop_segments_df[feat] = self.hitop_segments_df[feat] * sbert_std + sbert_mean
                self.wtc_segments_df[feat] = self.wtc_segments_df[feat] * sbert_std + sbert_mean
    
    def __getitem__(self, idx):
        audio_dir = os.getenv('HITOP_AUDIO_DIR') if idx < len(self.hitop_segments_df) else os.getenv('WTC_AUDIO_DIR')
        if idx < len(self.hitop_segments_df):
            i = idx
            df = self.hitop_segments_df
            dataset_name = 'hitop'
        else:
            i = idx - len(self.hitop_segments_df)
            df = self.wtc_segments_df
            dataset_name = 'wtc'
        
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
            # if isinstance(self.processor, transformers.models.whisper.processing_whisper.WhisperProcessor) \
            # or isinstance(self.processor, transformers.models.wav2vec2_bert.processing_wav2vec2_bert.Wav2Vec2BertProcessor):
            #     audio_inputs = audio_inputs['input_features']
            # elif isinstance(self.processor, transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2Processor):
            #     audio_inputs = audio_inputs['input_values']
        
        if self.mode == 'train':
            return (
                audio_inputs[0],
                df.iloc[i]['message'],
                audio_inputs[1],
                torch.from_numpy(df.iloc[i][4:].to_numpy(dtype=np.float32)).unsqueeze(0) if self.use_psych else None
            )
        elif self.mode == 'inference':
            return (
                dataset_name,
                df.iloc[i]['message_id'],
                audio_inputs[0],
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
    # Truncate HuBERT's preprocessed tensors to max length
    MAX_LENGTH = 400000
    hubert_inputs = [h[:, :MAX_LENGTH] for _, _, h, _ in batch]

    # Batch padding
    try:
        whisper_inputs = torch.cat([w for w, _, _, _ in batch], dim=0) if isinstance(batch[0][0], torch.Tensor) else None
        hubert_inputs = torch.cat(hubert_inputs, dim=0) if isinstance(batch[0][2], torch.Tensor) else None
    except Exception:
        whisper_inputs = pad_sequence(
            [w.squeeze(0) for w, _, _, _ in batch],
            batch_first=True,
            padding_value=0.0
        )
        hubert_inputs = pad_sequence(
            [h.squeeze(0) for h in hubert_inputs],
            batch_first=True,
            padding_value=0.0
        )
    
    return {
        'whisper_inputs': whisper_inputs,
        'message': [m for _, m, _, _  in batch],
        'hubert_inputs': hubert_inputs,
        'psych_emb': torch.cat([o for _, _, _, o in batch], dim=0) if isinstance(batch[0][-1], torch.Tensor) else None
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