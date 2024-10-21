from typing import List, Optional, Union
import os
import pandas as pd
import torch
import torchaudio
from transformers import WhisperModel


CACHE_DIR = '/cronus_data/rrao/cache/'
AUDIO_DIR = '/cronus_data/wtc_clinic/Clinic_Audio_Segments/'


class WhiSBERTConfig():
    def __init__(
        self,
        whisper_model_id: Optional[str] = None,
        pooling_mode: str = "cls",
        loss: str = "cos_sim",
        use_new_encoder_layers: bool = False,
        new_encoder_n_layers: int = 12,
        new_encoder_n_heads: int = 12,
        new_encoder_ffn_dim: int = 3072,
        activation_function: str = "gelu",
        eps: float = 1e-5,
        dropout: float = 0.1,
        encoder_layerdrop: float = 0.1,
        decoder_layerdrop: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        shuffle: bool = True,
        device: str = 'cpu',
        **kwargs,
    ):
        # Model hyperparameters
        try:
            self.emb_dim = (768, 1024)[['openai/whisper-small', 'openai/whisper-medium'].index(whisper_model_id)]
            self.whisper_model_id = whisper_model_id
        except ValueError:
            self.emb_dim = 768
            self.whisper_model_id = 'openai/whisper-small'
        self.pooling_mode = pooling_mode
        self.loss = loss
        self.use_new_encoder_layers = use_new_encoder_layers
        self.new_encoder_n_layers = new_encoder_n_layers
        self.new_encoder_n_heads = new_encoder_n_heads
        self.new_encoder_ffn_dim = new_encoder_ffn_dim
        self.activation_function = activation_function
        self.eps = eps
        self.dropout = dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device


class WhiSBERTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.whisper_model = WhisperModel.from_pretrained(self.config.whisper_model_id, cache_dir=CACHE_DIR).to(self.config.device)

        if self.config.use_new_encoder_layers:
            self.new_encoder_layers = torch.nn.ModuleList([
                WhiSBERTEncoderLayer(self.config) 
                for _ in range(self.config.new_encoder_n_layers)
            ])

    def forward(self, audio_inputs, text_input_ids, text_attention_mask):
        embs = self.whisper_model(
            audio_inputs,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask
        ).last_hidden_state
        
        if self.config.use_new_encoder_layers:
            for layer in self.new_encoder_layers:
                embs = layer(embs)
        
        if self.config.pooling_mode == 'cls':
            if self.config.use_new_encoder_layers:
                embs = embs[:, 0, :]
            else:
                non_padding_indices = text_attention_mask.cumsum(dim=1) - 1
                last_non_padding_indices = non_padding_indices.gather(1, (text_attention_mask.sum(dim=1, keepdim=True) - 1).clamp(min=0).long())
                embs = embs[torch.arange(text_attention_mask.size(0)).unsqueeze(1), last_non_padding_indices].squeeze()
        else:
            sum_embs = (embs * text_attention_mask.unsqueeze(-1).expand(embs.size())).sum(dim=1)
            non_padding_counts = text_attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            embs = sum_embs / non_padding_counts
        return embs


# TO-DO: Add attention masks for forward pass of self-attention block
class WhiSBERTEncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim = config.emb_dim, 
            num_heads = config.new_encoder_n_heads,
            dropout = config.attention_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(config.emb_dim, eps=config.eps)
        self.fc1 = torch.nn.Linear(config.emb_dim, config.new_encoder_ffn_dim)
        self.fc2 = torch.nn.Linear(config.new_encoder_ffn_dim, config.emb_dim)
        self.activation_fn = torch.nn.GELU() if config.activation_function == 'gelu' else torch.nn.ReLU()
        self.final_layer_norm = torch.nn.LayerNorm(config.emb_dim, eps=config.eps)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        # Self-attention block
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Feed-forward block
        ffn_output = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = hidden_states + self.dropout(ffn_output)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states
    

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, whisper_processor):
        self.df = pd.read_csv(df_path)
        self.processor = whisper_processor

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(AUDIO_DIR, self.df.iloc[idx]['segment_filename'])
        return preprocess_audio(self.processor, audio_path), self.df.iloc[idx]['segment_message']


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
