import os
import json
import torch


class WhiSPAConfig():
    def __init__(
        self,
        whisper_model_id: str = 'openai/whisper-medium',
        language_model_id: str = 'Qwen/Qwen3-Embedding-0.6B',
        # acoustic_teacher_id: str = 'hubert-large-ls960-ft',
        use_teacher_cache: bool = True,
        stage: str = 'encode',
        pooling_mode: str = 'mean',
        hidden_size: int = 1024,
        n_mel_bins: int = 80,
        # max_source_positions: int = 1500,
        # spectral_decoder_n_layers: int = 6,
        # spectral_decoder_n_heads: int = 8,
        # spectral_decoder_ffn_dim: int = 4096,
        loss: str = 'NCE',
        dtype: torch.dtype = torch.bfloat16,
        alpha: float = 1.0, # NCE loss weight
        beta: float = 0.1, # Spectral recon loss weight
        tau: float = 0.1, 
        batch_size: int = 1,
        num_workers: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-2,
        shuffle: bool = True,
        device: str = 'cpu',
        **kwargs,
    ):
        # Model IDs
        self.whisper_model_id = whisper_model_id
        self.linguistic_teacher_id = language_model_id
        # self.acoustic_teacher_id = acoustic_teacher_id
        self.use_teacher_cache = use_teacher_cache

        stages = {'train_enc', 'train_dec', 'encode', 'decode'}
        if stage not in stages:
            raise ValueError(f"Invalid stage: `{stage}`. Must be one of [{stages}].")
        self.stage = stage
                
        # Model hyperparameters
        self.pooling_mode = pooling_mode
        self.hidden_size = hidden_size
        self.n_mel_bins = n_mel_bins
        # self.max_source_positions = max_source_positions
        # self.spectral_decoder_n_layers = spectral_decoder_n_layers
        # self.spectral_decoder_n_heads = spectral_decoder_n_heads
        # self.spectral_decoder_ffn_dim = spectral_decoder_ffn_dim
        self.loss = loss
        self.dtype = dtype
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        # store any extra args
        for k, v in kwargs.items():
            setattr(self, k, v)


    def __str__(self):
        from pprint import pformat
        return pformat(self.__dict__)
    

    def to_dict(self):
        return self.__dict__


    def save_pretrained(self, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, local_dir: str):
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
