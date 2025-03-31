import os
import json
import torch


class WhiSPAConfig():
    def __init__(
        self,
        whisper_model_id: str = 'openai/whisper-medium',
        pooling_mode: str = 'mean',
        n_new_dims: int = 0,
        loss: str = 'NCE',
        alpha: float = 0.5,
        beta: float = 0.5,
        rho: float = 0.0,
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
        self.linguistic_teacher_id = 'jinaai/jina-embeddings-v3'
        self.acoustic_teacher_id = 'facebook/hubert-large-ls960-ft'
                
        # Model hyperparameters
        self.hidden_size = 1024
        self.pooling_mode = pooling_mode
        self.n_new_dims = n_new_dims
        self.loss = loss
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
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
