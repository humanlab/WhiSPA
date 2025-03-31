import os
import json
import torch


class WhiSPAConfig():
    def __init__(
        self,
        whisper_model_id: str = 'openai/whisper-medium',
        pooling_mode: str = 'mean',
        with_bidirectionality: bool = False,
        n_new_dims: int = 0,
        use_psych: bool = False,
        loss: str = 'NCE',
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
        whisper_model_id_choices = [
            'openai/whisper-tiny',
            'openai/whisper-small',
            'openai/whisper-medium'
        ]
        sbert_model_id_choices = [
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/all-roberta-large-v1'
        ]
        emb_dims_choices = [384, 768, 1024]

        try:
            self.emb_dims = emb_dims_choices[whisper_model_id_choices.index(whisper_model_id)]
            self.sbert_model_id = sbert_model_id_choices[whisper_model_id_choices.index(whisper_model_id)]
            self.whisper_model_id = whisper_model_id
        except ValueError:
            self.emb_dims = emb_dims_choices[0]
            self.sbert_model_id = sbert_model_id_choices[0]
            self.whisper_model_id = whisper_model_id_choices[0]
                
        # Model hyperparameters
        self.pooling_mode = pooling_mode
        self.with_bidirectionality = with_bidirectionality
        self.n_new_dims = n_new_dims
        self.use_psych = use_psych
        self.loss = loss
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
