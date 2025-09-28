import os
import json
import torch


class WhiSPAConfig():
    def __init__(
        self,
        backbone_model_id: str = 'mistralai/Voxtral-Mini-3B-2507',
        stage: str = 'inference',
        pooling_mode: str = 'mean',
        hidden_size: int = 1024,
        n_mel_bins: int = 80,
        # max_source_positions: int = 1500,
        # spectral_decoder_n_layers: int = 6,
        # spectral_decoder_n_heads: int = 8,
        # spectral_decoder_ffn_dim: int = 4096,
        loss: str = 'MMRL',
        dtype: torch.dtype = torch.bfloat16,
        tau: float = 0.1, # Temperature for NCE
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
        backbones = {'mistralai/Voxtral-Mini-3B-2507', 'mistralai/Voxtral-Small-24B-2507'}
        if backbone_model_id not in backbones:
            raise ValueError(f"Invalid backbone model: `{backbone_model_id}`. Must be one of [{backbones}].")
        self.backbone_model_id = backbone_model_id

        stages = {'train_enc', 'train_dec', 'inference'}
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

        """ 
        - MMRL: Manifolded Matryoshka Representation Loss
        - TAL: Trinary Alignment Loss
        """
        losses = {'MMRL', 'TAL'}
        if loss not in losses:
            raise ValueError(f"Invalid loss: `{loss}`. Must be one of [{losses}].")
        self.loss = loss
        self.dtype = dtype
        # self.alpha = alpha
        # self.beta = beta
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
        
        cfg = self.to_dict().copy()
        # Convert non-JSON-serializable fields (e.g., torch.dtype) to strings
        if isinstance(cfg.get("dtype"), torch.dtype):
            cfg["dtype"] = str(cfg["dtype"])
        if isinstance(cfg.get("device"), torch.device):
            cfg["device"] = str(cfg["device"])
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


    @classmethod
    def from_pretrained(cls, local_dir: str):
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Restore torch.dtype if serialized as string
        dtype_val = config_dict.get("dtype")
        if isinstance(dtype_val, str) and dtype_val.startswith("torch."):
            try:
                torch_dtype_name = dtype_val.split(".", 1)[1]
                if hasattr(torch, torch_dtype_name):
                    config_dict["dtype"] = getattr(torch, torch_dtype_name)
                else:
                    config_dict["dtype"] = torch.bfloat16
            except Exception:
                pass

        # Restore torch.device if serialized as string
        device_val = config_dict.get("device")
        if isinstance(device_val, str) and device_val.startswith("torch."):
            try:
                torch_device_name = device_val.split(".", 1)[1]
                if hasattr(torch, torch_device_name):
                    config_dict["device"] = getattr(torch, torch_device_name)
                else:
                    config_dict["device"] = torch.device("cpu")
            except Exception:
                pass
        return cls(**config_dict)
