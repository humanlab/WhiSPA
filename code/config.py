CACHE_DIR = '/cronus_data/rrao/cache/'
CHECKPOINT_DIR = '/cronus_data/rrao/WhiSBERT/models/'


class WhiSBERTConfig():
    def __init__(
        self,
        whisper_model_id: str = 'openai/whisper-small',
        pooling_mode: str = 'cls',
        loss: str = 'cos_sim',
        use_sbert_layers: bool = False,
        new_encoder_n_layers: int = 12,
        new_encoder_n_heads: int = 12,
        new_encoder_ffn_dim: int = 3072,
        activation_function: str = 'gelu',
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
        self.use_sbert_layers = use_sbert_layers
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
