CACHE_DIR = '/cronus_data/rrao/cache/'
CHECKPOINT_DIR = '/cronus_data/rrao/WhiSBERT/models/'


class WhiSBERTConfig():
    def __init__(
        self,
        whisper_model_id: str = 'openai/whisper-tiny',
        pooling_mode: str = 'mean',
        loss: str = 'cos_sim',
        use_sbert_encoder: bool = False,
        new_encoder_n_layers: int = 12,
        new_encoder_n_heads: int = 12,
        new_encoder_ffn_dim: int = 3072,
        activation_function: str = 'gelu',
        tau: float = 0.1,
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
        whisper_model_id_choices = [
            'openai/whisper-tiny',
            'openai/whisper-base',
            'openai/whisper-small'
        ]
        sbert_model_id_choices = [
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/distiluse-base-multilingual-cased-v1',
            'sentence-transformers/all-mpnet-base-v2'
        ]
        emb_dim_choices = [384, 512, 768]

        try:
            self.emb_dim = emb_dim_choices[whisper_model_id_choices.index(whisper_model_id)]
            self.sbert_model_id = sbert_model_id_choices[whisper_model_id_choices.index(whisper_model_id)]
            self.whisper_model_id = whisper_model_id
        except ValueError:
            self.emb_dim = emb_dim_choices[0]
            self.sbert_model_id = sbert_model_id_choices[0]
            self.whisper_model_id = whisper_model_id_choices[0]
                
        # Model hyperparameters
        self.pooling_mode = pooling_mode
        self.loss = loss
        self.use_sbert_encoder = use_sbert_encoder
        self.new_encoder_n_layers = new_encoder_n_layers
        self.new_encoder_n_heads = new_encoder_n_heads
        self.new_encoder_ffn_dim = new_encoder_ffn_dim
        self.activation_function = activation_function
        self.tau = tau
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

    def __str__(self):
        from pprint import pformat
        return pformat(self.__dict__)