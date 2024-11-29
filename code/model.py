import torch
from transformers import AutoModel, WhisperModel
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR
from utils import mean_pooling, last_pooling


class WhiSBERTModel(torch.nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.pooling_mode == 'mean':
            self.pooler = mean_pooling
        elif config.pooling_mode == 'last':
            self.pooler = last_pooling

        self.whisper_model = WhisperModel.from_pretrained(
            config.whisper_model_id,
            cache_dir=CACHE_DIR,
        )

        # Use HuggingFace library for:
        # - "sentence-transformers/all-MiniLM-L12-v2"
        # - "sentence-transformers/all-mpnet-base-v2"
        self.sbert_model = AutoModel.from_pretrained(
            config.sbert_model_id,
            cache_dir=CACHE_DIR
        )

        if config.use_sbert_encoder:
            self.sbert_encoder = self.sbert_model.encoder.to(config.device)

        if config.n_new_dims:
            # Learnable Emotion (PA) Projection Matrix (D x 10)
            self.projection = torch.nn.Linear(config.emb_dim, config.n_new_dims).to(config.device)
            # Dynamic Loss Balancing
            # self.log_sigma_cos = torch.nn.Parameter(torch.tensor(0.0))
            # self.log_sigma_mse = torch.nn.Parameter(torch.tensor(0.0))

        self.whisper_model.to(config.device)
        self.linear = self.sbert_model.pooler.dense.to(config.device)
        self.activation = self.sbert_model.pooler.activation.to(config.device)
    

    def expand_model(self):
        # WHISPER ENCODER EXPANSION
        self.whisper_model.encoder.conv1 = expand_conv1d_layer(self.whisper_model.encoder.conv1, added_out_channels=self.config.n_new_dims)
        self.whisper_model.encoder.conv2 = expand_conv1d_layer(self.whisper_model.encoder.conv2, added_in_channels=self.config.n_new_dims, added_out_channels=self.config.n_new_dims)

        self.whisper_model.encoder.embed_positions = expand_embedding_layer(self.whisper_model.encoder.embed_positions, self.config.n_new_dims, distribution='zeros')
        self.whisper_model.encoder.embed_positions.weight.requires_grad = False

        for layer in self.whisper_model.encoder.layers:
            layer.self_attn.k_proj = expand_linear_layer(layer.self_attn.k_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.v_proj = expand_linear_layer(layer.self_attn.v_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.q_proj = expand_linear_layer(layer.self_attn.q_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.out_proj = expand_linear_layer(layer.self_attn.out_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn_layer_norm = expand_layer_norm(layer.self_attn_layer_norm, self.config.n_new_dims)
            layer.fc1 = expand_linear_layer(layer.fc1, added_in_features=self.config.n_new_dims)
            layer.fc2 = expand_linear_layer(layer.fc2, added_out_features=self.config.n_new_dims)
            layer.final_layer_norm = expand_layer_norm(layer.final_layer_norm, self.config.n_new_dims)

        self.whisper_model.encoder.layer_norm = expand_layer_norm(self.whisper_model.encoder.layer_norm, self.config.n_new_dims)

        # WHISPER DECODER EXPANSION
        self.whisper_model.decoder.embed_tokens = expand_embedding_layer(self.whisper_model.decoder.embed_tokens, self.config.n_new_dims, distribution='normal')
        self.whisper_model.decoder.embed_positions = expand_positional_embedding(self.whisper_model.decoder.embed_positions, self.config.n_new_dims)

        for layer in self.whisper_model.decoder.layers:
            layer.self_attn.k_proj = expand_linear_layer(layer.self_attn.k_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.v_proj = expand_linear_layer(layer.self_attn.v_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.q_proj = expand_linear_layer(layer.self_attn.q_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn.out_proj = expand_linear_layer(layer.self_attn.out_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.self_attn_layer_norm = expand_layer_norm(layer.self_attn_layer_norm, self.config.n_new_dims)
            layer.encoder_attn.k_proj = expand_linear_layer(layer.encoder_attn.k_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.encoder_attn.v_proj = expand_linear_layer(layer.encoder_attn.v_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.encoder_attn.q_proj = expand_linear_layer(layer.encoder_attn.q_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.encoder_attn.out_proj = expand_linear_layer(layer.encoder_attn.out_proj, self.config.n_new_dims, self.config.n_new_dims)
            layer.encoder_attn_layer_norm = expand_layer_norm(layer.encoder_attn_layer_norm, self.config.n_new_dims)
            layer.fc1 = expand_linear_layer(layer.fc1, added_in_features=self.config.n_new_dims)
            layer.fc2 = expand_linear_layer(layer.fc2, added_out_features=self.config.n_new_dims)
            layer.final_layer_norm = expand_layer_norm(layer.final_layer_norm, self.config.n_new_dims)

        self.whisper_model.decoder.layer_norm = expand_layer_norm(self.whisper_model.decoder.layer_norm, self.config.n_new_dims)
        

    def forward(self, audio_inputs, text_input_ids, text_attention_mask):
        embs = self.whisper_model(
            audio_inputs,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask
        ).last_hidden_state
        
        if self.config.use_sbert_encoder:
            embs = self.sbert_encoder(
                embs,
                attention_mask=self.sbert_model.get_extended_attention_mask(
                    text_attention_mask,
                    embs.size()[:-1]
                ),
                head_mask=[None] * self.sbert_model.config.num_hidden_layers
            )[0]
        
        embs = self.pooler(embs, text_attention_mask)
        embs = self.linear(embs)
        embs = self.activation(embs)

        if self.config.n_new_dims:
            pa = self.activation(self.projection(embs))
            return torch.cat([embs, pa], dim=1)
        else:
            return embs


def expand_conv1d_layer(conv1d_layer, added_in_channels=None, added_out_channels=None):
    """
    Expands the input and/or output channels of a Conv1d layer while retaining the original weights
    and initializing the new channels with random values.
    
    Args:
        conv1d_layer (nn.Conv1d): Original Conv1d layer to expand.
        added_in_channels (int, optional): Number of new input channels to add. If `None` or 0, input channels remain unchanged.
        added_out_channels (int, optional): Number of new output channels to add. If `None` or 0, output channels remain unchanged.
    
    Returns:
        nn.Conv1d: New Conv1d layer with expanded input and/or output channels.
    """
    old_weight = conv1d_layer.weight.data
    old_bias = conv1d_layer.bias.data if conv1d_layer.bias is not None else None

    # Determine new dimensions
    new_out_channels = old_weight.shape[0] + (added_out_channels or 0)
    new_in_channels = old_weight.shape[1] + (added_in_channels or 0)
    kernel_size = conv1d_layer.kernel_size
    stride = conv1d_layer.stride
    padding = conv1d_layer.padding

    # Create the new Conv1d layer
    new_conv1d = torch.nn.Conv1d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=conv1d_layer.bias is not None,
    )

    # Copy the old weights to the appropriate slice
    new_weight = new_conv1d.weight.data
    new_weight[:old_weight.shape[0], :old_weight.shape[1], :] = old_weight

    # Initialize the new output channels (if any)
    if added_out_channels:
        torch.nn.init.normal_(new_weight[old_weight.shape[0]:, :, :], mean=0.0, std=0.01)

    # Initialize the new input channels (if any)
    if added_in_channels:
        torch.nn.init.normal_(new_weight[:, old_weight.shape[1]:, :], mean=0.0, std=0.01)

    # Copy and extend the bias if it exists
    if old_bias is not None:
        new_bias = torch.cat(
            [old_bias, torch.zeros(added_out_channels or 0).to(old_bias.device)]
        )
        new_conv1d.bias.data = new_bias

    return new_conv1d


def expand_embedding_layer(embedding_layer, added_dimensions, distribution='normal'):
    """
    Expands the embedding dimensions of a torch.nn.Embedding layer while retaining
    the original weights and initializing the new dimensions with random values or zeros.
    
    Args:
        embedding_layer (torch.nn.Embedding): Original embedding layer to expand.
        added_dimensions (int): Number of new dimensions to add to the embedding.
        distribution (str): Distribution to use for initializing new dimensions ('normal' or 'zeros').
    
    Returns:
        torch.nn.Embedding: New embedding layer with expanded dimensions.
    """
    # Get the original weights and parameters
    old_weight = embedding_layer.weight.data
    padding_idx = embedding_layer.padding_idx  # Keep the padding index

    # Create the new embedding layer with expanded dimensions
    new_num_embeddings = old_weight.shape[0]
    new_embedding_dim = old_weight.shape[1] + added_dimensions
    new_embedding_layer = torch.nn.Embedding(new_num_embeddings, new_embedding_dim, padding_idx=padding_idx)

    # Copy the old weights into the new embedding layer
    new_embedding_layer.weight.data[:, :old_weight.shape[1]] = old_weight

    # Initialize the new dimensions with random values or zeros
    if distribution == 'normal':
        torch.nn.init.normal_(new_embedding_layer.weight.data[:, old_weight.shape[1]:], mean=0.0, std=0.01)
    else:
        torch.nn.init.zeros_(new_embedding_layer.weight.data[:, old_weight.shape[1]:])

    # Ensure padding_idx row remains zero-initialized
    if padding_idx is not None:
        new_embedding_layer.weight.data[padding_idx] = 0

    return new_embedding_layer



def expand_linear_layer(linear_layer, added_in_features=None, added_out_features=None):
    """
    Expands the weight and bias dimensions of a torch.nn.Linear layer.

    Args:
        linear_layer (torch.nn.Linear): Original linear layer to expand.
        added_in_features (int or None): Number of new input features to add. If None, does not modify input features.
        added_out_features (int or None): Number of new output features to add. If None, does not modify output features.

    Returns:
        torch.nn.Linear: New linear layer with expanded dimensions.
    """
    added_in_features = added_in_features or 0
    added_out_features = added_out_features or 0

    # Get original dimensions
    old_in_features = linear_layer.weight.shape[1]
    old_out_features = linear_layer.weight.shape[0]
    old_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Calculate new dimensions
    new_in_features = old_in_features + added_in_features
    new_out_features = old_out_features + added_out_features

    # Create a new linear layer
    new_linear_layer = torch.nn.Linear(new_in_features, new_out_features, bias=linear_layer.bias is not None)

    # Copy old weights into the new layer's weights
    new_linear_layer.weight.data[:old_out_features, :old_in_features] = linear_layer.weight.data

    # Initialize new weights for added dimensions
    if added_out_features > 0:
        torch.nn.init.normal_(
            new_linear_layer.weight.data[old_out_features:, :old_in_features], mean=0.0, std=0.01
        )
    if added_in_features > 0:
        torch.nn.init.normal_(
            new_linear_layer.weight.data[:old_out_features, old_in_features:], mean=0.0, std=0.01
        )
    if added_out_features > 0 and added_in_features > 0:
        torch.nn.init.normal_(
            new_linear_layer.weight.data[old_out_features:, old_in_features:], mean=0.0, std=0.01
        )

    # Handle bias expansion
    if old_bias is not None:
        new_linear_layer.bias.data[:old_out_features] = old_bias
        if added_out_features > 0:
            torch.nn.init.zeros_(new_linear_layer.bias.data[old_out_features:])

    return new_linear_layer


def expand_layer_norm(layer_norm, added_dimensions=None):
    """
    Expands the dimensions of a torch.nn.LayerNorm layer.

    Args:
        layer_norm (torch.nn.LayerNorm): Original layer norm layer to expand.
        added_dimensions (int or None): Number of new dimensions to add. If None, no dimensions are added.

    Returns:
        torch.nn.LayerNorm: New layer norm with expanded dimensions.
    """
    # Get original dimensions
    old_num_features = layer_norm.normalized_shape[0]
    new_num_features = old_num_features + added_dimensions

    # Create a new LayerNorm layer
    new_layer_norm = torch.nn.LayerNorm(new_num_features, eps=layer_norm.eps, elementwise_affine=layer_norm.elementwise_affine)

    # Copy old weights and initialize new ones
    if layer_norm.elementwise_affine:
        old_weight = layer_norm.weight.data
        old_bias = layer_norm.bias.data if layer_norm.bias is not None else None

        # Initialize new weights
        new_layer_norm.weight.data[:old_num_features] = old_weight
        torch.nn.init.ones_(new_layer_norm.weight.data[old_num_features:])

        # Initialize new biases if applicable
        if old_bias is not None:
            new_layer_norm.bias.data[:old_num_features] = old_bias
            torch.nn.init.zeros_(new_layer_norm.bias.data[old_num_features:])

    return new_layer_norm


def expand_positional_embedding(positional_embedding_layer, added_dimensions):
    """
    Expands the embedding dimensions of a WhisperPositionalEmbedding layer while retaining
    the original weights and initializing the new dimensions with random values.
    
    Args:
        positional_embedding_layer (WhisperPositionalEmbedding): Original positional embedding layer.
        added_dimensions (int): Number of new dimensions to add to the embedding.
    
    Returns:
        WhisperPositionalEmbedding: New positional embedding layer with expanded dimensions.
    """
    # Get the original weights and properties
    old_weight = positional_embedding_layer.weight.data
    seq_len, old_dim = old_weight.shape
    new_dim = old_dim + added_dimensions

    # Create a new positional embedding layer of the same type
    new_positional_embedding_layer = type(positional_embedding_layer)(seq_len, new_dim)

    # Copy over the original weights
    new_positional_embedding_layer.weight.data[:, :old_dim] = old_weight

    # Initialize the new dimensions with random values
    torch.nn.init.normal_(new_positional_embedding_layer.weight.data[:, old_dim:], mean=0.0, std=0.01)

    return new_positional_embedding_layer
