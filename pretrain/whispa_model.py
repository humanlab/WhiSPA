import os
import torch
import torch.nn.functional as F
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    GenerationConfig
)
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from dotenv import load_dotenv

load_dotenv()

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_data import preprocess_audio
from pretrain.whispa_utils import (
    nce_loss,
    spectral_recon_loss
)


class WhiSPAProcessor:
    def __init__(self, config: WhiSPAConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(config.whisper_model_id)

    def __call__(self, audio_path):
        waveform = preprocess_audio(audio_path)
        processed = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors='pt')
        return processed.input_features # [B, T, D]


class WhiSPAModel(
    torch.nn.Module,
    PyTorchModelHubMixin,
    repo_url='whispa',
    pipeline_tag='feature-extraction',
    license='mit'
):
    """
    The WhiSPA model is a speech model that provides a unified embedding space rich in semantic and acoustic information.
    The model is trained in two stages:
    1. Pre-Training Stage 1: The spectral encoder (WhisperEncoder) is trained to align with semantic and acoustic embeddings.
    2. Pre-Training Stage 2: The text decoder (WhisperDecoder) is trained to transcribe the text from the spectral embeddings (while the spectral encoder is frozen).

    The model can be used in three modes:
    1. 'encode': The model is used to encode the input audio spectrogram into an embedding representation.
    2. 'decode': The model is used to autoregressively decode text provided the input audio spectrogram.
    3. 'train_enc': Pre-Training Stage 1.
    4. 'train_dec': Pre-Training Stage 2.
    """
    def __init__(self, config: WhiSPAConfig):
        super().__init__()
        self.config = config

        self.processor = WhiSPAProcessor(self.config)

        whisper = WhisperForConditionalGeneration.from_pretrained(self.config.whisper_model_id)
        self.whisper_config = whisper.model.config
        self.config.vocab_size = self.whisper_config.vocab_size

        self.spectral_encoder = whisper.model.encoder.to(dtype=config.dtype, device=config.device)

        if config.stage == 'train_enc': # Pre-Training Stage 1
            # self.spectral_decoder = WhiSPASpectralDecoder(config).to(dtype=config.dtype, device=config.device)
            self.gating_network = WhiSPAGatingNetwork(dtype=config.dtype, device=config.device)

        if config.stage == 'train_dec': # Pre-Training Stage 2
            self.text_decoder = whisper.model.decoder.to(dtype=config.dtype, device=config.device)
            self.vocab_proj = whisper.proj_out.to(dtype=config.dtype, device=config.device)
            self._freeze_spectral_encoder()

        elif config.stage == 'encode': # Inference (Encoding)
            self._freeze_spectral_encoder()

        elif config.stage == 'decode': # Inference (Decoding)
            self.text_decoder = whisper.model.decoder.to(dtype=config.dtype, device=config.device)
            self.vocab_proj = whisper.proj_out.to(dtype=config.dtype, device=config.device)
            self._freeze_spectral_encoder()
            self._freeze_text_decoder()

        self.activation = torch.nn.Tanh().to(config.device)


    def _freeze_spectral_encoder(self):
        for param in self.spectral_encoder.parameters():
            param.requires_grad = False


    def _freeze_text_decoder(self):
        for param in self.text_decoder.parameters():
            param.requires_grad = False
        
        for param in self.vocab_proj.parameters():
            param.requires_grad = False


    def forward(
        self,
        spectral_inputs: torch.Tensor,
        acoustic_embeddings: torch.Tensor = None,
        text_embeddings: torch.Tensor = None,
        text_labels: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
    ):
        """
        This method is the forward pass of the WhiSPA model.

        Inputs:
            spectral_inputs: Tensor of shape [B, T, D] - log-mel spectrogram passed to the Whisper encoder.
            acoustic_embeddings: Tensor of shape [B, D] - target acoustic embedding to align with the spectral embeddings.
            text_embeddings: Tensor of shape [B, D] - target text embedding to align with the spectral embeddings.
            text_labels: Tensor of shape [B, T] - transcription text tokens passed to the Whisper decoder.
            text_attention_mask: Tensor of shape [B, T] - text attention mask passed to the Whisper decoder.
        Outputs:
            spectral_latent: Output embedding of the spectral encoder.
            total_loss: Total loss
            semantic_loss: Loss for the text modality
            acoustic_loss: Loss for the audio modality
            gate_weight: Gated weight for the acoustic modality
            ortho_penalty: Orthogonality penalty
        """
        spectral_latent = self.spectral_encoder(spectral_inputs).last_hidden_state # [B, T', D]

        if self.config.stage == 'train_enc': # Pre-Training Stage 1
            assert isinstance(acoustic_embeddings, torch.Tensor), f"Input: `acoustic_embeddings` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            assert isinstance(text_embeddings, torch.Tensor), f"Input: `text_embeddings` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"

            spectral_embedding = self.activation(spectral_latent.mean(1)) # [B, D]
            spectral_embedding = F.normalize(spectral_embedding, p=2, dim=1) # [B, D]
            acoustic_embeddings = F.normalize(acoustic_embeddings, p=2, dim=1) # [B, D]
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1) # [B, D]

            semantic_loss = nce_loss(spectral_embedding, text_embeddings)
            acoustic_loss = nce_loss(spectral_embedding, acoustic_embeddings)

            # Compute gated weights
            mod_stats = torch.tensor(
                [
                    F.cosine_similarity(acoustic_embeddings, text_embeddings).mean(),
                    torch.var(acoustic_embeddings, 1).mean(),
                    torch.var(text_embeddings, 1).mean()
                ],
                dtype=self.gating_network.dtype,
                device=self.gating_network.device
            )
            gate_weight = self.gating_network(mod_stats).unsqueeze(0)

            # Compute orthogonality penalty
            ortho_penalty = torch.norm(acoustic_embeddings.T @ text_embeddings, p='fro')**2

            total_loss = gate_weight * semantic_loss + (1 - gate_weight) * acoustic_loss + ortho_penalty

            return spectral_embedding, total_loss, semantic_loss, acoustic_loss, gate_weight, ortho_penalty
        
        elif self.config.stage == 'train_dec': # Pre-Training Stage 2
            assert isinstance(text_labels, torch.Tensor), f"Input: `text_labels` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            assert isinstance(text_attention_mask, torch.Tensor), f"Input: `text_attention_mask` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"

            # Prepare decoder input IDs by shifting labels right
            decoder_input_ids = shift_tokens_right(
                text_labels, 
                self.whisper_config.pad_token_id, 
                self.whisper_config.decoder_start_token_id
            )

            # Call the text decoder
            decoder_outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                attention_mask=text_attention_mask,
                encoder_hidden_states=spectral_latent,
                use_cache=False,  # No caching during training
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            # Project decoder outputs to vocabulary
            lm_logits = self.vocab_proj(decoder_outputs.last_hidden_state)

            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            # Move labels to correct device
            text_labels = text_labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.whisper_config.vocab_size), text_labels.reshape(-1))

            return spectral_latent, loss, lm_logits

        elif self.config.stage == 'encode': # Inference (Encoding)
            """
            This stage is the inference stage for encoding the input audio spectrogram into an embedding representation.
            """
            spectral_embedding = self.activation(spectral_latent.mean(1)) # [B, D]
            return F.normalize(spectral_embedding, p=2, dim=1)
        
        elif self.config.stage == 'decode': # Inference (Decoding)
            """
            This stage is the inference stage for autoregressively decoding text provided the input audio spectrogram.
            """
            raise NotImplementedError(f"Autoregressive decoding for inference must be called using the `transcribe()` method, not `forward()`")

        else:
            raise NotImplementedError(f"The stage: {self.config.stage} is unknown. The options are [`train_enc`, `train_dec`, `encode`, `decode`]")


    def encode(self, audio_path):
        """
        Encode audio into an embedding representation.
        """
        # Ensure the model config is set to "encode"
        self.config.stage = "encode"
        return self.forward(self.processor(audio_path))


    def generate(self, spectral_inputs, **kwargs):
        """
        Generate text from audio spectrogram inputs using sophisticated decoding.
        
        Args:
            spectral_inputs: Tensor of shape [B, T, D] - log-mel spectrogram
            **kwargs: Additional arguments passed to the generation method
            
        Returns:
            Generated token IDs
        """
        if self.config.stage not in ['decode', 'train_dec']:
            raise ValueError(f"Generation is only supported in 'decode' or 'train_dec' stages, not '{self.config.stage}'")
        
        # Load generation model (no weights)
        generation_model = WhisperForConditionalGeneration(self.whisper_config)
        generation_model.generation_config = GenerationConfig.from_pretrained(self.config.whisper_model_id)
        
        # Set our components (this uses our tunable weights)
        generation_model.model.encoder = self.spectral_encoder
        generation_model.model.decoder = self.text_decoder
        generation_model.proj_out = self.vocab_proj
        
        # Generate using our weights
        with torch.no_grad():
            generated_ids = generation_model.generate(
                input_features=spectral_inputs,
                **kwargs
            )
        
        return generated_ids


    def transcribe(self, audio_path, **kwargs):
        """
        Transcribe audio spectrogram inputs to text.
        
        Args:
            spectral_inputs: Tensor of shape [B, T, D] - log-mel spectrogram
            **kwargs: Additional arguments passed to the generation method
            
        Returns:
            List of transcribed text strings
        """
        # Ensure the model config is set to "decode"
        self.config.stage = "decode"

        # Generate token IDs
        spectral_inputs = self.processor(audio_path)
        generated_ids = self.generate(spectral_inputs, **kwargs)
        
        # Decode tokens to text
        transcriptions = self.processor.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcriptions
    

    def _save_pretrained(self, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        self.config.save_pretrained(local_dir)
        torch.save(self.state_dict(), os.path.join(local_dir, 'pytorch_model.bin'))


    @classmethod
    def _from_pretrained(cls, local_dir, **kwargs):
        print("**Inside _from_pretrained** with local_dir=", local_dir)
        config = WhiSPAConfig.from_pretrained(local_dir)

        model = cls(config)
        state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)

        return model
    

    @classmethod
    def from_pretrained(cls, model_id, revision=None, cache_dir=None, force_download=False, **hub_kwargs):
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            **hub_kwargs
        )
        return cls._from_pretrained(local_dir=local_dir)


class WhiSPAGatingNetwork(torch.nn.Module):
    """
    Gated Weight (α) via Acoustic-Semantic Correlation Estimator
    --------------------------------------------------------------
    The gating network dynamically adjusts the contribution of acoustic (L_{Z,A}) and 
    semantic (L_{Z,L}) losses based on the aggregated statistics of each modality.
    Inputs:
        - mod_sim: Cosine similarity between acoustic and semantic embeddings
        - var_acoustic: Variance of the acoustic embeddings
        - var_semantic: Variance of the semantic embeddings
    Outputs:
        - α ∈ (0,1): Gated weight for acoustic loss
    --------------------------------------------------------------
    """
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.fc1 = torch.nn.Linear(3, 16).to(dtype).to(device)
        self.fc2 = torch.nn.Linear(16, 1).to(dtype).to(device)
        self.relu = torch.nn.ReLU().to(device)
        

    def forward(self, x):
        # Input: x [mod_sim, var_acoustic, var_semantic]
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)
    

class WhiSPASpectralDecoder(torch.nn.Module):
    """
        Originally intended to be used for reconstructing the log-mel spectrogram from the latent space of the Whisper encoder.
        However, it is not currently used in the training pipeline.
    """
    def __init__(self, config: WhiSPAConfig):
        super().__init__()
        self.config = config

        self.conv1 = torch.nn.ConvTranspose1d(config.hidden_size, 3 * config.n_mel_bins, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose1d(3 * config.n_mel_bins, config.n_mel_bins, kernel_size=3, stride=1, padding=1)
        self.gelu = torch.nn.GELU()

        expected_seq_length = config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        self.pos_embed = torch.nn.Parameter(torch.randn(1, expected_seq_length, config.hidden_size))

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model = config.hidden_size,
            nhead = config.spectral_decoder_n_heads,
            dim_feedforward = config.spectral_decoder_ffn_dim,
            batch_first = True
        )
        self.decoder_layers = torch.nn.TransformerDecoder(decoder_layer, num_layers=config.spectral_decoder_n_layers)
        
        self.in_proj = torch.nn.Linear(config.n_mel_bins, config.hidden_size)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.n_mel_bins)


    def forward(self, latent_space):
        """
        latent_space: Tensor of shape [B, T', D] (from Whisper's Encoder)
        Returns: Reconstructed log-mel (80 bins) spectrogram [B, 80, T]
        """
        # Upsampling (Deconvolution)
        x = self.gelu(self.conv1(latent_space.transpose(1, 2))) # [B, 3*80, T]
        x = self.gelu(self.conv2(x)).transpose(1, 2) # [B, T, 80]

        decoder_in = self.in_proj(x) + self.pos_embed[:, :x.shape[1], :] # [B, T, D]
        decoder_out = self.decoder_layers(tgt=decoder_in, memory=latent_space) # [B, T, D]
        return self.out_proj(decoder_out).transpose(1, 2) # [B, 80, T]
