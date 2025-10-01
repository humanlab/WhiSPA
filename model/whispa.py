#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from typing import List
import copy
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from safetensors.torch import load_file
from transformers.models.voxtral.processing_voxtral import VoxtralProcessor
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from dotenv import load_dotenv

load_dotenv()

from model.config import WhiSPAConfig
from model.losses import mmrl_loss
from model.losses import trinary_alignment_loss


class WhiSPAProcessor:
    def __init__(self, config: WhiSPAConfig):
        self.config = config
        # self.processor = VoxtralProcessor.from_pretrained(config.backbone_model_id)
        # Resolve a local snapshot of the Voxtral backbone to ensure availability across processes
        local_dir = snapshot_download(repo_id=config.backbone_model_id)
        self.processor = VoxtralProcessor.from_pretrained(local_dir)

    def __call__(self, audio: str | List[str], language: str = "en"):
        if isinstance(audio, str):
            audio = [audio]

        return self.apply_transcription_request(audio, language)

    def apply_transcription_request(self, audio, language, **kwargs):
        """
        Efficient batched transcription prep that matches VoxtralProcessor output and
        augments it with per-sample chunk spans using audio-token counts in input_ids.

        Returns a BatchFeature (dict-like) with keys identical to Voxtral's output plus:
          - sample_spans: LongTensor [batch, 2] (start, end) indices along dim-0 of input_features
        """
        batch = self.processor.apply_transcription_request(
            audio=audio,
            language=language,
            model_id=self.config.backbone_model_id,
            **kwargs
        )

        # Derive per-sample chunk counts from number of audio tokens per row
        counts = (batch['input_ids'] == self.processor.audio_token_id).sum(dim=1).to(torch.long)
        if counts.numel() > 0:
            csum = torch.cumsum(counts, dim=0)
            starts = torch.cat([torch.zeros_like(counts[:1]), csum[:-1]], dim=0)
        else:
            starts = counts
        spans = torch.stack([starts, starts + counts], dim=1)

        return {
            'spectral_inputs': batch['input_features'],
            'sample_spans': spans,
            'text_input_ids': batch['input_ids'],
            'text_attention_mask': batch['attention_mask'],
        }


class WhiSPAModel(
    torch.nn.Module,
    PyTorchModelHubMixin,
    repo_url='whispa',
    pipeline_tag='feature-extraction',
    license='mit'
):
    """
    The WhiSPA model is a speech model that provides a unified embedding space rich in semantic, acoustic, and affective information.
    The model is trained in two stages:
    1. Training Stage 1: The spectral encoder and multimodal projector are jointly trained to align with semantic, acoustic, and affective embeddings.
    2. Training Stage 2: The text decoder language model is trained to transcribe the text from the spectral embeddings (while the other components are frozen).

    The model can be used in three modes:
    1. 'train_enc': Training Stage 1.
    2. 'train_dec': Training Stage 2.
    3. 'inference': The model can be used in inference mode to encode/decode audio files.
    """
    def __init__(self, config: WhiSPAConfig):
        super().__init__()
        self.config = config

        self.processor = WhiSPAProcessor(self.config)

        # Load Voxtral backbone
        local_dir = snapshot_download(repo_id=self.config.backbone_model_id)
        voxtral = VoxtralForConditionalGeneration.from_pretrained(local_dir)
        self.voxtral_config = voxtral.config
        
        # Decomposed modules
        self.spectral_encoder = copy.deepcopy(voxtral.audio_tower).to(dtype=config.dtype, device=config.device)
        self.multi_modal_projector = copy.deepcopy(voxtral.multi_modal_projector).to(dtype=config.dtype, device=config.device)
        self.language_model = copy.deepcopy(voxtral.language_model).to(dtype=config.dtype, device=config.device)
        self.activation = torch.nn.Tanh().to(config.device)

        del voxtral

        # Set Loss functions
        if self.config.loss == "MMRL":
            self.loss_fn = mmrl_loss
        elif self.config.loss == "TAL":
            self.loss_fn = trinary_alignment_loss
        else:
            raise ValueError(f"Invalid loss function: {self.config.loss}")
        
        self.set_stage(config.stage)


    def set_stage(self, stage: str):
        if stage in {'train_enc', 'train_dec', 'inference'}:
            self.config.stage = stage
            if stage == 'train_enc': # Training Stage 1
                self._unfreeze_spectral_encoder()
                self._unfreeze_projector()
                self._freeze_language_model()
            elif stage == 'train_dec': # Training Stage 2
                self._freeze_spectral_encoder()
                self._freeze_projector()
                self._unfreeze_language_model()
            else: # Inference
                self._freeze_spectral_encoder()
                self._freeze_projector()
                self._freeze_language_model()
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be one of ['train_enc', 'train_dec', 'inference'].")


    def get_stage(self):
        return self.config.stage


    def _freeze_spectral_encoder(self):
        for param in self.spectral_encoder.parameters():
            param.requires_grad = False


    def _unfreeze_spectral_encoder(self):
        for name, param in self.spectral_encoder.named_parameters():
            if name.startswith('embed_positions.'):
                param.requires_grad = False
            else:
                param.requires_grad = True


    def _freeze_language_model(self):
        for param in self.language_model.parameters():
            param.requires_grad = False

    
    def _unfreeze_language_model(self):
        for param in self.language_model.parameters():
            param.requires_grad = True


    def _freeze_projector(self):
        for param in self.multi_modal_projector.parameters():
            param.requires_grad = False


    def _unfreeze_projector(self):
        for param in self.multi_modal_projector.parameters():
            param.requires_grad = True


    def compute_per_sample_means(self, spectral_latent: torch.Tensor, sample_spans: torch.Tensor) -> torch.Tensor:
        """
        Reduce variable-length chunk sequences per sample to mean embeddings using spans.

        Args:
            spectral_latent: [N_total_chunks, hidden_size]
            sample_spans: [batch_size, 2] start/end indices into spectral_latent
        Returns:
            [batch_size, hidden_size] mean embeddings per sample
        """
        batch_size = sample_spans.size(0)
        device = spectral_latent.device
        counts = (sample_spans[:, 1] - sample_spans[:, 0]).to(dtype=torch.long, device=device)
        segment_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), counts)
        sums = torch.zeros((batch_size, spectral_latent.size(1)), device=device, dtype=spectral_latent.dtype)
        sums.index_add_(0, segment_ids, spectral_latent)
        means = sums / counts.clamp_min(1).unsqueeze(1)
        return means


    def get_spectral_embeddings(self, spectral_inputs: torch.Tensor):
        spectral_latent = self.spectral_encoder(spectral_inputs).last_hidden_state # [N, max_source_positions, audio_config.hidden_size]
        spectral_latent = spectral_latent.reshape(-1, self.voxtral_config.audio_config.intermediate_size) # [375 * N, audio_config.intermediate_size]
        spectral_embs = self.multi_modal_projector(spectral_latent) # [375 * N, hidden_size]
        return spectral_embs

    
    def get_text_embeddings(self, text_input_ids: torch.Tensor, spectral_latent: torch.Tensor):
        inputs_embs = self.language_model.get_input_embeddings()(text_input_ids)        
        audio_token_mask = text_input_ids == self.voxtral_config.audio_token_id
        inputs_embs[audio_token_mask] = spectral_latent
        return inputs_embs


    def forward(
        self,
        spectral_inputs: torch.FloatTensor,
        sample_spans: torch.LongTensor,
        target_audio_embs: torch.FloatTensor = None,
        target_text_embs: torch.FloatTensor = None,
        target_psych_embs: torch.FloatTensor = None,
        text_input_ids: torch.LongTensor = None,
        text_attention_mask: torch.LongTensor = None,
        text_labels: torch.LongTensor = None,
    ):
        """
        This method is the forward pass of the WhiSPA model.

        Inputs:
            spectral_inputs: Tensor of shape [batch_size, mel_bins, time_steps] - log-mel spectrogram passed to the Voxtral encoder.
            sample_spans: Tensor of shape [batch_size, 2] - start/end indices into spectral_inputs.
            target_audio_embs: Tensor of shape [batch_size, hidden_size] - target acoustic embeddings to align with the spectral embeddings.
            target_text_embs: Tensor of shape [batch_size, hidden_size] - target text embeddings to align with the spectral embeddings.
            target_psych_embs: Tensor of shape [batch_size, hidden_size] - target psychological embeddings to align with the spectral embeddings.
            text_input_ids: Tensor of shape [batch_size, num_tokens] - input text tokens passed to the text decoder.
            text_attention_mask: Tensor of shape [batch_size, num_tokens] - text attention mask passed to the text decoder.
            text_labels: Tensor of shape [batch_size, num_tokens] - transcription text tokens passed to the text decoder.
        Outputs:
            spectral_latent: Output embedding of the spectral encoder.
            total_loss: Total loss
            semantic_loss: Loss for the text modality
            acoustic_loss: Loss for the audio modality
        """
        spectral_latent = self.get_spectral_embeddings(spectral_inputs.to(self.spectral_encoder.device)) # [375 * N, hidden_size]

        if self.config.stage == 'train_enc': # Training Stage 1
            """
            This stage is the training stage for the spectral encoder and multimodal projector.
            """
            assert isinstance(target_audio_embs, torch.Tensor), f"Input: `target_audio_embs` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            assert isinstance(target_text_embs, torch.Tensor), f"Input: `target_text_embs` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"

            # Compute per-sample mean embeddings
            spectral_embs = self.compute_per_sample_means(spectral_latent, sample_spans) # [B, D]
            spectral_embs = self.activation(spectral_embs) # [B, D]

            spectral_embs = F.normalize(spectral_embs, p=2, dim=1) # [B, D]
            target_audio_embs = F.normalize(target_audio_embs, p=2, dim=1) # [B, D]
            target_text_embs = F.normalize(target_text_embs, p=2, dim=1) # [B, D]
            target_psych_embs = F.normalize(target_psych_embs, p=2, dim=1) # [B, D]

            # Compute orthogonality penalty
            # ortho_penalty = torch.norm(target_audio_embs.T @ target_text_embs, p='fro')**2

            total_loss, acoustic_loss, semantic_loss, affective_loss = self.loss_fn(spectral_embs, target_audio_embs, target_text_embs, target_psych_embs)

            return {
                "spectral_embs": spectral_embs,
                "total_loss": total_loss,
                "acoustic_loss": acoustic_loss,
                "semantic_loss": semantic_loss,
                "affective_loss": affective_loss
            }
        
        elif self.config.stage == 'train_dec': # Training Stage 2
            """
            This stage is the training stage for the text decoder language model.
            """
            assert isinstance(text_input_ids, torch.Tensor), f"Input: `text_input_ids` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            assert isinstance(text_attention_mask, torch.Tensor), f"Input: `text_attention_mask` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            assert isinstance(text_labels, torch.Tensor), f"Input: `text_labels` should be a torch.Tensor. This is a required input for stage: {self.config.stage}"
            
            text_input_ids = text_input_ids.to(self.config.device)
            text_attention_mask = text_attention_mask.to(self.config.device)
            text_labels = text_labels.to(self.config.device)

            inputs_embs = self.get_text_embeddings(text_input_ids, spectral_latent)

            # Forward-pass through language model
            outputs = self.language_model(
                attention_mask=text_attention_mask,
                inputs_embeds=inputs_embs,
                labels=text_labels,
                use_cache=False, # False for training
            )
            return outputs

        elif self.config.stage == 'inference': # Inference (Encoding)
            """
            This stage is the inference stage for encoding the input audio spectrogram into an embedding representation.
            """
            # Compute per-sample mean embeddings
            spectral_embs = self.compute_per_sample_means(spectral_latent, sample_spans) # [B, D]
            spectral_embs = self.activation(spectral_embs) # [B, D]
            return F.normalize(spectral_embs, p=2, dim=1) # [B, D]

        else:
            raise NotImplementedError(f"The stage: {self.config.stage} is unknown. The options are [`train_enc`, `train_dec`, `inference`]")


    def encode(self, audio: str | List[str], language: str = "en"):
        """
        Encode audio into an embedding representation.
        """
        # Ensure the model config is set to "inference"
        self.config.stage = "inference"

        inputs = self.processor(audio, language=language)
        
        with torch.no_grad():
            return self.forward(**inputs)


    def transcribe(self, audio: str | List[str], language: str = "en", **kwargs):
        """
        Transcribe audio using Voxtral end-to-end.
        
        Args:
            audio_path: path to audio file
            language: language code for transcription request
            **kwargs: generation kwargs (e.g., max_new_tokens, do_sample, num_beams)
        Returns:
            List[str]: transcriptions
        """
        # Ensure inference stage
        self.config.stage = "inference"

        # Prepare raw inputs expected by VoxtralForConditionalGeneration
        inputs = self.processor.processor.apply_transcription_request(
            language=language,
            audio=audio,
            model_id=self.config.backbone_model_id,
        )
        for k, v in inputs.items():
            if v.dtype.is_floating_point:
                inputs[k] = v.to(self.config.device, dtype=self.config.dtype)
            else:
                inputs[k] = v.to(self.config.device)

        # Generate
        with torch.no_grad():
            outputs = self.voxtral_config.generate(**inputs, **kwargs)

        # Decode only continuation past the prompt
        continuations = outputs[:, inputs["input_ids"].shape[1]:]
        return self.processor.processor.batch_decode(continuations, skip_special_tokens=True)


    def save_pretrained(self, local_dir: str, safe_serialization: bool = True):
        """
        Save model locally in HuggingFace format without uploading.

        - Writes `config.json` via `WhiSPAConfig.save_pretrained`
        - Writes weights to `model.safetensors` if safe_serialization else `pytorch_model.bin`
        """
        os.makedirs(local_dir, exist_ok=True)
        # Save config
        self.config.save_pretrained(local_dir)
        
        # Save weights
        full_state = self.state_dict()
        weights_path = os.path.join(local_dir, "model.safetensors" if safe_serialization else "pytorch_model.bin")

        if safe_serialization:
            # Save as .safetensors
            save_file(full_state, weights_path)
        else:
            # Save as legacy PyTorch
            torch.save(full_state, weights_path)


    @classmethod
    def _from_pretrained(cls, local_dir, **kwargs):
        config = WhiSPAConfig.from_pretrained(local_dir)

        model = cls(config)
        
        # Try different weight loading strategies in order of preference
        state_dict = None
        
        # 1. Try single model.safetensors file
        safetensors_path = os.path.join(local_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path, device="cpu")
        
        # 2. Try sharded safetensors with index
        elif os.path.exists(os.path.join(local_dir, "model.safetensors.index.json")):
            import json
            with open(os.path.join(local_dir, "model.safetensors.index.json"), "r") as f:
                index = json.load(f)
            
            # Load all shards
            state_dict = {}
            weight_map = index.get("weight_map", {})
            loaded_shards = set()
            
            for param_name, shard_file in weight_map.items():
                if shard_file not in loaded_shards:
                    shard_path = os.path.join(local_dir, shard_file)
                    if os.path.exists(shard_path):
                        shard_state = load_file(shard_path, device="cpu")
                        state_dict.update(shard_state)
                        loaded_shards.add(shard_file)
        
        # 3. Try pytorch_model_fsdp.bin (from FSDP training)
        elif os.path.exists(os.path.join(local_dir, "pytorch_model_fsdp.bin")):
            state_dict = torch.load(os.path.join(local_dir, "pytorch_model_fsdp.bin"), map_location="cpu")
        
        # 4. Try pytorch_model.bin
        elif os.path.exists(os.path.join(local_dir, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
        
        else:
            raise FileNotFoundError(
                f"No model weights found in {local_dir}. "
                f"Looked for: model.safetensors, model.safetensors.index.json (sharded), "
                f"pytorch_model_fsdp.bin, pytorch_model.bin"
            )
        
        model.load_state_dict(state_dict, strict=False)
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

    
    @classmethod
    def from_pretrained_local(cls, local_dir: str):
        """Load a locally saved WhiSPAModel from `save_pretrained_local`."""
        return cls._from_pretrained(local_dir=local_dir)


"""
@deprecated("This class is deprecated and will be removed in a future version.")
"""
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
          

"""
@deprecated("This class is deprecated and will be removed in a future version.")
"""
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
