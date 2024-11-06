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
        ).to(config.device)

        if config.emb_dim == 512:
            # Use SentenceTransformer library for:
            # - "sentence-transformers/distiluse-base-multilingual-cased-v1"
            self.sbert_model = SentenceTransformer(
                config.sbert_model_id,
                cache_folder=CACHE_DIR
            )

            self.linear = self.sbert_model[2].linear.to(config.device) if config.use_sbert_encoder else torch.nn.Linear(512, 512, True, config.device)
            self.activation = self.sbert_model[2].activation_function.to(config.device)

            if config.use_sbert_encoder:
                self.sbert_encoder = self.sbert_model[0].auto_model.transformer.to(config.device)

        else:
            # Use HuggingFace library for:
            # - "sentence-transformers/all-MiniLM-L12-v2"
            # - "sentence-transformers/all-mpnet-base-v2"
            self.sbert_model = AutoModel.from_pretrained(
                config.sbert_model_id,
                cache_dir=CACHE_DIR
            )

            self.linear = self.sbert_model.pooler.dense.to(config.device)
            self.activation = self.sbert_model.pooler.activation.to(config.device)

            if config.use_sbert_encoder:
                self.sbert_encoder = self.sbert_model.encoder.to(config.device)
        

    def forward(self, audio_inputs, text_input_ids, text_attention_mask):
        embs = self.whisper_model(
            audio_inputs,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask
        ).last_hidden_state
        
        if self.config.use_sbert_encoder:
            if self.config.emb_dim == 512:
                # Case: "sentence-transformers/distiluse-base-multilingual-cased-v1"
                embs = self.sbert_encoder(
                    embs,
                    attn_mask=text_attention_mask,
                    head_mask=[None] * self.sbert_model[0].auto_model.config.num_hidden_layers
                )[0]
            else:
                # Case: "sentence-transformers/all-MiniLM-L12-v2"
                # Case: "sentence-transformers/all-mpnet-base-v2"
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
                
        return embs
