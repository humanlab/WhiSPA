import torch
from transformers import AutoModel, WhisperModel

from config import CACHE_DIR


class WhiSBERTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.whisper_model = WhisperModel.from_pretrained(
            self.config.whisper_model_id,
            cache_dir=CACHE_DIR
        ).to(self.config.device)

        self.sbert_model = AutoModel.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2',
            cache_dir=CACHE_DIR
        )
        self.sbert_encoder = self.sbert_model.encoder.to(self.config.device)
        self.pooler = self.sbert_model.pooler.to(self.config.device)

    def forward(self, audio_inputs, text_input_ids, text_attention_mask):
        embs = self.whisper_model(
            audio_inputs,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask
        ).last_hidden_state
        
        if self.config.use_sbert_layers:
            embs = self.sbert_encoder(
                embs,
                attention_mask=self.sbert_model.get_extended_attention_mask(
                    text_attention_mask,
                    embs.size()[:-1]
                ),
                head_mask=[None] * self.sbert_model.config.num_hidden_layers
            )[0]
        
        if self.config.pooling_mode == 'cls':
            if self.config.use_sbert_layers:
                embs = embs[:, 0, :]
            else:
                # Use last non-padding token if not using SBERT layers
                non_padding_indices = text_attention_mask.cumsum(dim=1) - 1
                last_non_padding_indices = non_padding_indices.gather(1, (text_attention_mask.sum(dim=1, keepdim=True) - 1).clamp(min=0).long())
                embs = embs[torch.arange(text_attention_mask.size(0)).unsqueeze(1), last_non_padding_indices].squeeze()
        else:
            embs = mean_pooling(embs, text_attention_mask)

        return self.pooler.activation(self.pooler.dense(embs))


def mean_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
