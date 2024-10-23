import torch
import torch.nn.functional as F


def last_pooling(embeddings, attention_mask):
    non_padding_indices = attention_mask.cumsum(dim=1) - 1
    last_non_padding_indices = non_padding_indices.gather(1, (attention_mask.sum(dim=1, keepdim=True) - 1).clamp(min=0).long())
    return embeddings[torch.arange(attention_mask.size(0)).unsqueeze(1), last_non_padding_indices].squeeze()


def mean_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cos_sim(whis_embs, sbert_embs):
    # Normalize embeddings to unit vectors for cosine similarity
    z_audio = F.normalize(whis_embs, p=2, dim=1)
    z_text = F.normalize(sbert_embs, p=2, dim=1)

    # Calculate contrastive loss
    similarity_matrix = torch.matmul(z_audio, z_text.T)
    positive_mask = torch.eye(whis_embs.shape[0], dtype=torch.float32).to(whis_embs.device)
    positive_loss = (positive_mask * similarity_matrix).sum()
    negative_loss = ((1 - positive_mask) * F.relu(1.0 - similarity_matrix)).sum()
    return positive_loss + negative_loss


def sim_clr(whis_embs, sbert_embs):
    return 0