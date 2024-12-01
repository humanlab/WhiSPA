import torch
import torch.nn.functional as F


def mean_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def last_pooling(embeddings, attention_mask):
    non_padding_indices = attention_mask.cumsum(dim=1) - 1
    last_non_padding_indices = non_padding_indices.gather(1, (attention_mask.sum(dim=1, keepdim=True) - 1).clamp(min=0).long())
    return embeddings[torch.arange(attention_mask.size(0)).unsqueeze(1), last_non_padding_indices].squeeze()


# Cosine Similarity Loss
def cos_sim_loss(whis_embs, sbert_embs):
    # z_audio = F.normalize(whis_embs, p=2, dim=1)
    # z_text = F.normalize(sbert_embs, p=2, dim=1)
    # return 1 - (z_audio * z_text).sum(dim=-1).mean()
    # Yields exactly the same result as torch.cosine_similarity()
    return 1 - torch.cosine_similarity(whis_embs, sbert_embs, dim=-1).mean()


# Cosine Similarity Contrastive Learning Loss
def sim_clr_loss(whis_embs, sbert_embs):
    z_audio = F.normalize(whis_embs, p=2, dim=1)
    z_text = F.normalize(sbert_embs, p=2, dim=1)
    similarity_matrix = torch.matmul(z_audio, z_text.T)

    positive_mask = torch.eye(whis_embs.shape[0], dtype=torch.float32).to(whis_embs.device)
    positive_loss = (positive_mask * similarity_matrix).sum()
    negative_loss = ((1 - positive_mask) * F.relu(1.0 - similarity_matrix)).sum()
    return positive_loss + negative_loss


# Normalized Temperature-Scaled Cross Entropy Loss
def nce_cont_loss(whis_embs, sbert_embs, tau=0.1, pooling_mode='sum'):
    """
        Helpful link I used for reference:
        https://jamesmccaffrey.wordpress.com/2022/04/11/an-example-of-normalized-temperature-scaled-cross-entropy-loss/
        
        Implemented from the paper:
        "A Simple Framework for Contrastive Learning of Visual Representations" (2020), Chen, et al.
    """
    combined = torch.cat([whis_embs, sbert_embs], dim=0)  # shape (2 * batch_size, emb_dims)
    combined = F.normalize(combined, dim=1)

    # Define positive pairs (each original data with its corresponding augmented data)
    batch_size = whis_embs.shape[0]
    pos_pairs = torch.arange(batch_size)
    
    # Compute cosine similarity for all pairs in the batch
    similarity_matrix = torch.matmul(combined, combined.T) / tau
    similarity_matrix = torch.exp(similarity_matrix)
    
    pos_sims = similarity_matrix[pos_pairs, pos_pairs + batch_size]
    neg_sims_sum = similarity_matrix[:batch_size].sum(dim=1) - torch.diag(similarity_matrix[:batch_size])
    
    losses = -torch.log(pos_sims / neg_sims_sum)
    return losses.sum() if pooling_mode == 'sum' else losses.mean()


# # Supervised Contrastive Loss
# def scl_loss(whis_embs, sbert_embs, labels, tau=0.10):
#     # This is a supervised contrastive loss objective which requires
#     # the labels to also be passed in for calculating loss
#     batch_size = len(whis_embs)

#     all_sims = torch.zeros((batch_size, batch_size), dtype=torch.float32)
#     for i in range(batch_size):
#         for j in range(batch_size):
#             all_sims[i,j] = torch.exp(torch.cosine_similarity(whis_embs[i,:], sbert_embs[j,:], dim=-1) / tau)
    
#     sum = 0.0
#     for i in range(batch_size):
#         # Get positive samples and negative samples
#         positives = []
#         negatives = []
#         for l in range(batch_size):
#             if not l == i:
#                 if labels[l] == labels[i]:
#                     positives.append(l)
#                 else:
#                     negatives.append(l)

#         inner_sum = 0.0
#         inv_cardinality = -1.0
#         # Use Supervised Contrastive Learning
#         if len(positives):
#             inv_cardinality /= len(positives)
#             for p in positives:
#                 negative_sum = 0.0
#                 for n in negatives:
#                     negative_sum += all_sims[i,n]
#                 inner_sum += torch.log(all_sims[i,p] / negative_sum)
#         # Use Self-Supervised Contrastive Learning
#         else:
#             inner_sum += torch.log(all_sims[i,i] / torch.sum(torch.cat([all_sims[i,:i], all_sims[i,i+1:]])))
#         sum += inv_cardinality * inner_sum
#     return sum
