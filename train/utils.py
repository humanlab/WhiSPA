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
def cs_loss(audio_embs, text_embs):
    # z_audio = F.normalize(audio_embs, p=2, dim=1)
    # z_text = F.normalize(text_embs, p=2, dim=1)
    # return 1 - (z_audio * z_text).sum(dim=-1).mean()
    # Yields exactly the same result as torch.cosine_similarity()
    return 1 - torch.cosine_similarity(audio_embs, text_embs, dim=-1).mean()


# Cosine Similarity Contrastive Learning Loss
def sim_clr_loss(audio_embs, text_embs):
    z_audio = F.normalize(audio_embs, p=2, dim=1)
    z_text = F.normalize(text_embs, p=2, dim=1)
    similarity_matrix = torch.matmul(z_audio, z_text.T)

    positive_mask = torch.eye(audio_embs.shape[0], dtype=torch.float32).to(audio_embs.device)
    positive_loss = (positive_mask * similarity_matrix).sum()
    negative_loss = ((1 - positive_mask) * F.relu(1.0 - similarity_matrix)).sum()
    return positive_loss + negative_loss


# Noise Contrastive Estimation Loss
def nce_loss(z_a, z_b, 𝜏=0.1, pooling_mode='sum'):
    """
        Helpful link I used for reference:
        https://jamesmccaffrey.wordpress.com/2022/04/11/an-example-of-normalized-temperature-scaled-cross-entropy-loss/
        
        Implemented from the paper:
        "A Simple Framework for Contrastive Learning of Visual Representations" (2020), Chen, et al.
    """
    combined = torch.cat([z_a, z_b], dim=0)  # shape (2 * batch_size, emb_dims)
    combined = F.normalize(combined, dim=1)

    # Define positive pairs (each original data with its corresponding augmented data)
    batch_size = z_a.shape[0]
    pos_pairs = torch.arange(batch_size)
    
    # Compute cosine similarity for all pairs in the batch
    similarity_matrix = torch.matmul(combined, combined.T) / 𝜏
    similarity_matrix = torch.exp(similarity_matrix)
    
    pos_sims = similarity_matrix[pos_pairs, pos_pairs + batch_size]
    neg_sims_sum = similarity_matrix[:batch_size].sum(dim=1) - torch.diag(similarity_matrix[:batch_size])
    
    losses = -torch.log(pos_sims / neg_sims_sum)
    return losses.sum() if pooling_mode == 'sum' else losses.mean()


def dwd_loss(
    gating_net,
    whispa_embs,
    linguistic_embs,
    acoustic_embs,
    psych_embs=None,
    λ=0.1,
    𝜏=0.1
):
    """
        Dynamically Weighted Distillation with Modality-Specific Gates
        ------------------------------
        L_DWD = α⋅Contrastive(Z, A) + (1−α)⋅Contrastive(Z, L) + λ⋅OrthoPenalty(A, L)
        α: Gated weight from acoustic-textual correlation estimator
        OrthoPenalty: Penalizes redundancy between A (acoustic) and L (linguistic) subspaces
        ------------------------------
    """
    # Normalize all embeddings
    Z = F.normalize(whispa_embs, dim=-1)
    A = F.normalize(acoustic_embs, dim=-1)
    L = F.normalize(linguistic_embs, dim=-1)

    if psych_embs is None:
        acoustic_loss = nce_loss(Z, A, 𝜏)
        linguistic_loss = nce_loss(Z, L, 𝜏)
        ortho = torch.norm(A.T @ L, p='fro')**2

        # Compute gating network inputs
        with torch.no_grad():
            mod_sim = F.cosine_similarity(A, L).mean()
            var_acoustic = torch.var(A)
            var_linguistic = torch.var(L)
        
        gate_inputs = torch.tensor(
            [mod_sim, var_acoustic, var_linguistic],
            dtype=gating_net.module.dtype if isinstance(gating_net, torch.nn.parallel.DistributedDataParallel) \
                else gating_net.dtype,
            device=gating_net.module.device if isinstance(gating_net, torch.nn.parallel.DistributedDataParallel) \
                else gating_net.device
        ).unsqueeze(0)
        α = gating_net(gate_inputs)

        # Final loss
        total_loss = (α * acoustic_loss + (1-α) * linguistic_loss + λ * ortho)
        return total_loss, α, acoustic_loss, linguistic_loss, ortho
    else:
        raise Exception('Not Implemented!')
        

def spectral_recon_loss(pred, target, alpha=1.0, beta=1.0):
    """
    Robust spectral reconstruction loss:
    - log-cosh frame loss
    - spectral convergence (linear magnitude)

    Args:
        pred: [B, 80, T] - predicted log-mel
        target: [B, 80, T] - ground truth log-mel

    Returns:
        Scalar loss
    """
    def log_cosh_loss(pred, target, eps=1e-6):
        return torch.mean(torch.log(torch.cosh(pred - target + eps)))

    def spectral_convergence_loss(pred_mag, target_mag):
        return torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-9)

    # Log-Cosh in log-mel domain
    logcosh = log_cosh_loss(pred, target)
    # Spectral convergence in linear scale
    spec_conv = spectral_convergence_loss(torch.exp(pred), torch.exp(target))

    return alpha * logcosh + beta * spec_conv
