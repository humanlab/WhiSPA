#!/usr/bin/env python3

import torch
import torch.nn.functional as F


# Cosine Similarity Loss
def cs_loss(audio_embs, text_embs):
    # z_audio = F.normalize(audio_embs, p=2, dim=-1)
    # z_text = F.normalize(text_embs, p=2, dim=-1)
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


# Noise Contrastive Estimation (NCE)
def nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, 𝜏: float = 0.1) -> torch.Tensor:
    """
        Helpful link for reference:
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
    return losses.sum()


# Matryoshka Representation Learning (MRL)
def mrl_loss(
    z_a: torch.Tensor, # [N, D]
    z_b: torch.Tensor, # [N, D]
    K: int = 5,
    𝜏: float = 0.1,
) -> torch.Tensor:
    """
    Efficient MRL loss using masked prefixes (no per-k slicing in Python loops).

    Args:
        z_a, z_b: float tensors of shape [N, D], unnormalized embeddings.
        K: number of Matryoshka prefixes
        𝜏: temperature for InfoNCE.

    Returns:
        Scalar tensor: sum of InfoNCE losses across all prefixes.
    """
    def _create_nesting_weights(
        D: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str = "inv_sqrt",
        alpha: float = 1.0,
        cap_ratio: float = None,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Generate Matryoshka nesting levels and their associated weights for MRL loss.

        Args:
            D (int): The embedding dimensionality (full dimension).
            K (int): Number of nesting levels (prefixes) to generate.
            device (torch.device): Device to place the resulting tensor on.
            dtype (torch.dtype): Data type for the weights tensor.
            mode (str, optional): Weighting mode. One of {"inv_sqrt", "inv", "exp"}.
                - "inv_sqrt": weights = (D / k) ** (0.5 * alpha)
                - "inv":      weights = (D / k) ** (1.0 * alpha)
                - "exp":      weights = exp(-alpha * k / D)
            alpha (float, optional): Exponent or scaling factor for weighting. Default: 1.0.
            cap_ratio (float, optional): If set, clamps minimum weight to (max_weight / cap_ratio)
                to avoid extreme ratios. Default: None (no capping).

        Returns:
            nesting (list[int]): List of prefix lengths for each nesting level, e.g. [1024, 512, 256, ...].
            weights (torch.Tensor): 1D tensor of shape [len(nesting)] with normalized weights (mean=1).

        Example:
            >>> nesting, weights = _create_nesting_weights(1024, 5, torch.device('cpu'), torch.float32)
            >>> print(nesting)  # [1024, 512, 256, 128, 64]
            >>> print(weights)  # tensor of shape [5], normalized so mean == 1
        """
        # Create nesting levels [D, D//2, D//4, D//8, D//16]
        nesting = [D]
        for _ in range(K - 1):
            k = nesting[-1] // 2
            if k < 1:
                break
            nesting.append(k)
        
        # Create weights using selected mode
        weights = torch.tensor(nesting, device=device, dtype=dtype)
        if mode == "inv_sqrt":
            weights = (D / weights).pow(0.5 * alpha)
        elif mode == "inv":
            weights = (D / weights).pow(1.0 * alpha)
        elif mode == "exp":
            weights = torch.exp(-alpha * weights / D)
        else:
            raise ValueError(f"Invalid MRL weighting mode: {mode}")

        # Optional: cap extreme ratios
        if cap_ratio is not None and cap_ratio > 0:
            weights = weights.clamp(min=weights.max() / cap_ratio)

        # Normalize so mean weight == 1 (keeps overall loss scale steady)
        weights = weights * (len(nesting) / weights.sum())

        return nesting, weights


    device, dtype = z_a.device, z_a.dtype
    nesting, weights = _create_nesting_weights(z_a.shape[-1], K, device, dtype)

    # Build a [K, D] binary mask where row i has ones in [0:k_i)
    mask = torch.zeros(K, z_a.shape[-1], device=device, dtype=dtype)
    for i, k in enumerate(nesting):
        mask[i, :k] = 1

    # Masked prefixes in one shot (truncated prefixes): [K, N, D]
    a_masked = z_a.unsqueeze(0) * mask[:, None, :] # [K, N, D]
    b_masked = z_b.unsqueeze(0) * mask[:, None, :] # [K, N, D]

    # Per-prefix L2 normalization
    a_unit = torch.nn.functional.normalize(a_masked, p=2, dim=-1) # [K, N, D]
    b_unit = torch.nn.functional.normalize(b_masked, p=2, dim=-1) # [K, N, D]

    # Accumulate NCE losses across K levels
    total = z_a.new_tensor(0.0)
    for i in range(K):
        # pass full-D masked & normalized vectors; trailing dims are zeros
        total = total + weights[i] * nce_loss(a_unit[i], b_unit[i], 𝜏)

    return total


# Manifolded Matryoshka Representation Loss (MMRL)
def mmrl_loss(
    whispa_embs: torch.Tensor,
    audio_embs: torch.Tensor,
    text_embs: torch.Tensor,
    psych_embs: torch.Tensor,
    K: int = 5,
    𝜏: float = 0.1,
) -> torch.Tensor:
    """
    Manifolded Matryoshka Representation Loss
    
        L_MMRL = 0.33 * MRL(Z, A) + 0.33 * MRL(Z, B) + 0.33 * MRL(Z, C)
        Z: Whispa embeddings
        A: Audio embeddings
        B: Text embeddings
        C: Psychological embeddings
    
    """
    acoustic_loss = mrl_loss(whispa_embs, audio_embs, K, 𝜏)
    semantic_loss = mrl_loss(whispa_embs[:, :text_embs.shape[1]], text_embs, K, 𝜏)
    affective_loss = mrl_loss(whispa_embs[:, -psych_embs.shape[1]:], psych_embs, K, 𝜏)
    total_loss = (acoustic_loss + semantic_loss + affective_loss) / 3
    return total_loss, acoustic_loss, semantic_loss, affective_loss


# Trinary Alignment Loss (TAL)
def trinary_alignment_loss(whispa_embs, audio_embs, text_embs, psych_embs, 𝜏=0.1):
    """
    Trinary Alignment Loss
    ==============================
    L_TAL = 0.33 * NCE(Z, A) + 0.33 * NCE(Z, B) + 0.33 * NCE(Z, C)
    Z: Whispa embeddings
    A: Audio embeddings
    B: Text embeddings
    C: Psychological embeddings
    ==============================
    """
    acoustic_loss = nce_loss(whispa_embs, audio_embs, 𝜏)
    semantic_loss = nce_loss(whispa_embs, text_embs, 𝜏)
    affective_loss = nce_loss(whispa_embs, psych_embs, 𝜏)
    total_loss = (acoustic_loss + semantic_loss + affective_loss) / 3
    return total_loss, acoustic_loss, semantic_loss, affective_loss


# def dwd_loss(
#     gating_net,
#     whispa_embs,
#     linguistic_embs,
#     acoustic_embs,
#     psych_embs=None,
#     λ=0.1,
#     𝜏=0.1
# ):
#     """
#         Dynamically Weighted Distillation with Modality-Specific Gates
#         ------------------------------
#         L_DWD = α⋅Contrastive(Z, A) + (1−α)⋅Contrastive(Z, L) + λ⋅OrthoPenalty(A, L)
#         α: Gated weight from acoustic-textual correlation estimator
#         OrthoPenalty: Penalizes redundancy between A (acoustic) and L (linguistic) subspaces
#         ------------------------------
#     """
#     # Normalize all embeddings
#     Z = F.normalize(whispa_embs, dim=-1)
#     A = F.normalize(acoustic_embs, dim=-1)
#     L = F.normalize(linguistic_embs, dim=-1)

#     if psych_embs is None:
#         acoustic_loss = nce_loss(Z, A, 𝜏)
#         linguistic_loss = nce_loss(Z, L, 𝜏)
#         ortho = torch.norm(A.T @ L, p='fro')**2

#         # Compute gating network inputs
#         with torch.no_grad():
#             mod_sim = F.cosine_similarity(A, L).mean()
#             var_acoustic = torch.var(A)
#             var_linguistic = torch.var(L)
        
#         gate_inputs = torch.tensor(
#             [mod_sim, var_acoustic, var_linguistic],
#             dtype=gating_net.module.dtype if isinstance(gating_net, torch.nn.parallel.DistributedDataParallel) \
#                 else gating_net.dtype,
#             device=gating_net.module.device if isinstance(gating_net, torch.nn.parallel.DistributedDataParallel) \
#                 else gating_net.device
#         ).unsqueeze(0)
#         α = gating_net(gate_inputs)

#         # Final loss
#         total_loss = (α * acoustic_loss + (1-α) * linguistic_loss + λ * ortho)
#         return total_loss, α, acoustic_loss, linguistic_loss, ortho
#     else:
#         raise Exception('Not Implemented!')
        

# def spectral_recon_loss(pred, target, alpha=1.0, beta=1.0):
#     """
#     Robust spectral reconstruction loss:
#     - log-cosh frame loss
#     - spectral convergence (linear magnitude)

#     Args:
#         pred: [B, 80, T] - predicted log-mel
#         target: [B, 80, T] - ground truth log-mel

#     Returns:
#         Scalar loss
#     """
#     def log_cosh_loss(pred, target, eps=1e-6):
#         return torch.mean(torch.log(torch.cosh(pred - target + eps)))

#     def spectral_convergence_loss(pred_mag, target_mag):
#         return torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-9)

#     # Log-Cosh in log-mel domain
#     logcosh = log_cosh_loss(pred, target)
#     # Spectral convergence in linear scale
#     spec_conv = spectral_convergence_loss(torch.exp(pred), torch.exp(target))

#     return alpha * logcosh + beta * spec_conv
