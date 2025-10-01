#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import math
import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta

import yaml
import wandb
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, random_split
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, InitProcessGroupKwargs
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import shutil


from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from data.peoplespeech.dataset import PeopleSpeechDataset
from data.gigaspeech.dataset import GigaSpeechDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Accelerate-powered training for WhiSPA")

    # Core config
    p.add_argument("--config_yaml", required=True, type=str, help="Path to YAML to build WhiSPAConfig and training params")
    p.add_argument("--datasets", nargs='*', choices=["gigaspeech", "peoplespeech"], default=None, help="Datasets to use; default is both if none provided")
    p.add_argument("--stage", required=False, choices=["train_enc", "train_dec"], help="Override stage (else taken from YAML)")
    p.add_argument("--language", default="en", type=str, help="Language code for processor")

    # Teacher embedding keys for train_enc
    p.add_argument("--audio_emb_key", default=None, type=str, help="Key in JSONL with path to acoustic teacher embedding .npy")
    p.add_argument("--text_emb_key", default=None, type=str, help="Key in JSONL with path to semantic/text teacher embedding .npy")
    p.add_argument("--psych_emb_key", default=None, type=str, help="Key in JSONL with path to affective/psych teacher embedding .npy")

    # Data and training hyper-params (can be overridden by YAML)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0, help="PyTorch DataLoader workers; 0 recommended (processor in collate)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight_decay", type=float, default=1.0e-2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of data for validation")
    # Cadence controls
    p.add_argument("--checkpoint_every_steps", type=int, default=5000, help="Save checkpoint every N steps (default via YAML or 5000)")
    p.add_argument("--validate_every_steps", type=int, default=20000, help="Run partial validation every N steps (default via YAML or 20000)")
    p.add_argument("--last_k_checkpoints", type=int, default=3, help="Keep only last K step checkpoints (default 3)")
    p.add_argument("--seed", type=int, default=42)

    # Mixed precision
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None, help="Accelerate mixed-precision override")

    # Resume
    p.add_argument("--resume_from", type=str, default=None, help="Path to an existing checkpoint directory to resume from (accelerate state)")

    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(yaml_cfg: Dict[str, Any], cli: argparse.Namespace) -> Tuple[WhiSPAConfig, Dict[str, Any]]:
    """
    Build WhiSPAConfig and training params from YAML and CLI overrides.
    YAML structure example:
      model:
        backbone_model_id: mistralai/Voxtral-Mini-3B-2507
        stage: train_enc
        dtype: torch.bfloat16
        device: cuda
        loss: MMRL
      train:
        batch_size: 8
        epochs: 3
        lr: 1e-4
        weight_decay: 1e-2
        mixed_precision: bf16
    """
    model_cfg = dict(yaml_cfg.get("model", {}))
    train_cfg = dict(yaml_cfg.get("train", {}))
    log_cfg = dict(yaml_cfg.get("logging", {}))
    sched_cfg = dict(yaml_cfg.get("scheduler", {}))

    # CLI overrides only if not present in YAML (YAML takes precedence)
    if ("stage" not in model_cfg or model_cfg.get("stage") is None) and cli.stage is not None:
        model_cfg["stage"] = cli.stage
    if "batch_size" not in train_cfg and cli.batch_size is not None:
        train_cfg["batch_size"] = cli.batch_size
    if "epochs" not in train_cfg and cli.epochs is not None:
        train_cfg["epochs"] = cli.epochs
    if "learning_rate" not in train_cfg and cli.lr is not None:
        train_cfg["learning_rate"] = cli.lr
    if "weight_decay" not in train_cfg and cli.weight_decay is not None:
        train_cfg["weight_decay"] = cli.weight_decay
    if "mixed_precision" not in train_cfg and cli.mixed_precision is not None:
        train_cfg["mixed_precision"] = cli.mixed_precision
    if "gradient_accumulation_steps" not in train_cfg and cli.gradient_accumulation_steps is not None:
        train_cfg["gradient_accumulation_steps"] = cli.gradient_accumulation_steps
    if "val_ratio" not in train_cfg and cli.val_ratio is not None:
        train_cfg["val_ratio"] = cli.val_ratio

    # Normalize dtype field if given as string (e.g., "float32", "bfloat16")
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "torch.float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        None: torch.bfloat16,
    }
    dt_in = model_cfg.get("dtype", None)
    if isinstance(dt_in, str):
        model_cfg["dtype"] = dtype_map.get(dt_in, torch.bfloat16)

    # Default device - always CPU for distributed training compatibility
    # Accelerator will handle device placement after prepare()
    model_cfg.setdefault("device", "cpu")

    # Build model config (extra keys are stored automatically)
    cfg = WhiSPAConfig(**model_cfg)

    # Training params with defaults
    train_params = {
        "batch_size": int(train_cfg.get("batch_size", cli.batch_size)),
        "epochs": int(train_cfg.get("epochs", cli.epochs)),
        "learning_rate": float(train_cfg.get("learning_rate", cli.lr)),
        "weight_decay": float(train_cfg.get("weight_decay", cli.weight_decay)),
        "mixed_precision": str(train_cfg.get("mixed_precision", "bf16" if cfg.dtype == torch.bfloat16 else ("fp16" if cfg.dtype == torch.float16 else "no"))),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", cli.gradient_accumulation_steps)),
        "gradient_clipping": float(train_cfg.get("gradient_clipping", 0.0)),
        "val_ratio": float(train_cfg.get("val_ratio", cli.val_ratio)),
        # Cadence defaults for large datasets (~10M samples) with YAML precedence
        "checkpoint_every_steps": int(train_cfg.get("checkpoint_every_steps", cli.checkpoint_every_steps)),
        "validate_every_steps": int(train_cfg.get("validate_every_steps", cli.validate_every_steps)),
        "last_k_checkpoints": int(train_cfg.get("last_k_checkpoints", cli.last_k_checkpoints)),
        "logging": {
            "log_with": log_cfg.get("log_with", "wandb"),
            "project": log_cfg.get("project", "WhiSPA"),
            "entity": log_cfg.get("entity", None),
            "run_name": log_cfg.get("run_name", None),
            "tags": log_cfg.get("tags", None),
            "notes": log_cfg.get("notes", None),
        },
        "scheduler": {
            "type": sched_cfg.get("type", "cosine"),
            "warmup_ratio": float(sched_cfg.get("warmup_ratio", 0.0)),
            "warmup_steps": int(sched_cfg.get("warmup_steps", 0)),
        },
    }
    return cfg, train_params


def choose_datasets(names: Optional[List[str]]) -> tuple[Dataset, Dict[str, int]]:
    if not names:
        names = ["gigaspeech", "peoplespeech"]
    ds_list: List[Dataset] = []
    sizes: Dict[str, int] = {}
    for name in names:
        if name == "gigaspeech":
            d = GigaSpeechDataset()
            ds_list.append(d)
            sizes["gigaspeech"] = len(d)
        elif name == "peoplespeech":
            d = PeopleSpeechDataset()
            ds_list.append(d)
            sizes["peoplespeech"] = len(d)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    if len(ds_list) == 1:
        return ds_list[0], sizes
    return ConcatDataset(ds_list), sizes


def split_dataset(ds: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Optional[Subset]]:
    n = len(ds)
    if val_ratio <= 0.0:
        return Subset(ds, list(range(n))), None
    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(ds, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def load_teacher_batch(batch: List[Dict[str, Any]], audio_key: str, text_key: str, psych_key: str, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load .npy embeddings for teacher targets for train_enc stage.
    """
    def _load_one(path: Optional[str]) -> Optional[np.ndarray]:
        if path is None:
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    audio_list = []
    text_list = []
    psych_list = []
    for sample in batch:
        a = _load_one(sample.get(audio_key)) if audio_key else None
        t = _load_one(sample.get(text_key)) if text_key else None
        p = _load_one(sample.get(psych_key)) if psych_key else None
        if a is None or t is None or p is None:
            raise RuntimeError("Missing teacher embeddings for train_enc stage; ensure *_emb_key CLI args match JSONL attributes and files exist")
        audio_list.append(torch.from_numpy(a))
        text_list.append(torch.from_numpy(t))
        psych_list.append(torch.from_numpy(p))

    audio = torch.stack(audio_list, dim=0).to(device=device, dtype=dtype)
    text = torch.stack(text_list, dim=0).to(device=device, dtype=dtype)
    psych = torch.stack(psych_list, dim=0).to(device=device, dtype=dtype)
    return audio, text, psych


def make_collate_for_stage(model: WhiSPAModel, stage: str, language: str, emb_keys: Tuple[Optional[str], Optional[str], Optional[str]]):
    processor = model.processor
    device = model.config.device
    dtype = model.config.dtype

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Collect audio paths
        audio_paths = [s["audio"] for s in batch]
        # Use WhiSPAProcessor to build inputs; runs on CPU and moves to correct device in processor
        inputs = processor(audio_paths, language=language)

        if stage == "train_dec":
            # Labels default to the same as text_input_ids for LM training
            labels = inputs["text_input_ids"].clone()
            return {
                "spectral_inputs": inputs["spectral_inputs"],
                "sample_spans": inputs["sample_spans"],
                "text_input_ids": inputs["text_input_ids"],
                "text_attention_mask": inputs["text_attention_mask"],
                "text_labels": labels,
            }
        elif stage == "train_enc":
            audio_key, text_key, psych_key = emb_keys
            a, t, p = load_teacher_batch(batch, audio_key, text_key, psych_key, device, dtype)
            return {
                "spectral_inputs": inputs["spectral_inputs"],
                "sample_spans": inputs["sample_spans"],
                "target_audio_embs": a,
                "target_text_embs": t,
                "target_psych_embs": p,
            }
        else:
            raise ValueError(f"Unsupported stage for training: {stage}")

    return collate


def create_optimizer(model: WhiSPAModel, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found; ensure stage is correct and components are unfrozen as intended")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)




def save_rotating_checkpoint(
    accelerator: Accelerator,
    model: WhiSPAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    base_dir: str,
    keep_last_k: int,
) -> None:
    """
    Save checkpoint with model weights and training state.
    Uses Accelerate's save_model for proper handling of distributed models.
    """
    # Create checkpoint directory
    ckpt_dir = os.path.join(base_dir, f"step-{global_step}")
    
    # Save model using accelerator (handles unwrapping and distributed saving correctly)
    accelerator.wait_for_everyone()
    accelerator.save_model(model, ckpt_dir, safe_serialization=True)
    
    # Save training state on main process only
    if accelerator.is_main_process:
        # Save config separately (save_model doesn't save it)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(ckpt_dir)
        
        # Save trainer state
        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
        }
        
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)
        
        torch.save(trainer_state, os.path.join(ckpt_dir, "trainer_state.pt"))
        
        # Update latest pointer
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "latest"), "w", encoding="utf-8") as f:
            f.write(f"step-{global_step}")
        
        # Rotate old checkpoints
        subdirs = [d for d in os.listdir(base_dir) if d.startswith("step-") and os.path.isdir(os.path.join(base_dir, d))]
        subdirs.sort(key=lambda n: int(n.split("-", 1)[1]) if n.startswith("step-") and "-" in n else -1)
        
        if len(subdirs) > keep_last_k:
            for d in subdirs[:-keep_last_k]:
                try:
                    shutil.rmtree(os.path.join(base_dir, d))
                    logging.info(f"Deleted old checkpoint: {d}")
                except Exception as e:
                    logging.warning(f"Failed to delete {d}: {e}")
        
        logging.info(f"Saved checkpoint at {ckpt_dir}")
    
    # Save accelerator state (optimizer, scheduler, RNG states, scaler)
    accelerator.wait_for_everyone()
    accelerator.save_state(ckpt_dir)
    accelerator.wait_for_everyone()


def copy_best_checkpoint_to_base(best_checkpoint_dir: str, base_dir: str, accelerator: Accelerator) -> None:
    """Copy the best checkpoint files to the base directory."""
    if not accelerator.is_main_process or not best_checkpoint_dir:
        return
    
    best_dir = os.path.join(base_dir, best_checkpoint_dir)
    if not os.path.exists(best_dir):
        logging.warning(f"Best checkpoint directory {best_dir} not found")
        return
    
    try:
        # Copy model files
        for filename in ["model.safetensors", "config.json"]:
            src = os.path.join(best_dir, filename)
            dst = os.path.join(base_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logging.info(f"Copied {filename} from best checkpoint {best_checkpoint_dir}")
            
    except Exception as e:
        logging.error(f"Failed to copy best checkpoint: {e}")


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory."""
    if not os.path.exists(base_dir):
        return None
    
    # First check for a 'latest' file
    latest_file = os.path.join(base_dir, "latest")
    if os.path.exists(latest_file):
        with open(latest_file, "r", encoding="utf-8") as f:
            latest_name = f.read().strip()
            latest_path = os.path.join(base_dir, latest_name)
            if os.path.exists(latest_path):
                return latest_path
    
    # Otherwise, find the highest numbered step-* directory
    subdirs = [d for d in os.listdir(base_dir) if d.startswith("step-") and os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    
    def _step_from_name(n: str) -> int:
        try:
            return int(n.split("-", 1)[1])
        except Exception:
            return -1
    
    subdirs.sort(key=_step_from_name)
    latest_checkpoint = os.path.join(base_dir, subdirs[-1])
    
    # Log available checkpoints for debugging
    logging.info(f"Found {len(subdirs)} checkpoint directories: {subdirs}")
    logging.info(f"Selected latest checkpoint: {subdirs[-1]}")
    
    return latest_checkpoint


def load_trainer_state_from_checkpoint(checkpoint_dir: str, grad_accum_steps: int) -> Tuple[int, int]:
    """Load epoch and global_step from a checkpoint directory.
    Note: The checkpoint directory name contains raw_step, so we need to convert back to global_step."""
    # Try JSON format first (new format)
    json_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            # global_step in the file is actually raw_step due to our naming convention
            raw_step = int(state.get("global_step", 0))
            # Convert raw_step back to optimizer step
            global_step = raw_step // grad_accum_steps if raw_step > 0 else 0
            return int(state.get("epoch", 0)), global_step
    
    # Try PT format (legacy)
    pt_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu")
        raw_step = int(state.get("global_step", 0))
        global_step = raw_step // grad_accum_steps if raw_step > 0 else 0
        return int(state.get("epoch", 0)), global_step
    
    return 0, 0


def train_loop(
    accelerator: Accelerator,
    model: WhiSPAModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    stage: str,
    epochs: int,
    grad_accum: int,
    output_dir: str,
    resume_dir: Optional[str] = None,
    start_epoch: int = 0,
) -> Tuple[int, int, Optional[str]]:
    # Prepare for accelerate
    if scheduler is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    # Pre-compute validation batch count for periodic validation
    # Use ratio based on validation frequency: validate_every_steps / total_training_steps
    if val_loader is not None and train_params_cache["validate_every_steps"] > 0:
        total_training_steps = epochs * math.ceil(len(train_loader) / max(1, grad_accum))
        validation_ratio = train_params_cache["validate_every_steps"] / total_training_steps
        max_val_batches = max(1, int(len(val_loader) * validation_ratio))
    else:
        max_val_batches = 0

    # Determine actual resume checkpoint directory
    actual_resume_dir = None
    resumed_epoch = start_epoch
    resumed_step = 0
    
    if resume_dir is not None:
        if accelerator.is_main_process:
            logging.info(f"Attempting to resume from: {resume_dir}")
        
        # Check if resume_dir is a specific checkpoint or a base directory
        if os.path.exists(os.path.join(resume_dir, "pytorch_model_fsdp.bin")) or \
           os.path.exists(os.path.join(resume_dir, "model.safetensors")) or \
           os.path.exists(os.path.join(resume_dir, "config.json")):
            # It's a specific checkpoint directory
            actual_resume_dir = resume_dir
            if accelerator.is_main_process:
                logging.info(f"Found specific checkpoint directory: {actual_resume_dir}")
        else:
            # It's a base directory, find the latest checkpoint
            if accelerator.is_main_process:
                logging.info(f"Searching for latest checkpoint in base directory: {resume_dir}")
            actual_resume_dir = find_latest_checkpoint(resume_dir)
            if actual_resume_dir:
                if accelerator.is_main_process:
                    logging.info(f"Found latest checkpoint: {actual_resume_dir}")
            else:
                if accelerator.is_main_process:
                    logging.warning(f"No step-* directories found in {resume_dir}")
        
        if actual_resume_dir and os.path.isdir(actual_resume_dir):
            try:
                # Load accelerator state (optimizer, scheduler, RNG states, scaler)
                # Note: Model weights are already loaded in main() before prepare()
                accelerator.load_state(actual_resume_dir)
                
                # Load trainer state to get epoch and step
                resumed_epoch, resumed_step = load_trainer_state_from_checkpoint(actual_resume_dir, grad_accum)
                
                if accelerator.is_main_process:
                    logging.info(f"Successfully resumed accelerator state from: {actual_resume_dir}")
                    logging.info(f"Resuming from epoch {resumed_epoch}, step {resumed_step}")
            except Exception as e:
                if accelerator.is_main_process:
                    logging.error(f"Failed to resume accelerator state from {actual_resume_dir}: {e}")
                    logging.error("Starting training from scratch...")
                resumed_epoch = start_epoch
                resumed_step = 0
        else:
            if accelerator.is_main_process:
                logging.warning(f"No valid checkpoint found in {resume_dir}, starting from scratch")

    global_step = resumed_step
    raw_step = resumed_step * grad_accum  # Track raw batch count
    best_val_loss = float("inf")
    best_checkpoint_dir = None

    # Continue from resumed_epoch (epochs are 1-indexed for logging)
    start_from_epoch = max(1, resumed_epoch if resumed_epoch > 0 else start_epoch + 1)
    for epoch in range(start_from_epoch, epochs + 1):
        model.train()
        train_running = 0.0
        train_count = 0
        val_running = 0.0
        val_count = 0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch} [train]", disable=not accelerator.is_local_main_process)
        comp_losses = {"acoustic": 0.0, "semantic": 0.0, "affective": 0.0}
        for step, batch in enumerate(train_loader, start=1):
            # Forward
            out = model(**batch)
            loss = out if isinstance(out, torch.Tensor) else (out.loss if hasattr(out, "loss") else out["total_loss"])  # type: ignore
            
            # Debug NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"NaN/Inf detected at step {step}, epoch {epoch}")
                if isinstance(out, dict):
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor):
                            logging.error(f"  {k}: {v.item() if v.numel() == 1 else 'tensor'}, nan={torch.isnan(v).any()}, inf={torch.isinf(v).any()}")
                # Log batch info
                sample_ids = [s.get('id', i) for i, s in enumerate(batch.get('sample_batch', []))]
                logging.error(f"  Batch sample IDs: {sample_ids[:5]}...")  # First 5 IDs
                raise RuntimeError("NaN/Inf loss detected!")
            
            loss = loss / grad_accum
            accelerator.backward(loss)
            
            # Increment raw step counter for every batch
            raw_step += 1

            if step % grad_accum == 0:
                max_grad_norm = train_params_cache["gradient_clipping"]
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                # Log every step to wandb
                if accelerator.is_main_process:
                    log_data = {
                        "train/loss": loss.item() * grad_accum,
                        "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                    }
                    
                    # Add component losses for train_enc
                    if stage == "train_enc" and isinstance(out, dict):
                        log_data.update({
                            "train/acoustic_loss": out.get("acoustic_loss", torch.tensor(0.0)).item(),
                            "train/semantic_loss": out.get("semantic_loss", torch.tensor(0.0)).item(),
                            "train/affective_loss": out.get("affective_loss", torch.tensor(0.0)).item(),
                        })
                    
                    wandb.log(log_data, step=global_step)

                # Validation per cadence (independent of checkpointing)
                if val_loader is not None and train_params_cache["validate_every_steps"] > 0 and (global_step % train_params_cache["validate_every_steps"] == 0):
                    model.eval()
                    total_step_val, n_step_val = 0.0, 0
                    val_comp_losses = {"acoustic": 0.0, "semantic": 0.0, "affective": 0.0}
                    
                    with torch.no_grad():
                        vbar = tqdm(total=max_val_batches, desc=f"Step {raw_step} [val]", disable=not accelerator.is_local_main_process)
                        for i, vb in enumerate(val_loader):
                            if i >= max_val_batches:
                                break
                            vout = model(**vb)
                            vloss = vout if isinstance(vout, torch.Tensor) else (vout.loss if hasattr(vout, "loss") else vout["total_loss"])  # type: ignore
                            total_step_val += float(vloss.item())
                            n_step_val += 1
                            
                            # Log component losses per batch (same as training loop)
                            if accelerator.is_main_process and stage == "train_enc" and isinstance(vout, dict):
                                log_data = {
                                    "val/acoustic_loss": vout.get("acoustic_loss", torch.tensor(0.0)).item(),
                                    "val/semantic_loss": vout.get("semantic_loss", torch.tensor(0.0)).item(),
                                    "val/affective_loss": vout.get("affective_loss", torch.tensor(0.0)).item(),
                                }
                                wandb.log(log_data, step=global_step)
                            
                            # Accumulate component losses for averaging
                            if stage == "train_enc" and isinstance(vout, dict):
                                val_comp_losses["acoustic"] += vout.get("acoustic_loss", torch.tensor(0.0)).item()
                                val_comp_losses["semantic"] += vout.get("semantic_loss", torch.tensor(0.0)).item()
                                val_comp_losses["affective"] += vout.get("affective_loss", torch.tensor(0.0)).item()
                            
                            vbar.update(1)
                        vbar.close()
                    
                    # Use total validation loss for this validation run
                    total_val_loss = total_step_val
                    model.train()
                    
                    # Log total validation loss
                    if accelerator.is_main_process:
                        wandb.log({"val/loss": total_val_loss}, step=global_step)
                        logging.info(f"Step {raw_step}: val_loss={total_val_loss:.4f}")
                        
                        # Accumulate validation loss for epoch summary (same pattern as training)
                        val_running += total_val_loss
                        val_count += 1
                        
                        # Track best validation loss using total loss
                        if total_val_loss < best_val_loss:
                            best_val_loss = total_val_loss
                            best_checkpoint_dir = f"step-{raw_step}"
                            logging.info(f"New best validation loss: {best_val_loss:.4f} at step {raw_step}")
                
                # Step-level checkpointing (independent of validation)
                if train_params_cache["checkpoint_every_steps"] > 0 and (global_step % train_params_cache["checkpoint_every_steps"] == 0):
                    # Ensure all processes are synchronized before checkpointing
                    accelerator.wait_for_everyone()
                    
                    # Save rotating last checkpoints under step subdirectories
                    save_rotating_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=raw_step,  # Use raw_step for naming
                        base_dir=output_dir,
                        keep_last_k=train_params_cache["last_k_checkpoints"],
                    )

            train_running += loss.item() * grad_accum
            train_count += 1
            
            # Update component losses for display
            if stage == "train_enc" and isinstance(out, dict):
                comp_losses["acoustic"] += out.get("acoustic_loss", torch.tensor(0.0)).item()
                comp_losses["semantic"] += out.get("semantic_loss", torch.tensor(0.0)).item()
                comp_losses["affective"] += out.get("affective_loss", torch.tensor(0.0)).item()
            
            if accelerator.is_local_main_process:
                postfix = {"loss": f"{(train_running/max(1,train_count)):.4f}"}
                if stage == "train_enc":
                    postfix.update({
                        "a": f"{(comp_losses['acoustic']/max(1,train_count)):.3f}",
                        "s": f"{(comp_losses['semantic']/max(1,train_count)):.3f}",
                        "p": f"{(comp_losses['affective']/max(1,train_count)):.3f}",
                    })
                pbar.set_postfix(postfix)
            pbar.update(1)

        train_loss = train_running / max(1, train_count)
        val_loss = val_running / max(1, val_count) if val_count > 0 else None
        pbar.close()

        if accelerator.is_main_process:
            # Log epoch-end metrics (same pattern for both train and val)
            epoch_data = {"epoch": epoch, "train/epoch_loss": train_loss}
            if val_loss is not None:
                epoch_data["val/epoch_loss"] = val_loss
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}" + (f", val_loss={val_loss:.4f}" if val_loss is not None else ""))
            wandb.log(epoch_data, step=global_step)

        accelerator.wait_for_everyone()
    
    # Return final epoch, global_step, and best checkpoint info
    return epoch, global_step, best_checkpoint_dir


def main() -> None:
    args = parse_args()

    # Load YAML and build configs
    yaml_cfg = load_yaml(args.config_yaml)
    whispa_cfg, train_params = build_config(yaml_cfg, args)

    # Accelerator with 6-hour timeout for wait_for_everyone
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # Set 6-hour timeout (21600 seconds) for distributed operations
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=21600))
    mixed_precision = args.mixed_precision or train_params["mixed_precision"]
    # Resolve output directory from env and run_name
    base_ckpt_dir = os.getenv('CHECKPOINT_DIR')
    if base_ckpt_dir is None or base_ckpt_dir.strip() == "":
        raise RuntimeError("CHECKPOINT_DIR environment variable is not set")
        
    run_name = train_params["logging"].get("run_name")
    if not run_name:
        run_name = f"whispa-{int(time.time())}"
        train_params["logging"]["run_name"] = run_name
    output_dir = os.path.join(base_ckpt_dir, run_name)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        log_with=train_params["logging"]["log_with"],
        project_dir=output_dir,
    )
    set_seed(args.seed)

    # IMPORTANT: Model MUST be on CPU before accelerator.prepare() for distributed training
    whispa_cfg.device = torch.device("cpu")
    if whispa_cfg.dtype is None:
        if accelerator.mixed_precision == "bf16":
            whispa_cfg.dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            whispa_cfg.dtype = torch.float16
        else:
            whispa_cfg.dtype = torch.float32

    # Logging setup
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.info("Initializing model with config:")
        logging.info(str(whispa_cfg))

    # Create model
    model_loaded_from_checkpoint = False
    if args.resume_from is not None:
        # Try to find the checkpoint directory
        resume_checkpoint_dir = None
        if os.path.exists(os.path.join(args.resume_from, "model.safetensors")) or \
           os.path.exists(os.path.join(args.resume_from, "model.safetensors.index.json")) or \
           os.path.exists(os.path.join(args.resume_from, "config.json")):
            # It's a specific checkpoint directory
            resume_checkpoint_dir = args.resume_from
        else:
            # It's a base directory, find the latest checkpoint
            resume_checkpoint_dir = find_latest_checkpoint(args.resume_from)
        
        if resume_checkpoint_dir and os.path.isdir(resume_checkpoint_dir):
            try:
                if accelerator.is_main_process:
                    logging.info(f"Loading model from checkpoint: {resume_checkpoint_dir}")
                    
                model = WhiSPAModel.from_pretrained_local(resume_checkpoint_dir)
                # Override config with current training config
                for key, value in whispa_cfg.__dict__.items():
                    setattr(model.config, key, value)
                model.config.device = torch.device("cpu")
                model = model.cpu()
                model_loaded_from_checkpoint = True

                if accelerator.is_main_process:
                    logging.info("Successfully loaded model from checkpoint")
            except Exception as e:
                if accelerator.is_main_process:
                    logging.warning(f"Failed to load model from checkpoint: {e}")
                    logging.info("Creating new model instead")
    
    # Create new model if not loaded from checkpoint
    if not model_loaded_from_checkpoint:
        model = WhiSPAModel(whispa_cfg)
        if accelerator.is_main_process:
            logging.info("Created new model from config")
    
    model.set_stage(whispa_cfg.stage)
    model.train()

    # Dataset
    full_ds, size_map = choose_datasets(args.datasets)
    total_size = len(full_ds)
    train_subset, val_subset = split_dataset(full_ds, train_params["val_ratio"], args.seed)
    if accelerator.is_main_process:
        logging.info(f"Datasets loaded: {size_map}")
        logging.info(f"Total combined samples: {total_size}")
        if val_subset is None:
            logging.info(f"Split: train={len(train_subset)}, val=0 (no validation split)")
        else:
            logging.info(f"Split: train={len(train_subset)}, val={len(val_subset)}")

    # Collate for stage
    stage = whispa_cfg.stage
    emb_keys = (args.audio_emb_key, args.text_emb_key, args.psych_emb_key)
    collate_fn = make_collate_for_stage(model, stage, args.language, emb_keys)

    # DataLoader (num_workers=0 recommended; processor used in collate)
    train_loader = DataLoader(train_subset, batch_size=train_params["batch_size"], shuffle=True, num_workers=max(0, int(args.num_workers)), collate_fn=collate_fn, pin_memory=False)
    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader = DataLoader(val_subset, batch_size=train_params["batch_size"], shuffle=False, num_workers=max(0, int(args.num_workers)), collate_fn=collate_fn, pin_memory=False)

    # Optimizer
    optimizer = create_optimizer(model, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"])

    # Scheduler
    scheduler = None
    if train_params["scheduler"]["type"] != "none":
        num_update_steps_per_epoch = math.ceil(len(train_loader) / max(1, train_params["gradient_accumulation_steps"]))
        max_train_steps = train_params["epochs"] * num_update_steps_per_epoch
        warmup_steps = train_params["scheduler"]["warmup_steps"]
        if warmup_steps == 0:
            warmup_steps = int(train_params["scheduler"]["warmup_ratio"] * max_train_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps)

    # Init trackers (wandb) on main process
    if accelerator.is_main_process and train_params["logging"]["log_with"] == "wandb":
        wandb_kwargs = {
            "project": train_params["logging"]["project"],
            "name": train_params["logging"]["run_name"],
            "entity": train_params["logging"]["entity"],
            "tags": train_params["logging"]["tags"],
            "notes": train_params["logging"]["notes"],
            "dir": output_dir,
            "config": {"model": whispa_cfg.to_dict() if hasattr(whispa_cfg, 'to_dict') else str(whispa_cfg), "train": train_params},
        }
        # Remove None entries
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        wandb.init(**wandb_kwargs)

    # Start epoch will be determined from resume_dir in train_loop
    start_epoch = 0

    # Train
    # Expose cadence to train_loop via module-level cache
    global train_params_cache
    train_params_cache = {
        "checkpoint_every_steps": train_params["checkpoint_every_steps"],
        "validate_every_steps": train_params["validate_every_steps"],
        "last_k_checkpoints": train_params["last_k_checkpoints"],
        "gradient_clipping": train_params["gradient_clipping"],
    }

    final_epoch, final_step, best_checkpoint_dir = train_loop(
        accelerator=accelerator,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        stage=stage,
        epochs=train_params["epochs"],
        grad_accum=int(train_params["gradient_accumulation_steps"]),
        output_dir=output_dir,
        resume_dir=args.resume_from or output_dir,
        start_epoch=start_epoch,
    )

    # Save final checkpoint at training completion
    if final_step > 0:
        final_raw_step = final_step * train_params["gradient_accumulation_steps"]
        save_rotating_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=final_epoch,
            global_step=final_raw_step,
            base_dir=output_dir,
            keep_last_k=train_params["last_k_checkpoints"],
        )
        if accelerator.is_main_process:
            logging.info(f"Saved final checkpoint at step {final_raw_step}")
    
    # Copy best checkpoint to base directory
    if accelerator.is_main_process and best_checkpoint_dir:
        copy_best_checkpoint_to_base(best_checkpoint_dir, output_dir, accelerator)
        logging.info(f"Best model checkpoint: {best_checkpoint_dir}")

    # Final synchronization before exit
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logging.info("Training complete.")


if __name__ == "__main__":
    main()
