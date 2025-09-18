#!/usr/bin/env python3

import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import time
import torch
from model.config import WhiSPAConfig
from model.whispa import WhiSPAModel
from dotenv import load_dotenv

load_dotenv()


def test_save_and_load_local():
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR")
    save_dir = os.path.join(CHECKPOINT_DIR, "Voxtral-Mini-3B")
    os.makedirs(save_dir, exist_ok=True)

    cfg = WhiSPAConfig(stage='inference', device='cpu', dtype=torch.bfloat16)
    model = WhiSPAModel(cfg).eval()

    # Save
    start_time = time.time()
    model.save_pretrained_local(save_dir, safe_serialization=True)
    elapsed = time.time() - start_time
    print(f"Model save time: {elapsed:.3f} seconds")
    assert os.path.exists(os.path.join(save_dir, "config.json")), "config.json missing"
    assert os.path.exists(os.path.join(save_dir, "model.safetensors")), "model.safetensors missing"

    # Load
    start_time = time.time()
    loaded = WhiSPAModel.from_pretrained_local(save_dir).eval()
    elapsed = time.time() - start_time
    print(f"Model load time: {elapsed:.3f} seconds")

    # Compare a couple of parameters to ensure weights round-trip
    keys = list(model.state_dict().keys())[:5]
    for k in keys:
        a = model.state_dict()[k]
        b = loaded.state_dict()[k]
        assert torch.allclose(a.cpu(), b.cpu(), atol=0, rtol=0), f"Mismatch on param {k}"

    print("Save/load local OK →", save_dir)


if __name__ == "__main__":
    test_save_and_load_local()
