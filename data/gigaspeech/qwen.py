import os
import torch
from transformers import AutoTokenizer, AutoModel
from train.utils import last_token_pool

model_id = "Qwen/Qwen3-Embedding-8B"
dataset_path = "/mnt/vast/data/speech/gigaspeech/data/data/gigaspeech_full.jsonl"
extract_key = "affect"
instruction = "Represent this speech transcript for semantic similarity search"

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
model.eval()

# Use left padding as recommended for Qwen3-Embedding models
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True,
    padding_side='left'
)
# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


