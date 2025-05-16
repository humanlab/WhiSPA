import os
import logging
import numpy as np
import pandas as pd
import torch, torchaudio
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset, load_from_disk
# import aiohttp

# dataset = load_dataset(
#     "openslr/librispeech_asr",
#     storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
# )
# dataset.save_to_disk("/cronus_data/rrao/datasets/librispeech")

dataset = load_from_disk("/cronus_data/rrao/datasets/iemocap")
print(dataset)

