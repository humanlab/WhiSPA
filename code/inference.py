import time
from datetime import timedelta
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    WhisperProcessor
)
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from config import (
    WhiSBERTConfig,
    CACHE_DIR,
    CHECKPOINT_DIR,
    EMBEDDINGS_DIR
)
from model import WhiSBERTModel
from data import AudioDataset, collate_inference
from utils import (
    mean_pooling,
    cos_sim_loss,
    sim_clr_loss,
    norm_temp_ce_loss
)


"""
CUDA_VISIBLE_DEVICES=0 python code/inference.py \
--load_name whisper-tiny_mean_cos-sim_50_512_1e-5_1e-2 \
--batch_size 1024 \
--num_workers 12 \
--no_shuffle
"""


os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_args():
    parser = argparse.ArgumentParser(description='Script to inference WhiSBERT model (Generates Embeddings)')
    parser.add_argument(
        '--load_name',
        required=True,
        type=str,
        help='Specify the filename to the model directory. It will use `config.pth` and `best.pth` saved in: /cronus_data/rrao/WhiSBERT/models/<MODEL_NAME>/`'
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="The number of workers for data pre-loading"
    )
    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not shuffle the dataset'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Specify whether to use CPU or GPU'
    )
    return parser.parse_args()


def load_models(config, load_name):
    # Load the WhiSBERT and Whisper processor
    processor = WhisperProcessor.from_pretrained(
        config.whisper_model_id,
        cache_dir=CACHE_DIR,
        device_map=config.device
    )
    whisbert = WhiSBERTModel(config).to(config.device)

    # # Load the pre-trained SentenceTransformer models
    # if config.sbert_model_id == 'sentence-transformers/distiluse-base-multilingual-cased-v1':
    #     tokenizer = None
    #     sbert = SentenceTransformer(config.sbert_model_id, cache_folder=CACHE_DIR, device=config.device)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR, TOKENIZERS_PARALLELISM=False)
    #     sbert = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR).to(config.device)

    if config.device == 'cuda':
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
            print(f"\nAvailable GPU IDs: {gpus}")
            for i in gpus:
                print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
            print()
        else:
            print("CUDA is not available. Only CPU will be used.\n")
        whisbert = torch.nn.DataParallel(whisbert, device_ids=gpus)
        # sbert = torch.nn.DataParallel(sbert, device_ids=gpus)

    print('Instantiating WhiSBERT with loaded state dict...')
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, load_name, 'best.pth'))
    try:
        whisbert.load_state_dict(state_dict)
    except:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        whisbert.load_state_dict(state_dict)

    return processor, whisbert, None, None


def inference(
    dataset,
    processor,
    whisbert,
    tokenizer,
    sbert,
    config,
    load_name
):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=collate_inference
    )

    hitop_output_filepath = os.path.join(EMBEDDINGS_DIR, f'hitop_{load_name}.csv')
    wtc_output_filepath = os.path.join(EMBEDDINGS_DIR, f'wtc_{load_name}.csv')
    if os.path.exists(hitop_output_filepath) or os.path.exists(wtc_output_filepath):
        raise Exception(f'OutputError: The output filepath(s) already exist.\n\t{hitop_output_filepath}\n\t{wtc_output_filepath}')
    else:
        cols = ['segment_id'] + [f'f{i:03d}' for i in range(config.emb_dim)]
        df = pd.DataFrame(columns=cols)
        df.to_csv(hitop_output_filepath, index=False)
        df.to_csv(wtc_output_filepath, index=False)

    whisbert.eval()
    # sbert.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader):           
            # # Get SBERT's embedding
            # if config.sbert_model_id != 'sentence-transformers/distiluse-base-multilingual-cased-v1':
            #     # SBERT-based tokenization
            #     sbert_inputs = tokenizer(
            #         batch['message'],
            #         padding=True,
            #         truncation=True,
            #         return_tensors='pt'
            #     ).to(config.device)
            #     sbert_embs = sbert(**sbert_inputs).last_hidden_state
            #     sbert_embs = mean_pooling(sbert_embs, sbert_inputs['attention_mask'])
            # else:
            #     if isinstance(sbert, torch.nn.DataParallel):
            #         sbert_embs = torch.from_numpy(sbert.module.encode(batch['message']))
            #     else:
            #         sbert_embs = torch.from_numpy(sbert.encode(batch['message']))

            # Whisper-based tokenization
            with torch.no_grad():
                outputs = processor.tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.device)

            # Get WhiSBERT's MEAN/LAST token
            whis_embs = whisbert(
                batch['audio_inputs'].to(config.device),
                outputs['input_ids'],
                outputs['attention_mask']
            )

            # sbert_embs = F.normalize(sbert_embs, p=2, dim=1)
            whis_embs = F.normalize(whis_embs, p=2, dim=1)

            for s_idx, segment_id in enumerate(batch['segment_id']):
                emb = whis_embs[s_idx].cpu().numpy().tolist()
                df = pd.DataFrame([[segment_id] + emb], columns=cols)
                if batch['dataset_name'][s_idx] == 'hitop':
                    df.to_csv(hitop_output_filepath, mode='a', header=False, index=False)
                elif batch['dataset_name'][s_idx] == 'wtc':
                    df.to_csv(wtc_output_filepath, mode='a', header=False, index=False)

    elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"Elapsed Time: {elapsed_time}")
        

def main():
    args = load_args()

    print('Preparing Model Configuration...')
    print('\tInitializing WhiSBERT Config from Load File...')
    config = torch.load(os.path.join(CHECKPOINT_DIR, args.load_name, 'config.pth'))
    config.shuffle = not args.no_shuffle
    if config.batch_size != args.batch_size:
        config.batch_size = args.batch_size
    if config.num_workers != args.num_workers:
        config.num_workers = args.num_workers
    if config.device != args.device:
        config.device = args.device
    print(config)

    print('\nLoading and Initializing Models with Config...')
    processor, whisbert, tokenizer, sbert = load_models(config, args.load_name)

    print('\nPreprocessing AudioDataset...')
    dataset = AudioDataset(processor, mode='inference')
    print(f'\tTotal dataset size (N): {len(dataset)}')

    print('\nStarting Inference...')
    inference(
        dataset,
        processor,
        whisbert,
        tokenizer,
        sbert,
        config,
        args.load_name
    )

    torch.cuda.empty_cache()


if __name__ == '__main__':    
    main()