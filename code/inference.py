import time
from datetime import timedelta
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from config import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    EMBEDDINGS_DIR
)
from model import WhiSBERTModel
from data import AudioDataset, collate_inference


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

    return processor, whisbert


def inference(
    dataset,
    processor,
    whisbert,
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

    output_path = os.path.join(EMBEDDINGS_DIR, load_name)
    os.makedirs(output_path, exist_ok=True)
    hitop_output_filepath = os.path.join(output_path, f'hitop_embeddings.csv')
    wtc_output_filepath = os.path.join(output_path, f'wtc_embeddings.csv')
    assert not (os.path.exists(hitop_output_filepath) or os.path.exists(wtc_output_filepath)), (
        f'OutputError: The output filepath(s) already exist.\n\t{hitop_output_filepath}\n\t{wtc_output_filepath}'
    )
 
    cols = ['message_id'] + [f'f{i:03d}' for i in range(config.emb_dims + config.n_new_dims)]
    df = pd.DataFrame(columns=cols)
    df.to_csv(hitop_output_filepath, index=False)
    df.to_csv(wtc_output_filepath, index=False)

    whisbert.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader):
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
            whis_embs = F.normalize(whis_embs, p=2, dim=1)

            for m_idx, message_id in enumerate(batch['message_id']):
                emb = whis_embs[m_idx].cpu().numpy().tolist()
                df = pd.DataFrame([[message_id] + emb], columns=cols)
                if batch['dataset_name'][m_idx] == 'hitop':
                    df.to_csv(hitop_output_filepath, mode='a', header=False, index=False)
                elif batch['dataset_name'][m_idx] == 'wtc':
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
    processor, whisbert = load_models(config, args.load_name)

    print('\nPreprocessing AudioDataset...')
    dataset = AudioDataset(config, processor, mode='inference')
    print(f'\tTotal dataset size (N): {len(dataset)}')

    print('\nStarting Inference...')
    inference(
        dataset,
        processor,
        whisbert,
        config,
        args.load_name
    )

    torch.cuda.empty_cache()


if __name__ == '__main__':    
    main()