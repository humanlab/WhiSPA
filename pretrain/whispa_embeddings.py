import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

import time
from datetime import timedelta
import argparse
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    WhisperProcessor,
    Wav2Vec2Processor,
    Wav2Vec2Model
)
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_utils import mean_pooling
from pretrain.whispa_model import WhiSPAModel
from pretrain.whispa_data import AudioDataset, collate_inference


def load_args():
    parser = argparse.ArgumentParser(description='Script to inference WhiSPA model (Generates Embeddings)')
    parser.add_argument(
        '--load_name',
        required=True,
        type=str,
        help='Specify the filename to the model directory. It will use `config.pth` and `best.pth` saved in: <CHECKPOINT_DIR>/<MODEL_NAME>/`\nOr specify the HuggingFace model id for a SBERT autoencoder from the sentence-transformers/ library. `Ex. sentence-transformers/all-MiniLM-L12-v2`'
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
        choices=[
            'cuda',
            'cpu'
        ],
        type=str,
        help='Specify whether to use CPU or GPU'
    )
    return parser.parse_args()


def load_models(config, load_name):
    processor = None
    tokenizer = None
    if 'sentence-transformers/' in load_name:
        # Load the pre-trained SentenceTransformer model and tokenizer
        model = AutoModel.from_pretrained(load_name, cache_dir=os.getenv('CACHE_DIR')).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(
            load_name,
            cache_dir=os.getenv('CACHE_DIR'),
            TOKENIZERS_PARALLELISM=False
        )
    elif 'wav2vec' in load_name:
        # Load the Wav2Vec model and processor
        model = Wav2Vec2Model.from_pretrained(load_name, cache_dir=os.getenv('CACHE_DIR')).to(config.device)
        processor = Wav2Vec2Processor.from_pretrained(
            load_name,
            cache_dir=os.getenv('CACHE_DIR'),
            device_map=config.device
        )
    else:
        # Load the Whisper model and processor
        model = WhiSPAModel(config).to(config.device)
        processor = WhisperProcessor.from_pretrained(
            config.whisper_model_id,
            cache_dir=os.getenv('CACHE_DIR'),
            device_map=config.device
        )

    if config.device == 'cuda':
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
            print(f"\nAvailable GPU IDs: {gpus}")
            for i in gpus:
                print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
            print()
        else:
            print("CUDA is not available. Only CPU will be used.\n")
        model = torch.nn.DataParallel(model, device_ids=gpus)

    if not ('sentence-transformers/' in load_name or 'wav2vec' in load_name):
        print('Instantiating WhiSPA with loaded state dict...')
        state_dict = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), load_name, 'best.pth'))
        try:
            model.load_state_dict(state_dict)
        except:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

    return processor, tokenizer, model


def inference(
    dataset,
    processor,
    tokenizer,
    model,
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

    # Handle config for SBERT or Wav2Vec during inference
    if tokenizer is not None:
        load_name = load_name.replace('sentence-transformers/', '')
        config.emb_dims = model.module.config.hidden_size
    elif isinstance(processor, transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2Processor):
        load_name = load_name.replace('facebook/', '')
        config.emb_dims = model.module.config.hidden_size
    
    # Handle output file and paths
    output_path = os.path.join(os.getenv('EMBEDDINGS_DIR'), load_name)
    os.makedirs(output_path, exist_ok=True)
    hitop_output_filepath = os.path.join(output_path, f'hitop_embeddings.csv')
    wtc_output_filepath = os.path.join(output_path, f'wtc_embeddings.csv')
    assert not (os.path.exists(hitop_output_filepath) or os.path.exists(wtc_output_filepath)), (
        f'OutputError: The output filepath(s) already exist.\n\t{hitop_output_filepath}\n\t{wtc_output_filepath}'
    )
 
    cols = ['message_id'] + [f'f{i:04d}' for i in range(config.emb_dims + config.n_new_dims)]
    df = pd.DataFrame(columns=cols)
    df.to_csv(hitop_output_filepath, index=False)
    df.to_csv(wtc_output_filepath, index=False)

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader):

            if tokenizer is not None:
                # SBERT-based tokenization
                sbert_inputs = tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(config.device)

                # Get SBERT's MEAN embedding
                sbert_embs = model(**sbert_inputs).last_hidden_state
                sbert_embs = mean_pooling(sbert_embs, sbert_inputs['attention_mask'])
                embs = F.normalize(sbert_embs, p=2, dim=1)
            
            elif isinstance(processor, transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2Processor):
                # Get Wav2Vec's MEAN embedding
                try:
                    wav_embs = model(input_values=batch['audio_inputs'].to(config.device))
                    wav_embs = wav_embs.last_hidden_state.mean(1)
                    embs = F.normalize(wav_embs, p=2, dim=1)
                except Exception as e:
                    print(Warning(e))

            else:
                # Whisper-based tokenization
                outputs = processor.tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.device)

                # Get WhiSPA's MEAN/LAST token
                whis_embs = model(
                    batch['audio_inputs'].to(config.device),
                    outputs['input_ids'],
                    outputs['attention_mask']
                )
                embs = F.normalize(whis_embs, p=2, dim=1)

            for m_idx, message_id in enumerate(batch['message_id']):
                emb = embs[m_idx].cpu().numpy().tolist()
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
    if 'sentence-transformers/' in args.load_name or 'wav2vec' in args.load_name:
        print(f'\tInitializing Pretrained Model: `{args.load_name}`...')
        config = WhiSPAConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=not args.no_shuffle,
            device=args.device
        )
    else:
        print('\tInitializing WhiSPA Config from Load File...')
        config = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), args.load_name, 'config.pth'))
        config.shuffle = not args.no_shuffle
        if config.batch_size != args.batch_size:
            config.batch_size = args.batch_size
        if config.num_workers != args.num_workers:
            config.num_workers = args.num_workers
        if config.device != args.device:
            config.device = args.device
        print(config)

    print('\nLoading and Initializing Models with Config...')
    processor, tokenizer, model = load_models(config, args.load_name)

    print('\nPreprocessing AudioDataset...')
    dataset = AudioDataset(config, processor, mode='inference')
    print(f'\tTotal dataset size (N): {len(dataset)}')

    print('\nStarting Inference...')
    inference(
        dataset,
        processor,
        tokenizer,
        model,
        config,
        args.load_name
    )

    torch.cuda.empty_cache()


if __name__ == '__main__':    
    main()