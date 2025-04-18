import sys, os

# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log"),
    ],
)

import time
from datetime import timedelta
import argparse
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    WhisperModel,
    WhisperProcessor,
    Wav2Vec2Model,
    Wav2Vec2BertModel,
    HubertModel
)
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs
)
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_utils import mean_pooling
from pretrain.whispa_model import WhiSPAModel
from pretrain.whispa_data import AudioDataset, collate_inference


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


def load_args():
    parser = argparse.ArgumentParser(description='Script to inference WhiSPA model (Generates Embeddings)')
    parser.add_argument(
        '--load_name',
        default='',
        type=str,
        help='Specify the filename to the model directory. It will use `config.pth` and `best.pth` saved in: <CHECKPOINT_DIR>/<MODEL_NAME>/`'
    )
    parser.add_argument(
        '--hf_model_id',
        default='',
        type=str,
        help='Specify the HuggingFace model id to use. `Ex. jinaai/jina-embeddings-v3`'
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


def load_models(config, load_name, hf_model_id):
    processor = None
    tokenizer = None
    model = None
    accelerator = None

    if config.device == 'cuda':
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        config.device = accelerator.device
        logging.info(f"  Accelerator using device: {config.device}")

    if hf_model_id:
        if 'jinaai/' in hf_model_id:
            model = AutoModel.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                trust_remote_code=True
            ).to(config.device)

        elif 'sentence-transformers/' in hf_model_id:
            # Load the pre-trained SentenceTransformer model and tokenizer
            model = AutoModel.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).to(config.device)
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                TOKENIZERS_PARALLELISM=False
            )

        elif 'whisper' in hf_model_id:
            # Load the Whisper encoder model and processor
            model = AutoModel.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).encoder.to(config.device)
            processor = AutoProcessor.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                device_map=config.device
            )

        elif 'wav2vec2-bert' in hf_model_id:
            # Load the Wav2Vec2-BERT model and processor
            model = Wav2Vec2BertModel.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).to(config.device)
            processor = AutoProcessor.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                device_map=config.device
            )

        elif 'wav2vec2' in hf_model_id:
            # Load the Wav2Vec2 model and processor
            model = Wav2Vec2Model.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).to(config.device)
            processor = AutoProcessor.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                device_map=config.device
            )

        elif 'hubert' in hf_model_id:
            # Load the HuBERT model and processor
            model = HubertModel.from_pretrained(hf_model_id, cache_dir=os.getenv('CACHE_DIR')).to(config.device)
            processor = AutoProcessor.from_pretrained(
                hf_model_id,
                cache_dir=os.getenv('CACHE_DIR'),
                device_map=config.device
            )
        else:
            raise ValueError('Not implemented yet. Please provide a valid model id.')
    else:
        # Load the WhiSPA model and processor
        model = WhiSPAModel(config).to(config.device)
        processor = WhisperProcessor.from_pretrained(
            config.whisper_model_id,
            cache_dir=os.getenv('CACHE_DIR'),
            device_map=config.device
        )

        logging.info('Instantiating WhiSPA with loaded state dict...')
        state_dict = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), load_name, 'best.pth'), map_location=config.device)
        try:
            model.load_state_dict(state_dict)
        except:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        
    
    # Compile the model for better performance
    # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    logging.debug(f'{type(model)}\n{model}\n')
        
    return processor, tokenizer, model, accelerator


def inference(
    dataset,
    processor,
    tokenizer,
    model,
    accelerator,
    config,
    save_name
):
    # Prepare data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=collate_inference
    )
    
    # Handle output file and paths
    if '/' in save_name:
        save_name = save_name[save_name.find('/') + 1:]
        config.hidden_size = model.config.hidden_size

    output_path = os.path.join(os.getenv('EMBEDDINGS_DIR'), save_name)
    os.makedirs(output_path, exist_ok=True)
    hitop_output_filepath = os.path.join(output_path, f'hitop_embeddings.csv')
    wtc_output_filepath = os.path.join(output_path, f'wtc_embeddings.csv')
 
    cols = ['message_id'] + [f'f{i:04d}' for i in range(config.hidden_size + config.n_new_dims)]
    df = pd.DataFrame(columns=cols)
    df.to_csv(hitop_output_filepath, index=False)
    df.to_csv(wtc_output_filepath, index=False)

    # Prepare model, and data loader with Accelerator
    if accelerator is not None:
        model, data_loader = accelerator.prepare(model, data_loader)

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
                ).to(model.module.dtype).to(config.device)

                # Get SBERT's MEAN embedding
                sbert_embs = model.module(**sbert_inputs).last_hidden_state
                sbert_embs = mean_pooling(sbert_embs, sbert_inputs['attention_mask'])
                embs = F.normalize(sbert_embs, p=2, dim=1)
            
            elif save_name == 'jina-embeddings-v3':
                # Get JINA's MEAN embedding
                jina_embs = torch.tensor(model.module.encode(
                    batch['message'],
                    task='classification',
                    show_progress_bar=False,
                    truncate_dim=config.hidden_size
                )).to(torch.float32).to(config.device)
                embs = F.normalize(jina_embs, p=2, dim=1)
            
            elif save_name == 'whisper-medium':
                # Get Whisper Encoder's MEAN embedding
                whis_embs = model.module(batch['audio_inputs'].to(config.device))
                whis_embs = whis_embs.last_hidden_state.mean(1)
                embs = F.normalize(whis_embs, p=2, dim=1)
            
            elif isinstance(processor, transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2Processor):
                # Get W2V2/HuBERT's MEAN embedding
                try:
                    wav_embs = model.module(input_values=batch['audio_inputs'].to(config.device))
                    wav_embs = wav_embs.last_hidden_state.mean(1)
                    embs = F.normalize(wav_embs, p=2, dim=1)
                except Exception as e:
                    logging.info(Warning(e))
            
            elif isinstance(processor, transformers.models.wav2vec2_bert.processing_wav2vec2_bert.Wav2Vec2BertProcessor):
                try:
                    # Get W2V2-BERT's MEAN embedding
                    wav_embs = model.module(input_features=batch['audio_inputs'].to(config.device))
                    wav_embs = wav_embs.last_hidden_state.mean(1)
                    embs = F.normalize(wav_embs, p=2, dim=1)
                except Exception as e:
                    logging.info(Warning(e))

            else:
                # Whisper-based tokenization
                outputs = processor.tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.dtype).to(config.device)

                # Get WhiSPA's embedding
                whis_embs = model.module(
                    batch['audio_inputs'].to(config.device),
                    outputs['input_ids'],
                    outputs['attention_mask']
                ).to(torch.float32)
                embs = F.normalize(whis_embs, p=2, dim=1)

            for m_idx, message_id in enumerate(batch['message_id']):
                emb = embs[m_idx].cpu().numpy().tolist()
                df = pd.DataFrame([[message_id] + emb], columns=cols)
                if batch['dataset_name'][m_idx] == 'hitop':
                    df.to_csv(hitop_output_filepath, mode='a', header=False, index=False)
                elif batch['dataset_name'][m_idx] == 'wtc':
                    df.to_csv(wtc_output_filepath, mode='a', header=False, index=False)

    # Clean the saved embedding files
    hitop_df = pd.read_csv(hitop_output_filepath).drop_duplicates(subset=['message_id'])
    hitop_df.to_csv(hitop_output_filepath, index=False)
    wtc_df = pd.read_csv(wtc_output_filepath).drop_duplicates(subset=['message_id'])
    wtc_df.to_csv(wtc_output_filepath, index=False)

    elapsed_time = timedelta(seconds=time.time() - start_time)
    logging.info(f"Elapsed Time: {elapsed_time}")
    

def main():
    args = load_args()
    save_name = args.load_name

    if torch.cuda.is_available():
        gpus = list(range(torch.cuda.device_count()))
        logging.info(f"Available GPU IDs: {gpus}")
        for i in gpus:
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("CUDA is not available. Only CPU will be used.")
        args.device = 'cpu'

    logging.info('Preparing Model Configuration...')
    if args.load_name:
        logging.info('  Initializing WhiSPA Config from Load File...')
        config = torch.load(os.path.join(
            os.getenv('CHECKPOINT_DIR'), 
            args.load_name, 
            'config.pth'
        ), weights_only=False)
        config.shuffle = not args.no_shuffle
        if config.batch_size != args.batch_size:
            config.batch_size = args.batch_size
        if config.num_workers != args.num_workers:
            config.num_workers = args.num_workers
        if config.device != args.device:
            config.device = args.device
        logging.info(config)
    elif args.hf_model_id:
        save_name = args.hf_model_id
        logging.info(f'  Initializing Pretrained Model: `{args.hf_model_id}` from HuggingFace...')
        config = WhiSPAConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=not args.no_shuffle,
            device=args.device
        )
    else:
        raise ValueError('Please provide either a `load_name` or `hf_model_id` argument.')

    logging.info('Loading and Initializing Models with Config...')
    processor, tokenizer, model, accelerator = load_models(config, args.load_name, args.hf_model_id)

    logging.info('Preprocessing AudioDataset...')
    dataset = AudioDataset(config, [processor], dtype=config.dtype, mode='inference')
    logging.info(f'  Total dataset size (N): {len(dataset)}')

    logging.info('Starting Inference...')
    inference(
        dataset,
        processor,
        tokenizer,
        model,
        accelerator,
        config,
        save_name
    )

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    logging.info('Finished Inference!')


if __name__ == '__main__':    
    main()