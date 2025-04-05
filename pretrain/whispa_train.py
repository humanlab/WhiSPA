import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log"),
    ],
)

import time
from datetime import timedelta
from collections import OrderedDict
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_model import WhiSPAModel
from pretrain.whispa_data import AudioDataset, collate_train
from pretrain.whispa_utils import (
    dwd_loss
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_args():
    parser = argparse.ArgumentParser(description='Script to train WhiSPA model')
    # Training params
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The batch size for the training loop"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="The number of workers for data pre-loading"
    )
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int,
        help="The number of epochs for the training loop"
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        default=5e-5,
        type=float,
        help="The learning rate for the training loop"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=0.01,
        type=float,
        help="The weight decay for the training loop"
    )
    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not shuffle the dataset during training'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Specify whether to use CPU or GPU during training'
    )
    parser.add_argument(
        '--save_name',
        default='',
        type=str,
        help='Specify the filename of the model directory. After training, the best state will be saved to: `<CHECKPOINT_DIR>/<MODEL_NAME>/`'
    )
    parser.add_argument(
        '--load_name',
        default='',
        type=str,
        help='Specify the filename to the model directory. It will use `config.pth` and `best.pth` saved in: <CHECKPOINT_DIR>/<MODEL_NAME>/`'
    )
    
    # Hyperparams
    parser.add_argument(
        '--whisper_model_id',
        default='openai/whisper-medium',
        choices=[
            'openai/whisper-tiny',
            'openai/whisper-small',
            'openai/whisper-medium'
        ],
        type=str,
        help='Specify the model_id of the Whisper variant on HuggingFace repository'
    )
    parser.add_argument(
        '--pooling_mode',
        default='mean',
        choices=[
            'mean',
            'last'
        ],
        type=str,
        help='Specify the pooling mode to select the embedding from WhiSPA'
    )
    parser.add_argument(
        "--n_new_dims",
        default=0,
        choices=[
            0,
            10,
        ],
        type=int,
        help="The number of additional dimensions to be added"
    )
    parser.add_argument(
        '--loss',
        default='DWD',
        choices=[
            'DWD',
            'MOW',
        ],
        type=str,
        help='Specify the type of loss criteria during training. `Default value set to DWD`'
    )
    parser.add_argument(
        "--dtype",
        default='BF16',
        choices=[
            'FP32',
            'FP16',
            'BF16'
        ],
        type=str,
        help="The data type for the model. Default is `BF16`"
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="The alpha value for the multiweighted objective dual loss function. `Default value set to 0.5`"
    )
    parser.add_argument(
        "--beta",
        default=0.5,
        type=float,
        help="The beta value for the multiweighted objective dual loss function. `Default value set to 0.5`"
    )
    parser.add_argument(
        "--rho",
        default=0.0,
        type=float,
        help="The rho value for the multiweighted objective dual loss function. `Default value set to 0.0`"
    )
    parser.add_argument(
        "--tau",
        default=0.1,
        type=float,
        help="The temperature value for NCE loss. `Default value set to 0.1`"
    )
    parser.add_argument(
        '--freeze', # Not Implemented
        action='store_true',
        help='Specify whether to freeze the Whisper backbone'
    )
    return parser.parse_args()


def load_models(config, load_name):
    if config.device == 'cuda':
        accelerator = Accelerator()
        config.device = accelerator.device
        logging.info(f"\tAcclerator using device: {config.device}")

    # Load the pre-trained JINA model
    jina = AutoModel.from_pretrained(
        config.linguistic_teacher_id,
        cache_dir=os.getenv('CACHE_DIR'),
        trust_remote_code=True
    ).to(config.device)

    # Load the pre-trained HuBERT model/processor
    hubert_processor = AutoProcessor.from_pretrained(
        config.acoustic_teacher_id,
        cache_dir=os.getenv('CACHE_DIR'),
        device_map=config.device
    )
    hubert = AutoModel.from_pretrained(
        config.acoustic_teacher_id,
        cache_dir=os.getenv('CACHE_DIR')
    ).to(config.device)

    # Load the WhiSPA and Whisper processor
    whisper_processor = AutoProcessor.from_pretrained(
        config.whisper_model_id,
        cache_dir=os.getenv('CACHE_DIR'),
        device_map=config.device
    )
    whispa = WhiSPAModel(config).to(config.device)

    if load_name:
        logging.info('Instantiating WhiSPA with loaded state dict...')
        state_dict = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), load_name, 'best.pth'))
        whispa.load_state_dict(state_dict)

    # Compile the model for better performance
    whispa = torch.compile(whispa, mode='reduce-overhead', fullgraph=True)

    return whispa, jina, hubert, whisper_processor, hubert_processor, accelerator


def load_dataset(config, whisper_processor, hubert_processor):
    dataset = AudioDataset(config, [whisper_processor, hubert_processor], dtype=config.dtype, mode='train')

    # Calculate lengths for the train/val split (80:20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation

    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    logging.info(f'\tTotal dataset size (N): {total_size}')
    logging.info(f'\tTraining dataset size (N): {train_size}')
    logging.info(f'\tValidation dataset size (N): {val_size}')

    return train_dataset, val_dataset


def save_config(save_name, config):
    if save_name:
        save_dir = os.path.join(os.getenv('CHECKPOINT_DIR'), save_name)
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, 'config.pth')
        torch.save(config, config_path)
        logging.info(f'\tConfiguration saved to: `{config_path}`')


def save_model(save_name, model, accelerator, mode='last'):
    if save_name:
        save_path = os.path.join(os.getenv('CHECKPOINT_DIR'), save_name, f'{mode}.pth')
        accelerator.save(model.state_dict(), save_path)


def train(
    train_dataset,
    val_dataset,
    whispa,
    jina,
    hubert,
    whisper_tokenizer,
    config,
    save_name
):
    # Initialize WandB for logging
    logging.info(f'\nInitializing WandB...')
    wandb.init(
        project="WhiSPAA-Training",
        config=vars(config),
        name=save_name,
        group=save_name,
        job_type='training',
        save_code=True
    )

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=collate_train
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=collate_train
    )

    scaler = torch.amp.GradScaler(config.device)
    optimizer = torch.optim.AdamW(
        whispa.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    if config.loss == 'DWD':
        loss_func = dwd_loss
    elif config.loss == 'MOW':
        loss_func = None # Not yet implemented

    jina.eval()
    train_loss = []
    val_loss = []
    best_state = None

    start_time = time.time()
    for epoch in range(config.num_epochs):
        whispa.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training"):
            # Whisper-based tokenization
            inputs = whisper_tokenizer(
                batch['message'],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(config.device)
            
            # Forward pass
            with torch.no_grad():
                # Get JINA's MEAN embedding
                jina_embs = torch.tensor(jina.module.encode(
                    batch['message'],
                    task='classification',
                    show_progress_bar=False,
                    truncate_dim=config.hidden_size
                ), dtype=torch.float32, device=config.device)

                # Get HuBERT's MEAN embedding
                hubert_embs = hubert(batch['hubert_inputs'].to(config.device)).last_hidden_state.mean(1)

            # Augment with psychological features
            psych_embs = batch['psych_emb'].to(config.device) if config.n_new_dims else None

            # Get WhiSPA's embedding
            whispa_embs = whispa(
                batch['whisper_inputs'].to(config.device),
                inputs['input_ids'],
                inputs['attention_mask']
            )

            # Apply loss functions on embeddings
            loss = loss_func(
                whispa_embs,
                jina_embs,
                hubert_embs,
                psych_embs,
                config.alpha,
                config.beta,
                config.rho,
                config.tau,
            )
            epoch_train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

            # Scale the losses and perform backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (optional but common for stability)
            torch.nn.utils.clip_grad_norm_(whispa.parameters(), max_norm=1.0)

            # Unscale gradients and perform optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        whispa.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation"):
                # Whisper-based tokenization
                inputs = whisper_tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.device)
                
                # Get JINA's MEAN embedding
                jina_embs = torch.tensor(jina.module.encode(
                    batch['message'],
                    task='classification',
                    show_progress_bar=False,
                    truncate_dim=config.hidden_size
                ), dtype=torch.float32, device=config.device)

                # Get HuBERT's MEAN embedding
                hubert_embs = hubert(batch['hubert_inputs'].to(config.device)).last_hidden_state.mean(1)

                # Augment with psychological features
                psych_embs = batch['psych_emb'].to(config.device) if config.n_new_dims else None

                # Get WhiSPA's embedding
                whispa_embs = whispa(
                    batch['whisper_inputs'].to(config.device),
                    inputs['input_ids'],
                    inputs['attention_mask']
                )

                # Apply loss functions on embeddings
                loss = loss_func(
                    whispa_embs,
                    jina_embs,
                    hubert_embs,
                    psych_embs,
                    config.alpha,
                    config.beta,
                    config.rho,
                    config.tau,
                )
                epoch_val_loss += loss.item()
                wandb.log({"train_loss": loss.item()})

        # Adjust the learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        # Handle first epoch iteration for validation loss tracking
        if epoch == 0:
            best_val_loss = avg_val_loss
            best_state = OrderedDict({name: param.clone() for name, param in whispa.state_dict().items()})
        
        # Compare validation loss to preserve the best model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = OrderedDict({name: param.clone() for name, param in whispa.state_dict().items()})
            best_path = os.path.join(os.getenv('CHECKPOINT_DIR'), save_name, 'best.pth')
            torch.save(best_state, best_path)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        epoch_elapsed_time = timedelta(seconds=time.time() - epoch_start_time)

        # Log epoch information to wandb
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_elapsed_time": epoch_elapsed_time
        })

        # Log epoch information to console
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logging.info(f"\tTraining ({config.loss}) Loss: {avg_train_loss:.4f}")
        logging.info(f"\tValidation ({config.loss}) Loss: {avg_val_loss:.4f}")
        logging.info(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"\tEpoch Elapsed Time: {epoch_elapsed_time}")
    
    total_elapsed_time = timedelta(seconds=time.time() - start_time)
    logging.info(f"\nTotal Elapsed Time: {total_elapsed_time}")
    wandb.finish()
        

def main():
    args = load_args()
    if args.save_name:
        args.save_name += f'_{time.strftime("%Y-%m-%d-%H-%M-%S")}'

    if torch.cuda.is_available():
        gpus = list(range(torch.cuda.device_count()))
        logging.info(f"\nAvailable GPU IDs: {gpus}")
        for i in gpus:
            logging.info(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
        logging.info('\n')
    else:
        logging.info("CUDA is not available. Only CPU will be used.\n")
        args.device = 'cpu'

    logging.info('Preparing Model Configuration...')
    if args.load_name:
        logging.info('\tInitializing WhiSPA Config from Load File...')
        config = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), args.load_name, 'config.pth'))
        config.shuffle = not args.no_shuffle
        if config.loss != args.loss:
            config.loss = args.loss
        if config.alpha != args.alpha:
            config.alpha = args.alpha
        if config.beta != args.beta:
            config.beta = args.beta
        if config.rho != args.rho:
            config.rho = args.rho
        if config.tau != args.tau:
            config.tau = args.tau
        if config.batch_size != args.batch_size:
            config.batch_size = args.batch_size
        if config.num_workers != args.num_workers:
            config.num_workers = args.num_workers
        if config.num_epochs != args.num_epochs:
            config.num_epochs = args.num_epochs
        if config.learning_rate != args.learning_rate:
            config.learning_rate = args.learning_rate
        if config.weight_decay != args.weight_decay:
            config.weight_decay = args.weight_decay
        if config.device != args.device:
            config.device = args.device
    else:
        dtype_choices = {
            'FP32': torch.float32,
            'FP16': torch.float16,
            'BF16': torch.bfloat16
        }
        config = WhiSPAConfig(
            whisper_model_id = args.whisper_model_id,
            pooling_mode = args.pooling_mode,
            n_new_dims= args.n_new_dims,
            loss = args.loss,
            dtype = dtype_choices[args.dtype],
            alpha = args.alpha,
            beta = args.beta,
            rho = args.rho,
            tau = args.tau,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            num_epochs = args.num_epochs,
            shuffle = not args.no_shuffle,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            device = args.device
        )

    logging.info(config)

    logging.info(f'\nSaving WhiSPA Config...')
    save_config(args.save_name, config)

    logging.info('\nLoading and Initializing Models with Config...')
    whispa, jina, hubert, whisper_processor, hubert_processor, accelerator = load_models(config, args.load_name)

    logging.info('\nPreprocessing AudioDataset...')
    train_dataset, val_dataset = load_dataset(config, whisper_processor, hubert_processor)

    logging.info('\nStarting Training...')
    train(
        train_dataset,
        val_dataset,
        whispa,
        jina,
        hubert,
        whisper_processor.tokenizer,
        config,
        args.save_name
    )

    # Save Model Checkpoint
    logging.info(f'\nSaving WhiSPA Model...')
    save_model(args.save_name, whispa, accelerator, mode='last')

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':    
    main()