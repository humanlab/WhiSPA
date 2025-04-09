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
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs
)
import wandb
from tqdm import tqdm

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_model import WhiSPAModel
from pretrain.whispa_data import AudioDataset, collate_train
from pretrain.whispa_utils import (
    dwd_loss
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


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
        '--use_teacher_cache',
        action='store_true',
        help='Specify whether to use cached teacher embeddings'
    )
    parser.add_argument(
        '--whisper_model_id',
        default='openai/whisper-medium',
        choices=[
            'openai/whisper-medium'
        ],
        type=str,
        help='Specify the model_id of the Whisper variant on HuggingFace repository'
    )
    parser.add_argument(
        '--linguistic_teacher_id',
        default='jinaai/jina-embeddings-v3',
        choices=[
            'jinaai/jina-embeddings-v3',
            'sentence-transformers/all-roberta-large-v1',
        ],
        type=str,
        help='Specify the model_id of the language teacher model on HuggingFace repository'
    )
    parser.add_argument(
        '--acoustic_teacher_id',
        default='facebook/hubert-large-ls960-ft',
        choices=[
            'facebook/hubert-large-ls960-ft',
            'openai/whisper-medium'
        ]
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
    accelerator, whispa, processor = None, None, None
    linguistic_teacher, acoustic_teacher, acoustic_processor = None, None, None

    if config.device == 'cuda':
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        config.device = accelerator.device
        logging.info(f"  Acclerator using device: {config.device}")

    if not config.use_teacher_cache:
        # Load the pre-trained linguistic teacher model
        linguistic_teacher = AutoModel.from_pretrained(
            config.linguistic_teacher_id,
            cache_dir=os.getenv('CACHE_DIR'),
            trust_remote_code=True
        ).to(config.dtype).to(config.device)

        # Load the pre-trained acoustic model/processor
        if config.acoustic_teacher_id == config.whisper_model_id:
            acoustic_teacher = AutoModel.from_pretrained(
                config.acoustic_teacher_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).encoder.to(config.dtype).to(config.device)
        else:
            acoustic_teacher = AutoModel.from_pretrained(
                config.acoustic_teacher_id,
                cache_dir=os.getenv('CACHE_DIR')
            ).to(config.dtype).to(config.device)
            acoustic_processor = AutoProcessor.from_pretrained(
                config.acoustic_teacher_id,
                cache_dir=os.getenv('CACHE_DIR'),
                device_map=config.device
            )

    # Load the WhiSPA and Whisper processor
    processor = AutoProcessor.from_pretrained(
        config.whisper_model_id,
        cache_dir=os.getenv('CACHE_DIR'),
        device_map=config.device
    )
    whispa = WhiSPAModel(config)

    if load_name:
        logging.info('Instantiating WhiSPA with loaded state dict...')
        state_dict = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), load_name, 'best.pth'), map_location=config.device)
        whispa.load_state_dict(state_dict)

    # Compile the model for better performance
    whispa = torch.compile(whispa, mode='reduce-overhead', fullgraph=True)

    return (
        accelerator,
        whispa,
        processor,
        linguistic_teacher,
        acoustic_teacher,
        acoustic_processor
    )


def load_dataset(config, whisper_processor, hubert_processor):
    dataset = AudioDataset(config, [whisper_processor, hubert_processor], dtype=config.dtype, mode='train')

    # Calculate lengths for the train/val split (80:20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation

    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    logging.info(f'  Total dataset size (N): {total_size}')
    logging.info(f'  Training dataset size (N): {train_size}')
    logging.info(f'  Validation dataset size (N): {val_size}')

    return train_dataset, val_dataset


def save_config(save_name, config):
    if save_name:
        save_dir = os.path.join(os.getenv('CHECKPOINT_DIR'), save_name)
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, 'config.pth')
        torch.save(config, config_path)
        logging.info(f'  Configuration saved to: `{config_path}`')


def save_model(save_name, model, accelerator, mode='last'):
    if save_name:
        save_path = os.path.join(os.getenv('CHECKPOINT_DIR'), save_name, f'{mode}.pth')
        accelerator.save(model.state_dict(), save_path)


def train(
    accelerator,
    whispa,
    whisper_tokenizer,
    linguistic_teacher,
    acoustic_teacher,
    train_dataset,
    val_dataset,
    config,
    save_name
):
    # Initialize WandB for logging
    logging.info(f'Initializing WandB...')
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

    # Prepare optimizer and scheduler
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

    # Prepare model(s), optimizer, and data loaders with Accelerator
    whispa, whisper_tokenizer, optimizer, train_loader, val_loader = accelerator.prepare(
        whispa,
        whisper_tokenizer,
        optimizer,
        train_loader,
        val_loader
    )
    if not config.use_teacher_cache:
        linguistic_teacher, acoustic_teacher = accelerator.prepare(
            linguistic_teacher,
            acoustic_teacher
        )
        linguistic_teacher.eval()
        acoustic_teacher.eval()

    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(config.num_epochs):
        whispa.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0.0

        # TRAINING LOOP
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training")):
            if config.use_teacher_cache:
                linguistic_embs = None
                acoustic_embs = None
            else:
                # Get linguistic teacher's MEAN embedding
                linguistic_embs = torch.tensor(linguistic_teacher.module.encode(
                    batch['message'],
                    task='classification',
                    show_progress_bar=False,
                    truncate_dim=config.hidden_size
                ), dtype=torch.float32, device=config.device)

                # Get acoustic teacher's MEAN embedding
                acoustic_embs = acoustic_teacher(
                    batch['acoustic_inputs'].to(config.device)
                ).last_hidden_state.mean(1).to(torch.float32)
            
            # Get psychological features
            psych_embs = batch['psych_emb'].to(config.device) if config.n_new_dims else None
            
            # Whisper-based tokenization
            inputs = whisper_tokenizer(
                batch['message'],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(config.dtype).to(config.device)
            
            # Forward-pass
            whispa_embs = whispa(
                batch['audio_inputs'].to(config.device),
                inputs['input_ids'],
                inputs['attention_mask']
            )

            # Apply loss functions on embeddings
            loss = loss_func(
                whispa_embs,
                linguistic_embs,
                acoustic_embs,
                psych_embs,
                config.alpha,
                config.beta,
                config.rho,
                config.tau,
            )
            epoch_train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

            # Backward-pass
            accelerator.backward(loss)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(whispa.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        whispa.eval()
        epoch_val_loss = 0.0

        # VALIDATION LOOP
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation")):
                if config.use_teacher_cache:
                    linguistic_embs = None
                    acoustic_embs = None
                else:
                    # Get linguistic teacher's MEAN embedding
                    linguistic_embs = torch.tensor(linguistic_teacher.module.encode(
                        batch['message'],
                        task='classification',
                        show_progress_bar=False,
                        truncate_dim=config.hidden_size
                    ), dtype=torch.float32, device=config.device)

                    # Get acoustic teacher's MEAN embedding
                    acoustic_embs = acoustic_teacher(batch['acoustic_inputs'].to(config.device)).last_hidden_state.mean(1)
                
                # Get psychological features
                psych_embs = batch['psych_emb'].to(config.device) if config.n_new_dims else None

                # Whisper-based tokenization
                inputs = whisper_tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.dtype).to(config.device)

                # Forward-pass
                whispa_embs = whispa(
                    batch['audio_inputs'].to(config.device),
                    inputs['input_ids'],
                    inputs['attention_mask']
                )

                # Apply loss functions on embeddings
                loss = loss_func(
                    whispa_embs,
                    linguistic_embs,
                    acoustic_embs,
                    psych_embs,
                    config.alpha,
                    config.beta,
                    config.rho,
                    config.tau,
                )
                epoch_val_loss += loss.item()
                wandb.log({"val_loss": loss.item()})
        
        # Adjust the learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        epoch_elapsed_time = timedelta(seconds=time.time() - epoch_start_time)
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        # Log epoch information to wandb
        wandb.log({
            "epoch": epoch,
            "epoch_elapsed_time": epoch_elapsed_time,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        # Save the best and last model
        save_model(save_name, whispa, accelerator, mode='last')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(save_name, whispa, accelerator, mode='best')

        # Log epoch information to console
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logging.info(f"  Training ({config.loss}) Loss: {avg_train_loss:.4f}")
        logging.info(f"  Validation ({config.loss}) Loss: {avg_val_loss:.4f}")
        logging.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"  Epoch Elapsed Time: {epoch_elapsed_time}")
    
    total_elapsed_time = timedelta(seconds=time.time() - start_time)
    logging.info(f"Total Elapsed Time: {total_elapsed_time}")
    wandb.finish()
        

def main():
    args = load_args()
    if args.save_name:
        args.save_name += f'_{time.strftime("%Y-%m-%d-%H-%M-%S")}'

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
        config = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), args.load_name, 'config.pth'))
        config.shuffle = not args.no_shuffle
        if config.use_teacher_cache != args.use_teacher_cache:
            config.use_teacher_cache = args.use_teacher_cache
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
            linguistic_teacher_id = args.linguistic_teacher_id,
            acoustic_teacher_id = args.acoustic_teacher_id,
            use_teacher_cache = args.use_teacher_cache,
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

    logging.info(f'Saving WhiSPA Config...')
    save_config(args.save_name, config)

    logging.info('Loading and Initializing Models with Config...')
    accelerator, whispa, processor, linguistic_teacher, acoustic_teacher, acoustic_processor = load_models(config, args.load_name)

    logging.info('Preprocessing AudioDataset...')
    train_dataset, val_dataset = load_dataset(config, processor, acoustic_processor)

    logging.info('Starting Training...')
    train(
        accelerator,
        whispa,
        processor.tokenizer,
        linguistic_teacher,
        acoustic_teacher,
        train_dataset,
        val_dataset,
        config,
        args.save_name
    )

    # Save Model Checkpoint
    logging.info(f'Saving WhiSPA Model...')
    save_model(args.save_name, whispa, accelerator, mode='last')

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':    
    main()