import sys, os
# Add the root directory of the project to the Python path
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(BASE_DIR))

import time
from datetime import timedelta
from collections import OrderedDict
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    WhisperProcessor
)
from tqdm import tqdm
from matplotlib import pyplot as plt
from pprint import pprint

from pretrain.whispa_config import WhiSPAConfig
from pretrain.whispa_model import WhiSPAModel
from pretrain.whispa_data import AudioDataset, collate_train
from pretrain.whispa_utils import (
    mean_pooling,
    cos_sim_loss,
    nce_cont_loss,
    mow_loss,
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
        default='openai/whisper-small',
        choices=[
            'openai/whisper-tiny',
            'openai/whisper-small'
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
        '--use_psych',
        action='store_true',
        help='Specify whether to use psychological features during alignment'
    )
    parser.add_argument(
        '--loss',
        default='NCE',
        choices=[
            'CS',
            'NCE',
            'MOW',
        ],
        type=str,
        help='Specify the type of loss criteria during training'
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
    # Load the Whisper audio processor
    processor = WhisperProcessor.from_pretrained(
        config.whisper_model_id,
        cache_dir=os.getenv('CACHE_DIR'),
        device_map=config.device
    )

    # Load WhiSPA and Whisper Encoder
    whispa = WhiSPAModel(config).to(config.device)
    whisper = AutoModel.from_pretrained(
        'openai/whisper-tiny',
        cache_dir=os.getenv('CACHE_DIR')
    ).encoder.to(config.device)

    # Load the pre-trained SentenceTransformer model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L12-v2',
        cache_dir=os.getenv('CACHE_DIR'),
        TOKENIZERS_PARALLELISM=False
    )
    sbert = AutoModel.from_pretrained(
        'sentence-transformers/all-MiniLM-L12-v2',
        cache_dir=os.getenv('CACHE_DIR')
    ).to(config.device)

    if config.device == 'cuda':
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
            print(f"\nAvailable GPU IDs: {gpus}")
            for i in gpus:
                print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
            print()
            whispa = torch.nn.DataParallel(whispa, device_ids=gpus)
            whisper = torch.nn.DataParallel(whisper, device_ids=gpus)
            sbert = torch.nn.DataParallel(sbert, device_ids=gpus)
        else:
            print("CUDA is not available. Only CPU will be used.\n")

    if load_name:
        print('Instantiating WhiSPA with loaded state dict...')
        state_dict = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), load_name, 'best.pth'))
        try:
            whispa.load_state_dict(state_dict)
        except:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            whispa.load_state_dict(state_dict)

    return whispa, whisper, sbert, processor, tokenizer


def plot_loss(train_loss, val_loss, save_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'r-', label='Validation Loss')
    plt.title(f'Training vs Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.join(BASE_DIR, 'loss'), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, f'loss/{save_name}.png'), format='png')


def train(
    train_dataset,
    val_dataset,
    whispa,
    whisper,
    sbert,
    processor,
    tokenizer,
    config,
    use_psych,
    save_name
):
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

    if config.loss == 'CS':
        loss_func = cos_sim_loss

    elif config.loss == 'NCE':
        loss_func = nce_cont_loss

    elif config.loss == 'MWO':
        loss_func = None # Not implemented yet

    whisper.eval()
    sbert.eval()
    train_loss = []
    val_loss = []
    best_state = None

    start_time = time.time()
    for epoch in range(config.num_epochs):
        whispa.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training"):

            # SBERT-based tokenization
            sbert_inputs = tokenizer(
                batch['message'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(config.device)

            # Whisper-based tokenization
            inputs = processor.tokenizer(
                batch['message'],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(config.device)
            
            # Forward pass
            with torch.amp.autocast(config.device):

                with torch.no_grad():
                    # Get SBERT's MEAN embedding
                    sbert_embs = sbert(**sbert_inputs).last_hidden_state
                    sbert_embs = mean_pooling(sbert_embs, sbert_inputs['attention_mask'])
                    # Get Whisper's MEAN embedding
                    whisper_embs = whisper(batch['audio_inputs'].to(config.device)).last_hidden_state
                    whisper_embs = torch.mean(whisper_embs, dim=1)
                    # Concatenate both embeddings
                    target_embs = torch.cat([whisper_embs, sbert_embs], dim=1)
                
                # Augment with psychological features
                if use_psych:
                    psych_emb = batch['psych_emb'].to(config.device)
                    if config.n_new_dims:
                        # Concatenation
                        target_embs = torch.cat([target_embs, psych_emb], dim=1)
                    else:
                        # Replacement
                        target_embs[:, :10] = psych_emb

                # Get WhiSPA's embedding
                whispa_embs = whispa(
                    batch['audio_inputs'].to(config.device),
                    inputs['input_ids'],
                    inputs['attention_mask']
                )

                # Apply loss functions on embeddings
                loss = loss_func(whispa_embs, target_embs)
                epoch_train_loss += loss.item()

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
                
                # SBERT-based tokenization
                sbert_inputs = tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(config.device)

                # Whisper-based tokenization
                inputs = processor.tokenizer(
                    batch['message'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(config.device)
                
                # Forward pass
                with torch.amp.autocast(config.device):

                    # Get SBERT's MEAN embedding
                    sbert_embs = sbert(**sbert_inputs).last_hidden_state
                    sbert_embs = mean_pooling(sbert_embs, sbert_inputs['attention_mask'])
                    # Get Whisper's MEAN embedding
                    whisper_embs = whisper(batch['audio_inputs'].to(config.device)).last_hidden_state
                    whisper_embs = torch.mean(whisper_embs, dim=1)
                    # Concatenate both embeddings
                    target_embs = torch.cat([whisper_embs, sbert_embs], dim=1)
                
                    # Augment with psychological features
                    if use_psych:
                        psych_emb = batch['psych_emb'].to(config.device)
                        if config.n_new_dims:
                            # Concatenation
                            target_embs = torch.cat([target_embs, psych_emb], dim=1)
                        else:
                            # Replacement
                            target_embs[:, :10] = psych_emb

                    # Get WhiSPA's embedding
                    whispa_embs = whispa(
                        batch['audio_inputs'].to(config.device),
                        inputs['input_ids'],
                        inputs['attention_mask']
                    )

                    # Apply loss functions on embeddings
                    loss = loss_func(whispa_embs, target_embs)
                    epoch_val_loss += loss.item()

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

        # Plot and save loss curves
        plot_loss(train_loss, val_loss, save_name)

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"\tTraining ({config.loss}) Loss: {avg_train_loss:.4f}")
        print(f"\tValidation ({config.loss}) Loss: {avg_val_loss:.4f}")
        print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"\tEpoch Elapsed Time: {epoch_elapsed_time}")
    
    total_elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"\nTotal Elapsed Time: {total_elapsed_time}")
        

def main():
    args = load_args()

    print('Preparing Model Configuration...')
    if args.load_name:
        print('\tInitializing WhiSPA Config from Load File...')
        config = torch.load(os.path.join(os.getenv('CHECKPOINT_DIR'), args.load_name, 'config.pth'))
        config.shuffle = not args.no_shuffle
        if config.loss != args.loss:
            config.loss = args.loss
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
        if args.n_new_dims:
            args.use_psych = True
        config = WhiSPAConfig(
            whisper_model_id = args.whisper_model_id,
            pooling_mode = args.pooling_mode,
            n_new_dims= args.n_new_dims,
            loss = args.loss,
            tau = args.tau,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            num_epochs = args.num_epochs,
            shuffle = not args.no_shuffle,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            device = args.device
        )

    print(config)

    if args.save_name:
        print(f'\nSaving WhiSPA Config...')
        save_dir = os.path.join(os.getenv('CHECKPOINT_DIR'), args.save_name)
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, 'config.pth')
        torch.save(config, config_path)

    print('\nLoading and Initializing Models with Config...')
    whispa, whisper, sbert, processor, tokenizer = load_models(config, args.load_name)

    print('\nPreprocessing AudioDataset...')
    dataset = AudioDataset(config, processor, args.use_psych, mode='train')

    # Calculate lengths for the train/val split (80:20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation
    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f'\tTotal dataset size (N): {total_size}')
    print(f'\tTraining dataset size (N): {train_size}')
    print(f'\tValidation dataset size (N): {val_size}')

    if args.save_name and os.path.exists(args.save_name):
        print(f'WARNING: Overwriting existing model directory!')
        print(f'\t"{args.save_name}" already exists in "{os.getenv("CHECKPOINT_DIR")}"')

    print('\nStarting Training...')
    train(
        train_dataset,
        val_dataset,
        whispa,
        whisper,
        sbert,
        processor,
        tokenizer,
        config,
        args.use_psych,
        args.save_name
    )

    if args.save_name:
        print(f'\nSaving WhiSPA Model...')
        save_dir = os.path.join(os.getenv('CHECKPOINT_DIR'), args.save_name)
        best_path = os.path.join(save_dir, 'best.pth')
        last_path = os.path.join(save_dir, 'last.pth')
        torch.save(whispa.state_dict(), last_path)
        print(f'\tDone.\t`{best_path}`\n')
        print(f'\tDone.\t`{last_path}`\n')

    torch.cuda.empty_cache()


if __name__ == '__main__':    
    main()