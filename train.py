import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    WhisperProcessor,
    RobertaTokenizer,
    RobertaModel,
)
from collections import OrderedDict
from tqdm import tqdm
from pprint import pprint

from utils import (
    WhiSBERTConfig,
    WhiSBERTModel,
    AudioDataset,
    collate
)


"""
python train.py \
--num_epochs 50 \
--batch_size 200 \
--num_workers 12 \
--lr 5e-5 \
--wd 0.01 \
--save_name tbd \
--whisper_model_id openai/whisper-small \
--pooling_mode cls
"""


CACHE_DIR = '/cronus_data/rrao/cache/'
CHECKPOINT_DIR = '/cronus_data/rrao/WhisBERT/models/'


def load_args():
    parser = argparse.ArgumentParser(description='Script to train WhiSBERT model')
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
        help='Specify the filename of the model directory. After training, the last state will be saved to: `/cronus_data/rrao/WhisBERT/models/<MODEL_NAME>/last.pth`'
    )
    parser.add_argument(
        '--load_path',
        default='',
        type=str,
        help='Specify the filepath to the model state dict. Must be `.pth` file type.\t`Ex) /cronus_data/rrao/WhisBERT/models/<MODEL_NAME>/best.pth`'
    )
    # Hyperparams
    parser.add_argument(
        '--whisper_model_id',
        default='openai/whisper-small',
        choices=[
            'openai/whisper-small',
            'openai/whisper-medium',
        ],
        type=str,
        help='Specify the model_id of the Whisper variant on HuggingFace repository'
    )
    parser.add_argument(
        '--pooling_mode',
        default='cls',
        choices=[
            'cls',
            'mean'
        ],
        type=str,
        help='Specify the type of token encoding to output from WhisBERT'
    )
    parser.add_argument(
        '--loss',
        default='cos_sim',
        type=str,
        help='Specify the type of loss criteria during training'
    )
    parser.add_argument(
        '--use_new_encoder_layers',
        action='store_true',
        help='Specify whether to use the additional encoder layers for WhisBERT'
    )
    parser.add_argument(
        "--new_encoder_n_layers",
        default=12,
        type=int,
        help="The number of new encoder layers for WhisBERT"
    )
    parser.add_argument(
        "--new_encoder_n_heads",
        default=12,
        type=int,
        help="The number of new encoder attention heads for WhisBERT"
    )
    parser.add_argument(
        "--new_encoder_ffn_dim",
        default=3072,
        type=int,
        help="The number of new encoder FFN dimensions for WhisBERT"
    )
    parser.add_argument(
        '--activation_function',
        default='gelu',
        type=str,
        help='The activation function for WhisBERT [Default: `GELU`]'
    )
    parser.add_argument(
        "--eps",
        default=1e-5,
        type=float,
        help="The epsilon value for LayerNorm for WhisBERT"
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="The dropout for WhisBERT"
    )
    parser.add_argument(
        "--encoder_layerdrop",
        default=0.1,
        type=float,
        help="The encoder layer dropout for WhisBERT"
    )
    parser.add_argument(
        "--decoder_layerdrop",
        default=0.1,
        type=float,
        help="The decoder layer dropout for WhisBERT"
    )
    parser.add_argument(
        "--attention_dropout",
        default=0.1,
        type=float,
        help="The attention dropout for WhisBERT"
    )
    parser.add_argument(
        "--activation_dropout",
        default=0.1,
        type=float,
        help="The activation dropout for WhisBERT"
    )
    return parser.parse_args()


def load_models(config, load_path):
    # Load the pre-trained Whisper model and processor
    whisper_processor = WhisperProcessor.from_pretrained(config.whisper_model_id, cache_dir=CACHE_DIR, device_map=config.device)
    whisbert_model = WhiSBERTModel(config).to(config.device)

    # Load the pre-trained RoBERTa model and tokenizer
    roberta_model_id = f'FacebookAI/roberta-{("large", "base")[config.whisper_model_id.endswith("small")]}'
    roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_id, cache_dir=CACHE_DIR)
    roberta_model = RobertaModel.from_pretrained(roberta_model_id, cache_dir=CACHE_DIR).to(config.device)

    if config.device == 'cuda':
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
            print(f"\nAvailable GPU IDs: {gpus}")
            for i in gpus:
                print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
            print()
        else:
            print("CUDA is not available. Only CPU will be used.\n")
        whisbert_model = torch.nn.DataParallel(whisbert_model, device_ids=gpus)
        roberta_model = torch.nn.DataParallel(roberta_model, device_ids=gpus)

    if os.path.exists(load_path):
        print('Instantiating WhisBERT with loaded state dict...')
        state_dict = torch.load(load_path)
        try:
            whisbert_model.load_state_dict(state_dict)
        except:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            whisbert_model.load_state_dict(state_dict)

    return whisper_processor, whisbert_model, roberta_tokenizer, roberta_model


def train(
    train_dataset,
    val_dataset,
    whisper_processor,
    whisbert_model,
    roberta_tokenizer,
    roberta_model,
    config,
    save_name      
):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=config.shuffle, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=config.shuffle, collate_fn=collate)

    scaler = torch.amp.GradScaler(config.device)
    optimizer = torch.optim.AdamW(
        whisbert_model.parameters(),
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

    roberta_model.eval()
    train_loss = []
    val_loss = []
    best_state = None

    for epoch in range(config.num_epochs):
        whisbert_model.train()
        whisbert_model.module.whisper_model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training"):
            # Forward pass
            with torch.amp.autocast(config.device):
                # Roberta-based tokenization
                roberta_inputs = roberta_tokenizer(
                    batch['text'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(config.device)

                # Get Roberta CLS/MEAN token
                roberta_embs = roberta_model(**roberta_inputs).last_hidden_state[:, 0, :]

                # Whisper-based tokenization
                with torch.no_grad():
                    outputs = whisper_processor.tokenizer(
                        batch['text'],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(config.device)

                # Get WhisBERT's CLS/MEAN token
                whis_embs = whisbert_model(
                    batch['audio_inputs'],
                    outputs['input_ids'],
                    outputs['attention_mask']
                )

                if config.loss == 'cos_sim':
                    # Normalize embeddings to unit vectors for cosine similarity
                    z_audio = F.normalize(whis_embs, p=2, dim=1)
                    z_text = F.normalize(roberta_embs, p=2, dim=1)

                # Calculate contrastive loss
                similarity_matrix = torch.matmul(z_audio, z_text.T)
                positive_mask = torch.eye(whis_embs.shape[0], dtype=torch.float32).to(config.device)
                positive_loss = (positive_mask * similarity_matrix).sum()
                negative_loss = ((1 - positive_mask) * F.relu(1.0 - similarity_matrix)).sum()
                
                contrastive_loss = positive_loss + negative_loss
                epoch_train_loss += contrastive_loss.item()
                # print(contrastive_loss)

            # Scale the loss and perform backward pass
            scaler.scale(contrastive_loss).backward()

            # Gradient clipping (optional but common for stability)
            torch.nn.utils.clip_grad_norm_(whisbert_model.parameters(), max_norm=1.0)

            # Unscale gradients and perform optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        whisbert_model.eval()
        whisbert_model.module.whisper_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation"):
                # Forward pass
                with torch.amp.autocast(config.device):
                    # Roberta-based tokenization
                    roberta_inputs = roberta_tokenizer(
                        batch['text'],
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(config.device)

                    # Get Roberta CLS/MEAN token
                    roberta_embs = roberta_model(**roberta_inputs).last_hidden_state[:, 0, :]

                    # Whisper-based tokenization
                    outputs = whisper_processor.tokenizer(
                        batch['text'],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(config.device)

                    # Get WhisBERT's CLS/MEAN token
                    whis_embs = whisbert_model(
                        batch['audio_inputs'],
                        outputs['input_ids'],
                        outputs['attention_mask']
                    )

                    if config.loss == 'cos_sim':
                        # Normalize embeddings to unit vectors for cosine similarity
                        z_audio = F.normalize(whis_embs, p=2, dim=1)
                        z_text = F.normalize(roberta_embs, p=2, dim=1)

                    # Calculate contrastive loss
                    similarity_matrix = torch.matmul(z_audio, z_text.T)
                    positive_mask = torch.eye(whis_embs.shape[0], dtype=torch.float32).to(config.device)
                    positive_loss = (positive_mask * similarity_matrix).sum()
                    negative_loss = ((1 - positive_mask) * F.relu(1.0 - similarity_matrix)).sum()
                    contrastive_loss = positive_loss + negative_loss
                    epoch_val_loss += contrastive_loss.item()
                    # print(contrastive_loss)

        # Adjust the learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        # Handle first epoch iteration for validation loss tracking
        if epoch == 0:
            best_val_loss = avg_val_loss
            best_state = OrderedDict({name: param.clone() for name, param in whisbert_model.state_dict().items()})
        
        # Compare validation loss to preserve the best model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = OrderedDict({name: param.clone() for name, param in whisbert_model.state_dict().items()})

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"\tTraining Loss: {avg_train_loss:.4f}")
        print(f"\tValidation Loss: {avg_val_loss:.4f}")
        print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}")
    
    return best_state
    

def main():
    args = load_args()

    print('\nPreparing Model Configuration...')
    config = WhiSBERTConfig(
        whisper_model_id = args.whisper_model_id,
        pooling_mode = args.pooling_mode,
        loss = args.loss,
        use_new_encoder_layers = args.use_new_encoder_layers,
        new_encoder_n_layers = args.new_encoder_n_layers,
        new_encoder_n_heads = args.new_encoder_n_heads,
        new_encoder_ffn_dim = args.new_encoder_ffn_dim,
        activation_function = args.activation_function,
        eps = args.eps,
        dropout = args.dropout,
        encoder_layerdrop = args.encoder_layerdrop,
        decoder_layerdrop = args.decoder_layerdrop,
        attention_dropout = args.attention_dropout,
        activation_dropout = args.activation_dropout,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        num_epochs = args.num_epochs,
        shuffle = not args.no_shuffle,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        device = args.device
    )
    pprint(config.__dict__)

    print('\nLoading and Initializing Models with Config...')
    whisper_processor, whisbert_model, roberta_tokenizer, roberta_model = load_models(config, args.load_path)

    print('\nPreprocessing AudioDataset...')
    dataset = AudioDataset('/cronus_data/rrao/wtc_clinic/whisper_segments_transripts.csv', whisper_processor)

    # Calculate lengths for the train/val split (80:20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation
    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f'\tTotal dataset size (N): {total_size}')
    print(f'\tTraining dataset size (N): {train_size}')
    print(f'\tValidation dataset size (N): {val_size}')

    print('\nStarting Training...')
    best_state = train(
        train_dataset,
        val_dataset,
        whisper_processor,
        whisbert_model,
        roberta_tokenizer,
        roberta_model,
        config,
        args.save_name
    )

    if args.save_name:
        print(f'\nSaving `best/last` WhisBERT state_dict...')
        save_dir = os.path.join(CHECKPOINT_DIR, args.save_name)
        os.makedirs(save_dir, exist_ok=True)

        best_path = os.path.join(save_dir, 'best.pth')
        torch.save(best_state, best_path)
        print(f'\tDone.\t`{best_path}`\n')

        last_path = os.path.join(save_dir, 'last.pth')
        torch.save(whisbert_model.state_dict(), last_path)
        print(f'\tDone.\t`{last_path}`\n')

    torch.cuda.empty_cache()


if __name__ == '__main__':    
    main()