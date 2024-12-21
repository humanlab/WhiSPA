import os, sys
import torch
import torch.nn.functional as F

import huggingface_hub
import transformers
from transformers import AutoConfig, AutoModel, AutoProcessor

try:
    from transformers.utils import logging
except ImportError:
    print("Warning: Unable to importing transformers.utils logging")


# COPIED FROM TEXT PACKAGE
def set_logging_level(logging_level):
    """
    Set the logging level

    Parameters
    ----------
    logging_level : str
        set logging level, options: critical, error, warning, info, debug
    """
    logging_level = logging_level.lower()
    # default level is warning, which is in between "error" and "info"
    if logging_level in ['warn', 'warning']:
        logging.set_verbosity_warning()
    elif logging_level == "critical":
        logging.set_verbosity(50)
    elif logging_level == "error":
        logging.set_verbosity_error()
    elif logging_level == "info":
        logging.set_verbosity_info()
    elif logging_level == "debug":
        logging.set_verbosity_debug()
    else:
        print("Warning: Logging level {l} is not an option.".format(l=logging_level))
        print("\tUse one of: critical, error, warning, info, debug")


# COPIED FROM TEXT PACKAGE
def set_hg_gated_access(access_token):
    """
    Local save of the access token for gated models on hg.
    
    Parameters
    ----------
    access_token : str
        Steps to get the access_token:
        1. Log in to your Hugging Face account.
        2. Click on your profile picture in the top right corner.
        3. Select ‘Settings’ from the dropdown menu.
        4. In the settings, you’ll find an option to generate a new token.
        Or, visit URL: https://huggingface.co/settings/tokens
    """
    huggingface_hub.login(access_token)
    print("Successfully login to Huggingface!")


# COPIED FROM TEXT PACKAGE
def del_hg_gated_access():
    """
    Remove the access_token saved locally.

    """
    huggingface_hub.logout()
    print("Successfully logout to Huggingface!")


# COPIED FROM TEXT PACKAGE
def get_device(device):
    """
    Get device and device number

    Parameters
    ----------
    device : str
        name of device: 'cpu', 'gpu', 'cuda', 'mps', or of the form 'gpu:k', 'cuda:k', or 'mps:0'
        where k is a specific device number

    Returns
    -------
    device : str
        final selected device name
    device_num : int
        device number, -1 for CPU
    """
    device = device.lower()
    if not device.startswith('cpu') and not device.startswith('gpu') and not device.startswith('cuda') and not device.startswith('mps'):
        print("device must be 'cpu', 'gpu', 'cuda', 'mps', or of the form 'gpu:k', 'cuda:k', or 'mps:0'")
        print("\twhere k is an integer value for the device")
        print("Trying CPUs")
        device = 'cpu'
    
    device_num = -1
    if device != 'cpu':
        attached = False
        
        if hasattr(torch.backends, "mps"):
            mps_available = torch.backends.mps.is_available()
        else:
            mps_available = False
        print(f"MPS for Mac available: {mps_available}")
        if torch.cuda.is_available():
            if device == 'gpu' or device == 'cuda': 
                # assign to first gpu device number
                device = 'cuda'
                device_num = list(range(torch.cuda.device_count()))[0]
                attached = True
            elif 'gpu:' in device or 'cuda:' in device:
                try:
                    device_num = int(device.split(":")[-1])
                    device = 'cuda:' + str(device_num)
                    attached = True
                except:
                    attached = False
                    print(f"Device number {str(device_num)} does not exist! Use 'device = gpus' to see available gpu numbers.")
            elif 'gpus' in device:
                device = 'cuda'
                device_num = list(range(torch.cuda.device_count()))
                device = [device + ':' + str(num1) for num1 in device_num]
                attached = True
                print(f"Running on {str(len(device))} GPUs!")
                print(f"Available gpus to set: \n {device}")
        elif "mps" in device:
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not built with MPS enabled.")
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            else:
                device_num = 0 # list(range(torch.cuda.device_count()))[0]
                device = 'mps:' + str(device_num)
                attached = True
                print("Using Metal Performance Shaders (MPS) backend for GPU training acceleration!")
        else:
            attached = False
        if not attached:
            print("Unable to use MPS (Mac M1+), CUDA (GPU), using CPU")
            device = "cpu"
            device_num = -1

    return device, device_num


# COPIED FROM TEXT PACKAGE
def set_tokenizer_parallelism(tokenizer_parallelism):
    if tokenizer_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


# INSPIRED FROM TEXT PACKAGE
def get_audio_model(model, processor_only=False, config_only=False, hg_gated=False, hg_token="", trust_remote_code=False):
    """
    Get audio model and tokenizer from model string

    Parameters
    ----------
    model : str
        shortcut name for Hugging Face pretained model
        Full list https://huggingface.co/transformers/pretrained_models.html
    hg_gated : bool
        Set to True if the model is gated
    hg_token: str
        The token to access the gated model got in huggingface website
    
    Returns
    -------
    config
    processor
    model
    """
    if hg_gated:
        set_hg_gated_access(access_token=hg_token)
    else: 
        pass
    config = AutoConfig.from_pretrained(model)
    if not config_only:
        processor = AutoProcessor.from_pretrained(model)
        transformer_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=trust_remote_code)
            
    if config_only:
        return config
    elif processor_only:
        return processor
    else:     
        return config, processor, transformer_model


# INSPIRED FROM TEXT PACKAGE
def get_audio_embeddings(
    audio_filepaths,
    model = 'whisper-tiny',
    layers = 'all',
    return_tokens = True,
    device = 'cpu',
    tokenizer_parallelism = False,
    model_max_length = None,
    hg_gated = False,
    hg_token = "",
    trust_remote_code = False,
    logging_level = 'warning',
    sentence_tokenize = True
):
    """
    Simple Python method for embedding speech with pretained Hugging Face models

    Parameters
    ----------
    audio_filepaths : list
        list of audio filepaths, each is embedded separately
    model : str
        shortcut name for Hugging Face pretained model
        Full list https://huggingface.co/transformers/pretrained_models.html
    layers : str or list
        'all' or an integer list of layers to keep
    max_token_to_sentence : int
        maximum number of tokens in a string to handle before switching to embedding text
        sentence by sentence
    device : str
        name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
    tokenizer_parallelism :  bool
        Whether to use device parallelization during tokenization
    model_max_length : int
        maximum length of the tokenized text
    hg_gated : bool
        Whether the accessed model is gated
    hg_token: str
        The token generated in huggingface website
    trust_remote_code : bool
        use a model with custom code on the Huggingface Hub
    logging_level : str
        set logging level, options: critical, error, warning, info, debug

    Returns
    -------
    all_embs : list
        embeddings for each item in text_strings
    all_toks : list, optional
        tokenized version of text_strings
    """
    set_logging_level(logging_level)
    set_tokenizer_parallelism(tokenizer_parallelism)
    device, device_num = get_device(device)

    config, processor, transformer_model = get_audio_model(model, hg_gated=hg_gated, hg_token=hg_token, trust_remote_code=trust_remote_code)

    if device != 'cpu':
        transformer_model.to(device)
    transformer_model.eval()

    # check and adjust input types
    if not isinstance(audio_filepaths, list):
        audio_filepaths = [audio_filepaths]

    if layers != 'all':
        if not isinstance(layers, list):
            layers = [layers]
        layers = [int(i) for i in layers]

    all_embs = []
    all_toks = []


    for audio_filepath in audio_filepaths:
        waveform = preprocess_audio(audio_filepath)
        audio_inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        
        if device != 'cpu':
            audio_inputs = audio_inputs.to(device)

        try:
            with torch.no_grad():
                # Wav2Vec Embedding Generation
                if isinstance(transformer_model, transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model):
                    embedding = transformer_model(**audio_inputs).last_hidden_state.mean(1).squeeze()

                # Whisper Embedding Generation (Encoder Representation)
                elif isinstance(transformer_model, transformers.models.whisper.modeling_whisper.WhisperModel):
                    embedding = transformer_model.encoder(**audio_inputs).last_hidden_state.mean(1).squeeze()
                
                else:
                    raise AssertionError('Not implemented yet...')

            all_embs.append(embedding)
        except Exception as e:
            print(f'\"{audio_filepath}\" failed with the following error:')
            print(Warning(e))

    if hg_gated:
        del_hg_gated_access()
    if return_tokens:
        return all_embs, all_toks
    else:
        return all_embs


def preprocess_audio(audio_path):
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    # Convert stereo (or multi-channel) to mono if needed   
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform


def mean_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



if __name__ == '__main__':
    embs, toks = get_audio_embeddings(
        '/cronus_data/rrao/Audio_Segments/WTC/0729-converted - 0011.wav',
        model = 'openai/whisper-tiny', # facebook/wav2vec2-base-960h
        layers = 'all',
        return_tokens = True,
        device = 'cpu',
        tokenizer_parallelism = False,
        model_max_length = None,
        hg_gated = False,
        hg_token = "",
        trust_remote_code = False,
        logging_level = 'warning',
        sentence_tokenize = True
    )

    print(embs[0].shape)