# WhiSPA: Whisper Semantically and Psychologically Aligned

<img src="visuals/WhiSPA_Spirit_Figure.jpg" alt="WhiSPA Spirit Figure" width="75%"/>

This is the code repository for the [WhiSPA paper](https://aclanthology.org/2025.acl-long.1098.pdf).

## Table of Contents

1. [Introduction](#intro)
2. [Quickstart](#quick)
3. [Inference](#inference)
4. [Pretraining](#pretrain)

<a id="intro"></a>

## Introduction

WhiSPA (Whisper with Semantic-Psychological Alignment) is a novel speech encoder that leverages the Whisper model as a backbone and aligns its audio embeddings with text representations from SBERT and psychological embeddings. This alignment is achieved through a contrastive student-teacher learning objective, using hundreds of thousands of audio segments from mental health interviews. WhiSPA aims to capture both semantic and psychological information in audio-only encoder models, surpassing state-of-the-art speech models in various tasks.

<a id="quick"></a>

## Quickstart

Clone the repo
```bash
git clone https://github.com/humanlab/WhiSPA.git
```

Create a conda environment
```bash
conda env create -f environment.yaml
conda activate whispa
```

Use the model
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pretrain.whispa_model import WhiSPAModel
from inference.encode import encode

processor = WhisperProcessor.from_pretrained('openai/whisper-small')
whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
whispa = WhiSPAModel.from_pretrained('Jarhatz/WhiSPA-V1-Small')

audio_paths = [
      '/path/to/audio/file.wav',
      '/path/to/audio/file.mp3',
      '/path/to/audio/file.m4a',
]

audio_embeddings = encode(whispa, whisper, processor, audio_paths)
for name, embedding in audio_embeddings.items():
      print(f'audio: {name}   emb: {embedding.shape}')
```

<a id="inference"></a>

## Inference

We have two model checkpoints on HuggingFace which are [Jarhatz/WhiSPA-V1-Tiny](https://huggingface.co/Jarhatz/WhiSPA-V1-Tiny) and [Jarhatz/WhiSPA-V1-Small](https://huggingface.co/Jarhatz/WhiSPA-V1-Small). Depending on your use-case and compute/memory constraints, use whichever ones. You can run inference on a directory of audio files or a singular audio file using the `encode.py` script.

```bash
python inference/encode.py \
--model_id Jarhatz/WhiSPA-V1-Small \
--audio_path <AUDIO_FILE_PATH or AUDIO_DIR_PATH> \
--output_path <OUTPUT_DIR_PATH> \
--device cuda
```

<a id="pretrain"></a>

## Pretraining WhiSPA

```bash
python pretrain/whispa_train.py \
--whisper_model_id openai/whisper-tiny \
--with_bidirectionality \
--loss CS \
--num_epochs 50 \
--batch_size 700 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name <MODEL_SAVE_NAME>
```

### Training Procedure

![WhiSPA Training Architecture](visuals/WhiSPA_Training_Procedure.jpg)

WhiSPA is trained using a student-teacher contrastive alignment approach. The Whisper model (student) is aligned with SBERT and psychological embeddings (teacher) to increase the cosine similarity between their embeddings. This alignment helps WhiSPA capture both semantic and psychological information in the audio embeddings.

_\*Note: .wav, .mp3, and .m4a are known to be supported with our pipeline._

## Citation
Please cite our work if you choose to use it. We appreciate it.
```bash
@misc{rao2025whispasemanticallypsychologicallyaligned,
      title={WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning}, 
      author={Rajath Rao and Adithya Ganesan and Oscar Kjell and Jonah Luby and Akshay Raghavan and Scott Feltman and Whitney Ringwald and Ryan L. Boyd and Benjamin Luft and Camilo Ruggero and Neville Ryant and Roman Kotov and H. Andrew Schwartz},
      year={2025},
      eprint={2501.16344},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2501.16344}, 
}
```
