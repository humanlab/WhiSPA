# WhiSPA: Whisper Semantically and Psychologically Aligned

This is the code repository for the [WhiSPA paper](https:google.com).

![WhiSPA Spirit]("visuals/WhiSPA_Spirit_Figure.png")

## Abstract
Current speech encoding pipelines often rely on separate processing pipelines between text and audio, not fully leveraging the inherent overlap between these modalities for understanding human communication. Language models excel at capturing semantic meaning from text that can complement the additional prosodic, emotional, and acoustic cues from speech. This work bridges the gap by proposing WhiSPA (Whisper with Semantic-Psychological Alignment), a novel audio encoder trained with a contrastive student-teacher learning objective. Using over 500k speech segments from mental health audio interviews, we evaluate the utility of aligning Whisper’s audio embeddings with text representations from an SBERT encoder and text-based assessments of psychological dimensions: emotion and personality. Over self-supervised and downstream mental health tasks, WhiSPA surpasses state-of-the-art speech models, achieving an average error reduction of 73.4% on the segment-level self-supervised objective and 83.8% on 11 psychological downstream tasks. WhiSPA demonstrates that cross-modal alignment can increase the amount of text-semantic and psychological information captured in audio-only encoder models.

## Getting started
