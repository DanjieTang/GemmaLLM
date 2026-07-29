![image](https://github.com/DanjieTang/FoundationLLM/assets/37476565/1d0dfa5a-89dd-4cfd-80af-06db247f2720)

# My implementation of the Gemma LLM.

## Training data.

    a) All English Wikipedia pages(6.5 million).

    b) ~2 billion tokens.

## Key insights from this implementation.

    a)RMS Normalization

    b)ROPE Embedding

    c)MultiQueryAttention

    d)GeGLU Activations

    e)Pre-Norm Transformers

    f)Mixtral of Experts

    g)LoRA: Low-Rank Adaptation of Large Language Models

    h)Gated Attention for Large Language Models

    i)Late fusion for LLM image capability

## Training detail.

    a) 665 Million parameters Mixture of Experts Architecture

    b) Contextual length of 64 tokens.

## Image inputs

`VLM.forward` uses the first hidden state from CLIP ViT—the CLS token—as
one image token. The model input is arranged as:

    [CLIP CLS] [learned separator] [text tokens]

For multimodal training, pass `--train_image_paths` and
`--val_image_paths` to `train.py`. Each file must be a one-dimensional
NumPy string array with one image path per tokenized sample. Paths may be
absolute or relative to the image-path array. Use an empty string for a
text-only sample.

Run `python -m pytest` to verify CLS-token fusion and image-path loading
without downloading CLIP weights.
