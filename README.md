![image](https://github.com/DanjieTang/FoundationLLM/assets/37476565/1d0dfa5a-89dd-4cfd-80af-06db247f2720)

# My implementation of the Gemma LLM.

## Training data.

    a) All English Wikipedia 6.5 million pages(~2 billion tokens.).

    b) COCO 2017

    c) Open Images V7

## Throughput optimizations

    a) KV cache

    b) MTP Speculative decoding

Original speed = 28.09 tokens/s

Latest speed = 153.82 tokens/s

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

    a) 199.7 Million parameters Mixture of Experts Architecture

    b) Contextual length of 256 tokens.
