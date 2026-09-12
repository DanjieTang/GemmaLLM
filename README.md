# My implementation of the Gemma LLM.

## Model structure

The PyTorch architecture lives in the `model/` package:

| File | Component |
| --- | --- |
| `rope.py` | `ROPEEmbedding`: rotary positional embeddings |
| `attention.py` | `Attention`: gated attention, grouped KV heads, and LoRA |
| `feed_forward.py` | `FeedForward`: gated feed-forward network and LoRA |
| `moe.py` | `MOE`: expert routing and load-balancing loss |
| `llm_layer.py` | `LLMLayer`: attention and feed-forward decoder block |
| `llm.py` | `LLM`: decoder stack and vocabulary classifier |
| `vlm.py` | `VLM`: CLIP image encoding and text/image fusion |
| `cache.py` | `KVCache`, `PastKeyValues`, and `VLMCache`: shared cache types |

## Training data.

    a) Text data: All English Wikipedia 6.5 million pages(~2 billion tokens.).

    b) Multimodal data: COCO 2017, Open Images V7

## Throughput optimizations

    a) KV cache

    b) MTP Speculative decoding

    c) CUDA cublas implementation

    d) Flash attention

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

    j)FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Training detail.

    a) 199.7 Million parameters Mixture of Experts Architecture

    b) Contextual length of 256 tokens.
