![image](https://github.com/DanjieTang/FoundationLLM/assets/37476565/1d0dfa5a-89dd-4cfd-80af-06db247f2720)

# My implementation of the Gemma LLM.

## Key insights from this implementation.

    a)RMS Normalization (Zhang and Sennrich, 2019)

    b)ROPE Embedding (Shazeer, 2019)

    c)MultiQueryAttention (Shazeer, 2019)

    d)GeGLU Activations (Shazeer, 2020)

## Training data.

    a) Extracted data from 200 thousand Wikipedia websites.

    b) 216 million pre-train tokens.

## Training detail.

    a) 2 Million parameters

    b) Contextual length of 32 tokens.

## Next step.

    a) Increase pre-train data.

    b) Increase context length.
