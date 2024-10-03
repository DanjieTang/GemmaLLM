# generate.py

import torch
import torch.nn.functional as F
from torch.cuda import amp
from config import *
from data_loader import load_data
from model import Gemma

def generate_text(prompt: str, max_length: int = MAX_TOKEN, temperature: float = 0.1):
    # Load data and embeddings
    (
        _,
        _,
        word_embeddings_tensor,
        num_embeddings,
        embedding_dim,
        tokenizer,
    ) = load_data()

    # Initialize model
    gemma = Gemma(
        NUM_LAYER,
        num_embeddings,
        MAX_TOKEN,
        embedding_dim,
        embedding_dim * EXPANSION_FACTOR,
        HEAD_DIM,
        projection_dim=PROJECTION_DIM,
    ).cuda(DEVICE_ID)
    gemma.load_state_dict(torch.load('gemma_model.pth'))
    gemma.eval()

    tokenized_sentence = tokenizer(prompt)["input_ids"]
    if tokenized_sentence[-1] == tokenizer.eos_token_id:
        tokenized_sentence = tokenized_sentence[:-1]

    with torch.no_grad():
        while (
            tokenized_sentence[-1] != tokenizer.eos_token_id and len(tokenized_sentence) < max_length
        ):
            tokenized_sentence_tensor = torch.tensor(tokenized_sentence).unsqueeze(0).cuda(DEVICE_ID)
            sentence_embedding = word_embeddings_tensor[tokenized_sentence_tensor]

            with amp.autocast():
                prediction = gemma(sentence_embedding)

            prediction = prediction[0, -1]
            prediction = prediction / temperature
            prediction = F.softmax(prediction, dim=-1)
            output_token = torch.multinomial(prediction, 1)

            tokenized_sentence.append(output_token.item())

    generated_text = tokenizer.decode(tokenized_sentence, skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "arXiv is an open-access"
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)
