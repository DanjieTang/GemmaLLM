# data_loader.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from config import *

def load_data():
    # Load word embeddings
    word_embeddings_tensor = torch.load(WORD_EMBEDDINGS_PATH).cuda(DEVICE_ID)
    word_embeddings_tensor.requires_grad = False
    num_embeddings, embedding_dim = word_embeddings_tensor.shape

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side=PADDING_SIDE)
    tokenizer.pad_token_id = PAD_TOKEN_ID

    # Load training data
    tensor = torch.load(TRAINING_DATA_PATH)

    total_data_num = tensor.shape[0]
    training_data_num = int(total_data_num * 0.98)

    training_data = tensor[:training_data_num]
    validation_data = tensor[training_data_num:]

    # Create TensorDatasets
    training_dataset = TensorDataset(training_data)
    validation_dataset = TensorDataset(validation_data)

    # Create DataLoaders
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)

    # Free up memory
    del tensor

    return (training_loader, validation_loader, word_embeddings_tensor, num_embeddings, embedding_dim, tokenizer)
