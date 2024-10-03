# config.py

import torch

# Set random seed for reproducibility
torch.manual_seed(0)

# Hyperparameters
MAX_TOKEN = 64
DEVICE_ID = 1  # GPU device id
EPOCHS = 3
BATCH_SIZE = 300
VALIDATION_BATCH_SIZE = 10
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3
NUM_LAYER = 4
HEAD_DIM = 64
PROJECTION_DIM = 1024
EXPANSION_FACTOR = 4
CHECKPOINT_FILEPATH = ""

# Model and tokenizer settings
MODEL_ID = "NousResearch/Meta-Llama-3-8B"
PADDING_SIDE = "right"
PAD_TOKEN_ID = 128002

# Paths to data files
WORD_EMBEDDINGS_PATH = 'word_embeddings_tensor.pt'
TRAINING_DATA_PATH = 'llama3_wiki_64.pt'
