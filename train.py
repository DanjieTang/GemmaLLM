# train.py

import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from data_loader import load_data
from model import Gemma
from utils import save_checkpoint, load_checkpoint
import os

def train():
    # Load data and embeddings
    (
        training_loader,
        validation_loader,
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

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(gemma.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(training_loader), eta_min=1e-5
    )

    # Load checkpoint if exists
    if CHECKPOINT_FILEPATH and os.path.exists(CHECKPOINT_FILEPATH):
        current_epoch = load_checkpoint(gemma, optimizer, CHECKPOINT_FILEPATH) + 1
    else:
        current_epoch = 0

    print("This model has", sum(p.numel() for p in gemma.parameters()), "parameters.")
    scaler = amp.GradScaler()

    loss_train = []
    loss_valid = []

    for epoch in range(current_epoch, EPOCHS):
        loss_train_epoch = []
        loss_val_epoch = []

        gemma.train()
        for data in tqdm(training_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            input_data = data[0][:, :-1].cuda(DEVICE_ID)
            target_data = data[0][:, 1:].cuda(DEVICE_ID)

            input_embeddings = word_embeddings_tensor[input_data]

            with amp.autocast():
                prediction = gemma(input_embeddings)

                prediction = prediction.view(-1, num_embeddings)
                target_data = target_data.reshape(-1)

                mask = target_data != tokenizer.pad_token_id
                prediction = prediction[mask]
                target_data = target_data[mask]

                loss = criterion(prediction, target_data)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(gemma.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_train_epoch.append(loss.item())
            scheduler.step()

        loss_train.append(np.mean(loss_train_epoch))

        gemma.eval()
        with torch.no_grad():
            for data in tqdm(validation_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}"):
                input_data = data[0][:, :-1].cuda(DEVICE_ID)
                target_data = data[0][:, 1:].cuda(DEVICE_ID)

                input_embeddings = word_embeddings_tensor[input_data]

                with amp.autocast():
                    prediction = gemma(input_embeddings)

                    prediction = prediction.view(-1, num_embeddings)
                    target_data = target_data.reshape(-1)

                    mask = target_data != tokenizer.pad_token_id
                    prediction = prediction[mask]
                    target_data = target_data[mask]

                    loss = criterion(prediction, target_data)

                loss_val_epoch.append(loss.item())

            loss_valid.append(np.mean(loss_val_epoch))

        save_checkpoint(gemma, optimizer, epoch, loss_valid[-1])

        plt.figure(figsize=(10, 5))
        plt.plot(loss_train, label="Training loss")
        plt.plot(loss_valid, label="Validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'loss_plot_epoch_{epoch+1}.png')
        plt.close()
        print(f"Training loss: {loss_train[-1]:.4f}")
        print(f"Validation loss: {loss_valid[-1]:.4f}")

    torch.save(gemma.state_dict(), 'gemma_model.pth')
    print("Training completed and model saved as 'gemma_model.pth'.")

if __name__ == "__main__":
    train()
