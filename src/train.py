from lazy_dataloader import prepare_dataset
from model import VLM

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

if __name__ == "__main__":
    # Hyperparameters
    max_context_length = 51
    device = "mps"
    epochs = 1
    train_batch_size = 256
    val_batch_size = 256
    weight_decay = 1e-3
    lr = 1e-3
    num_layer = 3
    head_dim = 64
    projection_dim = 512
    expansion_factor = 16
    checkpoint_filepath = ""
    q_head = 8
    kv_head = 4
    train_data_path = "languages_tokenized_50_train.npy"
    val_data_path = "languages_tokenized_50_eval.npy"

    train_loader, val_loader = prepare_dataset(train_data_path, val_data_path, train_batch_size, val_batch_size)

    vit = VLM(num_layer, max_context_length, 'word_embeddings_tensor_llama3.pt', projection_dim=projection_dim, expansion_factor=expansion_factor, use_moe=False, q_head=q_head, kv_head=kv_head, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(vit.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize the learning rate scheduler
    total_steps = epochs*len(train_loader)
    warmup_steps = int(total_steps * 0.01)

    # Warmup: LR linearly increases from 0 → base LR over warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    # Cosine annealing: after warmup
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=3e-5)

    # Combine them
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # if checkpoint_filepath != None and checkpoint_filepath != "":
    #     current_epoch = load_checkpoint(llm, optimizer, checkpoint_filepath) + 1
    # else:
    #     current_epoch = 0
    current_epoch = 0

    print("This model has", sum(p.numel() for p in vit.parameters()), "parameters.")

    loss_train = []
    loss_valid = []

    for epoch in range(current_epoch, epochs):
        loss_train_epoch = []
        loss_val_epoch = []

        vit.train()
        for data in tqdm(train_loader):
            # Teacher forcing
            input_data = data[:, :-1].long().to(device)
            target_data = data[:, 1:].long().to(device)

            # Forward pass
            prediction, load_balancing_loss = vit(input_data, [None] * input_data.shape[0])

            # Change shape for loss calculation
            prediction = prediction.view(-1, prediction.shape[-1])
            target_data = target_data.reshape(-1)

            loss = criterion(prediction, target_data) + load_balancing_loss # Calculate loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Record loss
            loss_train_epoch.append(loss.item())
            scheduler.step()

        loss_train.append(np.mean(loss_train_epoch))

        vit.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                # Teacher forcing
                input_data = data[:, :-1].long().to(device)
                target_data = data[:, 1:].long().to(device)

                # Forward pass
                prediction, load_balancing_loss = vit(input_data, [None] * input_data.shape[0])

                # Change shape for loss calculation
                prediction = prediction.view(-1, prediction.shape[-1])
                target_data = target_data.reshape(-1)

                loss = criterion(prediction, target_data) + load_balancing_loss # Calculate loss

                # Record loss
                loss_val_epoch.append(loss.item())

            loss_valid.append(np.mean(loss_val_epoch))

        # Save checkpoint
        save_checkpoint(vit, optimizer, epoch, loss_valid[-1])

        plt.plot(loss_train, label="Training loss")
        plt.plot(loss_valid, label="Validation loss")
        print("Training loss: ", loss_train[-1])
        print("Validation loss: ", loss_valid[-1])
        plt.legend()
        plt.show()