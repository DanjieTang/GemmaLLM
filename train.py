from lazy_dataloader import prepare_dataset
from model import VLM

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

import wandb

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VLM Model")

    # Dataset & Paths
    parser.add_argument("--train_path", type=str, default="languages_tokenized_50_train.npy")
    parser.add_argument("--val_path", type=str, default="languages_tokenized_50_eval.npy")
    parser.add_argument("--embeddings_path", type=str, default="word_embeddings_tensor_llama3.pt")
    
    # Model Architecture
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--max_context_length", type=int, default=51)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--q_head", type=int, default=8)
    parser.add_argument("--kv_head", type=int, default=4)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # WandB
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(0)

    # Use wandb if applicable
    use_wandb = args.project is not None and args.entity is not None

    if use_wandb:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            config={
                "num_layer": args.num_layer,
                "max_context_length": args.max_context_length,
                "projection_dim": args.projection_dim,
                "expansion_factor": args.expansion_factor,
                "q_head": args.q_head,
                "kv_head": args.kv_head,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        )

    train_loader, val_loader = prepare_dataset(args.train_path, args.val_path, args.batch_size, args.batch_size)

    model = VLM(
        num_layer=args.num_layer,
        max_context_length=args.max_context_length,
        word_embeddings_tensor=args.embeddings_path,
        projection_dim=args.projection_dim,
        expansion_factor=args.expansion_factor,
        use_moe=False,
        q_head=args.q_head,
        kv_head=args.kv_head,
        device=args.device
    ).to(args.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Schedulers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.01)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=3e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Tracking metrics
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = []
        counter = 0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            counter += 1
            if counter == 100:
                break
            # Teacher forcing
            input_data = data[:, :-1].long().to(args.device)
            target_data = data[:, 1:].long().to(args.device)

            # Forward pass
            prediction, load_balancing_loss = model(input_data, [None] * input_data.shape[0])

            # Change shape for loss calculation
            prediction = prediction.view(-1, prediction.shape[-1])
            target_data = target_data.reshape(-1)
            loss = criterion(prediction, target_data) + load_balancing_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Record loss
            epoch_train_loss.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validating"):
                counter += 1
                if counter == 200:
                    break
                # Teacher forcing
                input_data = data[:, :-1].long().to(args.device)
                target_data = data[:, 1:].long().to(args.device)

                # Forward pass
                prediction, load_balancing_loss = model(input_data, [None] * input_data.shape[0])

                # Change shape for loss calculation
                prediction = prediction.view(-1, prediction.shape[-1])
                target_data = target_data.reshape(-1)
                loss = criterion(prediction, target_data) + load_balancing_loss # Calculate loss

                # Record loss
                epoch_val_loss.append(loss.item())

        avg_val_loss = np.mean(epoch_val_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        if use_wandb:
            run.log({"Training Loss": train_losses[-1], "Val loss": val_losses[-1]})

    if not use_wandb:
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        print("Training loss: ", train_losses[-1])
        print("Validation loss: ", val_losses[-1])
        plt.legend()
        plt.show()

    if use_wandb:
        run.finish()

if __name__ == "__main__":
    main()