"""Train the custom CLIP-conditioned annotation decoder or legacy token arrays."""

import argparse
from itertools import islice
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

from lazy_dataloader import prepare_annotation_dataset, prepare_dataset
from model import VLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Enable paired-file training under this directory.")
    parser.add_argument("--train_folders", nargs="+", default=["train", "OpenImageV7_train"])
    parser.add_argument("--val_folders", nargs="+", default=["val", "OpenImageV7_val"])
    parser.add_argument("--tokenizer_path", default=str(Path.home() / "models/gemma-4-E2B-it"))
    parser.add_argument("--train_path", default="languages_tokenized_50_train.npy")
    parser.add_argument("--val_path", default="languages_tokenized_50_eval.npy")
    parser.add_argument("--train_image_paths", default=None)
    parser.add_argument("--val_image_paths", default=None)
    parser.add_argument("--embeddings_path", default="data/gemma-4-31B-it-embeddings.pt")
    parser.add_argument("--clip_model_id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--max_context_length", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--q_head", type=int, default=8)
    parser.add_argument("--kv_head", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit pairs per train/validation dataset for smoke tests.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit batches per training/validation epoch for smoke tests.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "max_context_length", "num_layer",
                 "max_samples", "max_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive.")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")
    if args.projection_dim != args.q_head * args.head_dim:
        parser.error("projection_dim must equal q_head * head_dim.")
    if args.kv_head < 1 or args.q_head % args.kv_head or args.head_dim % 2:
        parser.error("q_head must be divisible by kv_head and head_dim must be even.")
    if args.data_root and (args.train_image_paths or args.val_image_paths):
        parser.error("--data_root discovers image paths; omit image-path manifests.")
    return args


def unpack_batch(batch):
    """Keep support for the original fixed-size NumPy array datasets."""
    if isinstance(batch, (tuple, list)):
        data, image_paths = batch
        return data, [path or None for path in image_paths]
    return batch, [None] * batch.shape[0]


def prepare_batch(batch, device: str) -> dict:
    if isinstance(batch, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}
    tokens, images = unpack_batch(batch)
    tokens = tokens.long().to(device)
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:],
            "attention_mask": None, "image_paths": images}


def run_epoch(model, loader, device: str, optimizer=None, scheduler=None,
              max_steps: int | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    limit = min(len(loader), max_steps) if max_steps else len(loader)
    with torch.set_grad_enabled(training):
        for batch in tqdm(islice(loader, limit), total=limit,
                          desc="Training" if training else "Validating"):
            batch = prepare_batch(batch, device)
            labels = batch.pop("labels")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, auxiliary_loss = model(batch.pop("input_ids"), **batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels.reshape(-1), ignore_index=-100)
            if training:
                (loss + auxiliary_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            count = (labels != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("Dataset contains no target tokens.")
    return total_loss / total_tokens


def save_training_checkpoint(path: Path, model: VLM, model_config: dict,
                             tokenizer_path: str, special_ids: dict,
                             optimizer, scheduler, epoch: int, loss: float) -> None:
    """Save model construction details so inference requires no architecture flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(), "model_config": model_config,
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "special_token_ids": special_ids, "epoch": epoch, "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(Path(args.tokenizer_path).expanduser(),
                                              local_files_only=True)
    special_ids = {name: getattr(tokenizer, name) for name in
                   ("bos_token_id", "eos_token_id", "pad_token_id")}
    if any(value is None for value in special_ids.values()):
        raise ValueError("Tokenizer must define BOS, EOS, and PAD IDs.")
    model_config = dict(
        num_layer=args.num_layer, max_context_length=args.max_context_length,
        word_embeddings_tensor=str(Path(args.embeddings_path).expanduser().resolve()),
        projection_dim=args.projection_dim, expansion_factor=args.expansion_factor,
        head_dim=args.head_dim, q_head=args.q_head, kv_head=args.kv_head,
        clip_model_id=args.clip_model_id,
    )
    model = VLM(**model_config, device=args.device).to(args.device)
    if max(tokenizer.get_vocab().values()) >= model.vocabulary_size:
        raise ValueError("Tokenizer IDs exceed the embedding matrix vocabulary.")
    if args.data_root:
        train_loader, val_loader = prepare_annotation_dataset(
            args.data_root, args.train_folders, args.val_folders, args.batch_size,
            args.max_context_length, model.vocabulary_size, **special_ids,
            num_workers=args.num_workers, max_samples=args.max_samples,
        )
    else:
        train_loader, val_loader = prepare_dataset(
            args.train_path, args.val_path, args.batch_size, args.batch_size,
            args.train_image_paths, args.val_image_paths,
        )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.01)
    def lr_scale(step):
        if step < warmup_steps:
            return max(0.01, (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_scale)
    run = None
    if args.project is not None and args.entity is not None:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                         config={key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()})
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.device, optimizer, scheduler,
                               args.max_steps)
        val_loss = run_epoch(model, val_loader, args.device, max_steps=args.max_steps)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        output = args.output_dir / "latest.pt"
        save_training_checkpoint(output, model, model_config, args.tokenizer_path,
                                 special_ids, optimizer, scheduler, epoch, val_loss)
        print(f"Checkpoint saved: {output}")
        if run:
            run.log({"Training Loss": train_loss, "Val loss": val_loss, "epoch": epoch})
    if run:
        run.finish()


if __name__ == "__main__":
    main()
