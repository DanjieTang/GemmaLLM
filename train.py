"""Train the custom CLIP-conditioned annotation decoder or legacy token arrays."""

import argparse
from collections.abc import Callable, Iterator
from itertools import cycle, islice
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

from generate import generate
from lazy_dataloader import AnnotationDataset, prepare_annotation_dataset, prepare_dataset
from model import VLM


def parse_bool(value: str) -> bool:
    """Accept explicit boolean values from sweep commands and the CLI."""
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Parent of paired dataset folders (default: data).")
    parser.add_argument("--train_folders", nargs="+", default=["train", "OpenImageV7_train"])
    parser.add_argument("--val_folders", nargs="+", default=["val", "OpenImageV7_val"])
    parser.add_argument("--tokenizer_path", default=str(Path.home() / "models/gemma-4-E2B-it"))
    parser.add_argument("--train_path", default=None,
                        help="Legacy rectangular token array; requires --val_path.")
    parser.add_argument("--val_path", default=None,
                        help="Legacy validation array; requires --train_path.")
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
    parser.add_argument("--dropout_ratio", type=float, default=0.1)
    parser.add_argument("--theta", type=int, default=10000,
                        help="Base used for rotary position embeddings.")
    parser.add_argument("--use_moe", type=parse_bool, nargs="?", const=True,
                        default=False, help="Enable MoE; accepts true/false.")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--load_balancing_loss_weight", type=float, default=1e-2)
    parser.add_argument("--fine_tuning", type=parse_bool, nargs="?", const=True,
                        default=False, help="Train only LoRA parameters; accepts true/false.")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Load model weights before training (matching architecture required).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit pairs per train/validation dataset for smoke tests.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit batches per training/validation epoch for smoke tests.")
    parser.add_argument("--inference_every", type=int, default=1000,
                        help="Generate one validation image annotation every N training "
                             "iterations across epochs; 0 disables previews.")
    parser.add_argument("--inference_max_new_tokens", type=int, default=256,
                        help="Maximum generated tokens per preview, capped by model context.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "max_context_length", "num_layer",
                 "max_samples", "max_steps", "projection_dim", "q_head",
                 "kv_head", "head_dim", "expansion_factor", "theta",
                 "num_experts", "lora_rank", "inference_max_new_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive.")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")
    if args.inference_every < 0:
        parser.error("--inference_every cannot be negative.")
    if not 0 <= args.dropout_ratio <= 1:
        parser.error("--dropout_ratio must be between 0 and 1.")
    if (not math.isfinite(args.load_balancing_loss_weight)
            or args.load_balancing_loss_weight < 0):
        parser.error("--load_balancing_loss_weight must be finite and nonnegative.")
    if args.lora_alpha < 0:
        parser.error("--lora_alpha cannot be negative.")
    if args.use_moe and args.num_experts < 2:
        parser.error("--use_moe requires at least two experts for top-2 routing.")
    if args.projection_dim != args.q_head * args.head_dim:
        parser.error("projection_dim must equal q_head * head_dim.")
    if args.kv_head < 1 or args.q_head % args.kv_head or args.head_dim % 2:
        parser.error("q_head must be divisible by kv_head and head_dim must be even.")
    legacy = args.train_path is not None or args.val_path is not None
    if legacy and (args.train_path is None or args.val_path is None):
        parser.error("Legacy training requires both --train_path and --val_path.")
    if legacy and args.data_root is not None:
        parser.error("Choose --data_root or legacy --train_path/--val_path.")
    if not legacy and args.data_root is None:
        args.data_root = "data"
    if args.data_root is not None and (args.train_image_paths or args.val_image_paths):
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


def iter_validation_images(dataset) -> Iterator[str]:
    """Read paired image paths without loading annotations; support legacy pairs too."""
    if isinstance(dataset, AnnotationDataset):
        for _, image_path in dataset.samples:
            yield image_path
    else:
        for sample in dataset:
            if isinstance(sample, (tuple, list)) and sample[1]:
                yield sample[1]


def print_image_inference(model: VLM, tokenizer, image_path: str, step: int,
                          max_new_tokens: int) -> None:
    """Decode from BOS and the image only; generate restores the training mode."""
    tokens = generate(model, image_path, tokenizer.bos_token_id,
                      tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                      max_new_tokens=max_new_tokens, temperature=0.0)
    annotation = tokenizer.decode(tokens, skip_special_tokens=True)
    with tqdm.external_write_mode():
        print(f"\n[Inference | iteration {step}] Image: {image_path}\n"
              f"Generated annotation: {annotation}", flush=True)


def run_epoch(model, loader, device: str, optimizer=None, scheduler=None,
              max_steps: int | None = None, *, start_step: int = 0,
              inference_every: int = 0,
              inference_callback: Callable[[int], None] | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    limit = min(len(loader), max_steps) if max_steps else len(loader)
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(tqdm(
                islice(loader, limit), total=limit,
                desc="Training" if training else "Validating"), start=start_step + 1):
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
                if scheduler is not None:
                    scheduler.step()
            count = (labels != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
            # Release the batch logits before allocating the generation cache.
            del logits, loss, auxiliary_loss
            if (training and inference_every and step % inference_every == 0
                    and inference_callback is not None):
                inference_callback(step)
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
        dropout_ratio=args.dropout_ratio, theta=args.theta,
        use_moe=args.use_moe, num_experts=args.num_experts,
        load_balancing_loss_weight=args.load_balancing_loss_weight,
        fine_tuning=args.fine_tuning, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        clip_model_id=args.clip_model_id,
    )
    model = VLM(**model_config, device=args.device).to(args.device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint.expanduser(),
                                map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    if args.fine_tuning:
        model.begin_fine_tunning()
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
    inference_callback = None
    if args.inference_every:
        if args.data_root or args.val_image_paths:
            image_paths = cycle(iter_validation_images(val_loader.dataset))

            def inference_callback(step: int) -> None:
                image_path = next(image_paths, None)
                if image_path is None:
                    tqdm.write("Image inference skipped: no validation images available.")
                    return
                print_image_inference(model, tokenizer, image_path, step,
                                      args.inference_max_new_tokens)
        else:
            print("Image inference disabled: legacy validation has no image manifest.")
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
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.device, optimizer, scheduler,
                               args.max_steps, start_step=global_step,
                               inference_every=args.inference_every,
                               inference_callback=inference_callback)
        global_step += steps_per_epoch
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
