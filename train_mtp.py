"""Jointly train the VLM and DeepSeek-style multi-token prediction modules."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle, islice
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

from lazy_dataloader import prepare_annotation_dataset, prepare_dataset
from model import MTPModule, VLM
from train import (
    build_parser, iter_validation_images, prepare_batch,
    print_image_inference, validate_args,
)


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    parser.description = __doc__
    parser.epilog = (
        "With --fine_tuning, the backbone trains LoRA parameters while new draft "
        "blocks train in full. --init_checkpoint initializes base or matching MTP "
        "weights; optimizer, scheduler, and epoch count start fresh."
    )
    parser.set_defaults(output_dir=Path("checkpoints/mtp"))
    parser.add_argument("--mtp_depth", type=int, default=1,
                        help="Number of sequential MTP modules (default: 1).")
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3,
                        help="Weight of the mean future-token loss (default: 0.3).")
    args = validate_args(parser.parse_args(argv), parser)
    if args.mtp_depth < 1 or args.mtp_depth >= args.max_context_length:
        parser.error("--mtp_depth must be positive and less than --max_context_length.")
    if not math.isfinite(args.mtp_loss_weight) or args.mtp_loss_weight <= 0:
        parser.error("--mtp_loss_weight must be finite and positive.")
    return args


@dataclass
class TrainingLosses:
    objective: torch.Tensor
    main: torch.Tensor
    main_count: int
    mtp: list[torch.Tensor | None]
    mtp_counts: list[int]


class MTPTrainingModel(nn.Module):
    """Own the backbone and draft blocks so shared parameters are optimized once."""

    def __init__(self, base_model: VLM, model_config: dict, mtp_depth: int,
                 mtp_loss_weight: float, eos_token_id: int):
        super().__init__()
        self.base_model = base_model
        self.mtp_depth = mtp_depth
        self.mtp_loss_weight = mtp_loss_weight
        self.eos_token_id = eos_token_id
        module_config = {
            key: model_config[key]
            for key in (
                "expansion_factor", "head_dim", "q_head", "kv_head",
                "dropout_ratio", "theta", "use_moe", "num_experts",
                "load_balancing_loss_weight", "lora_rank", "lora_alpha",
            ) if key in model_config
        }
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                hidden_dim=base_model.llm.classifier.in_features,
                max_context_length=base_model.max_context_length,
                output_norm=base_model.llm.output_norm,
                classifier=base_model.llm.classifier,
                device=base_model.device, **module_config,
            ) for _ in range(mtp_depth)
        ])

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                image_paths=None, image_tokens: torch.Tensor | None = None) -> TrainingLosses:
        """Compute main and shifted MTP losses without detaching the backbone.

        input_ids[:, i] is t_i; labels[:, i] is t_(i+1) or -100. At depth k,
        pair h_i^(k-1) with Emb(t_(i+k)) and supervise using labels[:, i+k].
        The last label (often EOS) remains a target even though it isn't an
        input token. Image-prefix positions never enter the text shifts.
        """
        if input_ids.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching shapes.")
        valid_inputs = (torch.ones_like(input_ids, dtype=torch.bool)
                        if attention_mask is None else attention_mask.bool())
        logits, main_aux, hidden = self.base_model(
            input_ids, image_paths=image_paths, attention_mask=attention_mask,
            image_tokens=image_tokens, output_hidden_states=True,
        )
        main_targets = labels.masked_fill(~valid_inputs, -100)
        main_count = int((main_targets != -100).sum())
        if not main_count:
            raise ValueError("Batch contains no target tokens.")
        main_loss = F.cross_entropy(logits.flatten(0, 1), main_targets.flatten())
        del logits
        embeddings = self.base_model.embed_tokens(input_ids)
        mtp_losses = [None] * self.mtp_depth
        mtp_counts = [0] * self.mtp_depth
        draft_aux = main_aux.new_zeros(())
        # EOS may be a target, but no intervening input may be EOS. This
        # prevents future-token objectives from crossing packed document ends.
        pair_mask = valid_inputs & (input_ids != self.eos_token_id)
        for index, module in enumerate(self.mtp_modules):
            depth = index + 1
            length = input_ids.shape[1] - depth
            if length <= 0:
                break
            pair_mask = (pair_mask[:, :-1] & valid_inputs[:, depth:]
                         & (input_ids[:, depth:] != self.eos_token_id))
            targets = labels[:, depth:].masked_fill(~pair_mask, -100)
            count = int((targets != -100).sum())
            if not count:
                break
            draft_logits, auxiliary_loss, hidden = module(
                hidden[:, :length], embeddings[:, depth:],
                valid_token_mask=valid_inputs[:, depth:],
            )
            mtp_losses[index] = F.cross_entropy(
                draft_logits.flatten(0, 1), targets.flatten(),
            )
            mtp_counts[index] = count
            draft_aux = draft_aux + auxiliary_loss
        # Empty depths contribute zero; the denominator remains configured D.
        mtp_loss = sum((loss for loss in mtp_losses if loss is not None),
                       main_loss.new_zeros(())) / self.mtp_depth
        objective = (main_loss + main_aux + self.mtp_loss_weight * mtp_loss
                     + draft_aux / self.mtp_depth)
        return TrainingLosses(objective, main_loss, main_count, mtp_losses, mtp_counts)


def run_epoch(model: MTPTrainingModel, loader, device: str, optimizer=None,
              scheduler=None, max_steps: int | None = None, start_step: int = 0,
              inference_every: int = 0,
              inference_callback: Callable[[int], None] | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    main_sum, objective_sum, main_count = 0.0, 0.0, 0
    draft_sums = [0.0] * model.mtp_depth
    draft_counts = [0] * model.mtp_depth
    limit = min(len(loader), max_steps) if max_steps else len(loader)
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(tqdm(
                islice(loader, limit), total=limit,
                desc="Training MTP" if training else "Validating MTP"), start=start_step + 1):
            batch = prepare_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = model(**batch)
            if training:
                losses.objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            main_sum += losses.main.item() * losses.main_count
            objective_sum += losses.objective.item() * losses.main_count
            main_count += losses.main_count
            for index, loss in enumerate(losses.mtp):
                if loss is not None:
                    draft_sums[index] += loss.item() * losses.mtp_counts[index]
                    draft_counts[index] += losses.mtp_counts[index]
            del loss, losses
            if (training and inference_every and step % inference_every == 0
                    and inference_callback is not None):
                inference_callback(step)
    if not main_count:
        raise ValueError("Dataset contains no target tokens.")
    metrics = {"main_loss": main_sum / main_count,
               "objective": objective_sum / main_count}
    for index, (total, count) in enumerate(zip(draft_sums, draft_counts), start=1):
        metrics[f"mtp_{index}_loss"] = total / count if count else 0.0
        metrics[f"mtp_{index}_tokens"] = count
    metrics["mtp_loss"] = sum(metrics[f"mtp_{k}_loss"]
                              for k in range(1, model.mtp_depth + 1)) / model.mtp_depth
    return metrics


def load_initial_checkpoint(path: Path, model: MTPTrainingModel,
                            special_ids: dict) -> None:
    """Initialize from base or MTP weights; optimizer/scheduler start fresh."""
    checkpoint = torch.load(path.expanduser(), map_location="cpu", weights_only=True)
    if checkpoint.get("special_token_ids", special_ids) != special_ids:
        raise ValueError("Checkpoint special token IDs differ from the tokenizer.")
    if "mtp_state_dict" in checkpoint:
        if checkpoint["mtp_config"]["mtp_depth"] != model.mtp_depth:
            raise ValueError("Checkpoint mtp_depth differs from --mtp_depth.")
    model.base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "mtp_state_dict" in checkpoint:
        model.mtp_modules.load_state_dict(checkpoint["mtp_state_dict"], strict=True)


def save_training_checkpoint(path: Path, model: MTPTrainingModel, model_config: dict,
                             tokenizer_path: str, special_ids: dict,
                             optimizer, scheduler, epoch: int, metrics: dict) -> None:
    """Save the base in generate.py's format plus MTP weights and configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model_state_dict": model.base_model.state_dict(), "model_config": model_config,
        "mtp_state_dict": model.mtp_modules.state_dict(),
        "mtp_config": {"mtp_depth": model.mtp_depth,
                       "mtp_loss_weight": model.mtp_loss_weight},
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "special_token_ids": special_ids, "epoch": epoch,
        "loss": metrics["main_loss"], "metrics": metrics,
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
    base_model = VLM(**model_config, device=args.device).to(args.device)
    if args.fine_tuning:
        # Freeze the backbone first; newly created draft blocks train in full.
        base_model.begin_fine_tunning()
    model = MTPTrainingModel(base_model, model_config, args.mtp_depth,
                             args.mtp_loss_weight, special_ids["eos_token_id"]).to(args.device)
    if args.init_checkpoint is not None:
        load_initial_checkpoint(args.init_checkpoint, model, special_ids)
    if max(tokenizer.get_vocab().values()) >= base_model.vocabulary_size:
        raise ValueError("Tokenizer IDs exceed the embedding matrix vocabulary.")
    if args.data_root:
        train_loader, val_loader = prepare_annotation_dataset(
            args.data_root, args.train_folders, args.val_folders, args.batch_size,
            args.max_context_length, base_model.vocabulary_size, **special_ids,
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
                print_image_inference(base_model, tokenizer, image_path, step,
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
        train_metrics = run_epoch(model, train_loader, args.device, optimizer, scheduler,
                                  args.max_steps, start_step=global_step,
                                  inference_every=args.inference_every,
                                  inference_callback=inference_callback)
        global_step += steps_per_epoch
        val_metrics = run_epoch(model, val_loader, args.device, max_steps=args.max_steps)
        print(
            f"Epoch {epoch}: Train main {train_metrics['main_loss']:.4f} "
            f"| Train MTP {train_metrics['mtp_loss']:.4f} "
            f"| Val main {val_metrics['main_loss']:.4f} "
            f"| Val MTP {val_metrics['mtp_loss']:.4f}"
        )
        if not any(val_metrics[f"mtp_{k}_tokens"] for k in range(1, args.mtp_depth + 1)):
            print("Validation has no valid MTP targets; consider longer sequences.")
        output = args.output_dir / "latest.pt"
        save_training_checkpoint(output, model, model_config, args.tokenizer_path,
                                 special_ids, optimizer, scheduler, epoch, val_metrics)
        print(f"Checkpoint saved: {output}")
        if run:
            run.log({"epoch": epoch,
                     **{f"train/{key}": value for key, value in train_metrics.items()},
                     **{f"val/{key}": value for key, value in val_metrics.items()}})
    if run:
        run.finish()


if __name__ == "__main__":
    main()
