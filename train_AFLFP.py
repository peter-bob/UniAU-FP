"""Train the complete UniAU-FP model on AFLFP."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import datetime
import importlib
import random

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from engine_fer_first_stage_3 import train, test
from models.clip_vit import CLIPVIT
from utils.ema import EMA
from utils.misc import convert_models_to_fp32, init_log, write_description_to_folder
from utils.mixup import MixupRegression


AU_DESCRIPTIONS = {
    "Left_AU02": "left outer brow raiser",
    "Left_AU04": "left brow lowerer",
    "Left_AU06": "left cheek raiser",
    "Left_AU15": "left lip corner depressor",
    "Left_AU43": "left eye closed",
    "Right_AU02": "right outer brow raiser",
    "Right_AU04": "right brow lowerer",
    "Right_AU06": "right cheek raiser",
    "Right_AU15": "right lip corner depressor",
    "Right_AU43": "right eye closed",
}


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_metric(train_values, val_values, metric_name, save_path, ema_values=None):
    epochs = range(1, len(train_values) + 1)
    plt.figure()
    plt.plot(epochs, train_values, label=f"Train {metric_name}")
    plt.plot(epochs, val_values, label=f"Validation {metric_name}")
    if ema_values is not None:
        plt.plot(epochs, ema_values, "--", label=f"Validation {metric_name} (EMA)")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_optimizer(model, args):
    head_params = []
    for name in ("mad", "text_proj", "cross_encoder", "reg_head", "asym_aux_head"):
        head_params.extend(getattr(model, name).parameters())
    head_params.extend([model.au_query_embed, model.out_scale, model.out_bias])
    head_ids = {id(parameter) for parameter in head_params}
    trainable_head = [p for p in head_params if p.requires_grad]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    groups = [{"params": trainable_head, "lr_scale": 1.0}]
    if args.lr_backbone > 0 and backbone:
        groups.append({"params": backbone, "lr_scale": args.lr_backbone / args.lr_head})
    optimizer = torch.optim.AdamW(
        groups, lr=args.lr_head, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    print(
        f"[Optimizer] head={sum(p.numel() for p in trainable_head):,} "
        f"backbone={sum(p.numel() for p in backbone):,}"
    )
    return optimizer


def make_criterion(args):
    weights = torch.tensor(
        [1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0],
        dtype=torch.float32,
    )

    def criterion(prediction, target):
        losses = F.smooth_l1_loss(
            prediction, target, reduction="none", beta=args.huber_beta
        )
        if args.dataset == "MEEI":
            with torch.no_grad():
                dynamic_weights = 1.0 + torch.clamp(
                    (prediction - target).abs(), max=1.0
                ) * 2.0
            losses = losses * dynamic_weights
        if args.use_class_weights:
            losses = losses * weights.to(losses.device).view(1, -1)
        return losses.mean()

    return criterion


def configure_output_activation(model, train_loader, activation):
    if activation != "auto":
        return
    _, labels = next(iter(train_loader))
    if labels.min() >= -0.05 and labels.max() <= 1.05:
        model.cfg.output_activation = "sigmoid"
        print("[Output] Label range is approximately [0, 1]; using sigmoid.")
    else:
        model.cfg.output_activation = "none"
        print("[Output] Labels exceed [0, 1]; using linear output.")


def main(args):
    set_random_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    args.record_path = os.path.join("outputs", f"{args.dataset}_RUN", f"{stamp}-FullModel")
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    dataset_utils = importlib.import_module(
        f"dataloader.{args.dataset}_data_utils_1"
    )
    dataset_utils.set_up_datasets(args)
    _, _, train_loader, val_loader = dataset_utils.get_dataloader(args)
    dataset_utils.get_labelname(args)
    label_tokens = clip.tokenize([AU_DESCRIPTIONS[name] for name in args.label_nms])

    clip_model, _ = clip.load(args.clip_path, jit=False)
    model = CLIPVIT(args, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)
    configure_output_activation(model, train_loader, args.output_activation)
    args.lr = args.lr_head
    optimizer = build_optimizer(model, args)
    criterion = make_criterion(args)

    ema = EMA(model, decay=args.ema_decay, warmup_steps=len(train_loader) * 3)
    args.ema = ema
    args.mixup_fn = (
        MixupRegression(alpha=args.mixup_alpha, prob=args.mixup_prob)
        if args.mixup_alpha > 0 else None
    )

    train_rmse, train_mae, train_loss = [], [], []
    val_rmse, val_mae, val_loss, ema_rmse = [], [], [], []
    best_rmse, best_epoch, no_improve = float("inf"), -1, 0

    for epoch in range(args.epochs):
        model.train()
        tr_rmse, tr_mae, tr_loss = train(
            model, args, optimizer, criterion, train_loader, logger, label_tokens, epoch
        )
        model.eval()
        va_rmse, va_mae, va_loss = test(
            model, args, criterion, val_loader, logger, label_tokens, epoch
        )
        ema.apply_shadow()
        ema_r, _, _ = test(model, args, criterion, val_loader, logger, label_tokens, epoch)
        ema.restore()

        effective_rmse = min(va_rmse, ema_r)
        use_ema = ema_r < va_rmse
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_shadow": ema.shadow,
            "best_rmse": best_rmse,
        }, os.path.join(args.record_path, "last.pth"))

        if effective_rmse < best_rmse - args.min_delta:
            best_rmse, best_epoch, no_improve = effective_rmse, epoch, 0
            if use_ema:
                ema.apply_shadow()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_rmse": best_rmse,
                "used_ema": use_ema,
            }, os.path.join(args.record_path, "best_model.pth"))
            if use_ema:
                ema.restore()
            logger.info(f"[BEST] Epoch {epoch} | Val RMSE: {best_rmse:.4f}")
        else:
            no_improve += 1

        train_rmse.append(tr_rmse); train_mae.append(tr_mae); train_loss.append(tr_loss)
        val_rmse.append(va_rmse); val_mae.append(va_mae); val_loss.append(va_loss)
        ema_rmse.append(ema_r)
        plot_metric(train_rmse, val_rmse, "RMSE", os.path.join(args.record_path, "rmse_curve.png"), ema_rmse)
        plot_metric(train_mae, val_mae, "MAE", os.path.join(args.record_path, "mae_curve.png"))
        plot_metric(train_loss, val_loss, "Loss", os.path.join(args.record_path, "loss_curve.png"))
        if no_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}; best RMSE={best_rmse:.4f} at epoch {best_epoch}.")
            break

    logger.info(f"Training complete. Best validation RMSE: {best_rmse:.4f} at epoch {best_epoch}.")
    print(f"Training complete. Best validation RMSE: {best_rmse:.4f} at epoch {best_epoch}.")


def parse_args(default_dataset="AFLFP"):
    parser = argparse.ArgumentParser(description="Train the full UniAU-FP model")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--dataset", default=default_dataset, choices=["AFLFP", "MEEI"])
    parser.add_argument("--data-path", default=f"./data/{default_dataset}")
    parser.add_argument("--data_split_path", default="./annotation")
    parser.add_argument("--clip-path", default="./weights/ViT-B-16.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--freeze_clip", action="store_true", default=True)
    parser.add_argument("--no_freeze_clip", action="store_false", dest="freeze_clip")
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--output_activation", choices=["auto", "none", "sigmoid", "tanh"], default="auto")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=5e-5)
    parser.add_argument("--asym_loss_weight", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.997)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--mixup_prob", type=float, default=0.6)
    parser.add_argument("--label_noise_std", type=float, default=0.02)
    parser.add_argument("--rdrop_alpha", type=float, default=0.2)
    parser.add_argument("--asym_sigma", type=float, default=0.3)
    parser.add_argument("--asym_dist_beta", type=float, default=0.5)
    parser.add_argument("--asym_num_bins", type=int, default=51)
    parser.add_argument("--asym_bin_max", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
