"""
train_ablation_AFLFP.py - AFLFP Reverse Ablation Experiment Driver
===========================================================
Runs cumulative reverse ablations starting from the full model.

Config descriptions:
  1 - NoText / NoMAD / NoAux  (learnable AU queries over raw patch memory)
  2 - NoMAD / NoAux           (original text stream over raw patch memory)
  3 - NoAux                   (MAD + original text fusion, no aux branch)
  4 - Full original model     (original C4, parameters unchanged)

Expected capacity order for C1-C3: C1 < C2 < C3, so validation error should
ideally decrease from config 1 to config 3 under the same training setup.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from torch import nn

from utils.misc import init_log, write_description_to_folder, convert_models_to_fp32
from utils.ema import EMA
from utils.mixup import MixupRegression
from dataloader.AFLFP_data_utils_1 import set_up_datasets, get_dataloader, get_labelname
from models.clip_vit_ablation import build_ablation_model, get_config_name, get_head_params
from engine_fer_first_stage_3 import train, test
import engine_fer_first_stage_3
print("ENGINE FILE USED:", engine_fer_first_stage_3.__file__)


AU_DESC = {
    "Left_AU02": "left outer brow raiser", "Left_AU04": "left brow lowerer",
    "Left_AU06": "left cheek raiser",      "Left_AU15": "left lip corner depressor",
    "Left_AU43": "left eye closed",        "Right_AU02": "right outer brow raiser",
    "Right_AU04": "right brow lowerer",    "Right_AU06": "right cheek raiser",
    "Right_AU15": "right lip corner depressor", "Right_AU43": "right eye closed",
}

C2_AU_DESC = {
    "Left_AU02": "clinical face photo, left outer brow raiser AU02, compare left and right forehead asymmetry",
    "Left_AU04": "clinical face photo, left brow lowerer AU04, compare left and right brow asymmetry",
    "Left_AU06": "clinical face photo, left cheek raiser AU06, compare left and right cheek and eye wrinkle asymmetry",
    "Left_AU15": "clinical face photo, left lip corner depressor AU15, compare left and right mouth corner asymmetry",
    "Left_AU43": "clinical face photo, left eyelid closure AU43, compare left and right eye closing asymmetry",
    "Right_AU02": "clinical face photo, right outer brow raiser AU02, compare left and right forehead asymmetry",
    "Right_AU04": "clinical face photo, right brow lowerer AU04, compare left and right brow asymmetry",
    "Right_AU06": "clinical face photo, right cheek raiser AU06, compare left and right cheek and eye wrinkle asymmetry",
    "Right_AU15": "clinical face photo, right lip corner depressor AU15, compare left and right mouth corner asymmetry",
    "Right_AU43": "clinical face photo, right eyelid closure AU43, compare left and right eye closing asymmetry",
}


def set_random_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_metric(train_v, val_v, metric_name, save_path, ema_v=None):
    epochs = range(1, len(train_v) + 1)
    plt.figure()
    plt.plot(epochs, train_v, label=f"Train {metric_name}")
    plt.plot(epochs, val_v,   label=f"Val {metric_name}")
    if ema_v is not None:
        plt.plot(epochs, ema_v, label=f"Val {metric_name} (EMA)", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel(metric_name)
    plt.title(f"{metric_name} Curve"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def build_optimizer(model, config_id: int, args):
    lr_head      = float(getattr(args, "lr_head", args.lr))
    lr_backbone  = float(getattr(args, "lr_backbone", 0.0))
    weight_decay = float(getattr(args, "weight_decay", 0.05))

    head_params = get_head_params(model, config_id)
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for _, p in model.named_parameters()
                       if p.requires_grad and id(p) not in head_param_ids]

    trainable_head = [p for p in head_params if p.requires_grad]
    param_groups = []
    if trainable_head:
        param_groups.append({"params": trainable_head,
                             "weight_decay": weight_decay, "lr_scale": 1.0})
    if lr_backbone > 0 and backbone_params:
        param_groups.append({"params": backbone_params,
                             "weight_decay": weight_decay,
                             "lr_scale": lr_backbone / max(lr_head, 1e-12)})

    optimizer = torch.optim.AdamW(param_groups, lr=lr_head, betas=(0.9, 0.999), eps=1e-8)
    n_h = sum(p.numel() for p in trainable_head)
    n_b = sum(p.numel() for p in backbone_params)
    print(f"[Optimizer] head={n_h:,} backbone={n_b:,} total={n_h+n_b:,}")
    return optimizer


def make_criterion(args):
    huber_beta = float(getattr(args, "huber_beta", 0.1))
    # AFLFP class weights — uniform base with AU43/AU06 emphasis
    w_list = getattr(args, "class_weights", None) or [1.0, 1.0, 1.5, 1.0, 2.0,
                                                       1.0, 1.0, 1.5, 1.0, 2.0]
    w = torch.tensor(w_list, dtype=torch.float32)

    def criterion(pred, target):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=huber_beta)
        if bool(getattr(args, "use_class_weights", True)):
            loss = loss * w.to(loss.device).view(1, -1)
        return loss.mean()
    return criterion


def auto_set_output_activation(model, train_loader, args):
    if str(getattr(args, "output_activation", "none")).lower() != "auto":
        return
    for _, y in train_loader:
        y_min, y_max = float(y.min()), float(y.max()); break
    else:
        return
    if y_min >= -0.05 and y_max <= 1.05:
        model.cfg.output_activation = "sigmoid"
        print("[AutoAct] Label range ~[0,1]. Use sigmoid.")
    else:
        model.cfg.output_activation = "none"
        print(f"[AutoAct] Label range ~[{y_min:.3f},{y_max:.3f}]. Use linear.")


def main(args):
    set_random_seed(int(getattr(args, "seed", 3407)))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_name = get_config_name(args.ablation_config)
    record_name = (
        datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        + f"-{args.dataset}-Ablation_{cfg_name}"
        + f"-ema{args.ema_decay}-mixup{args.mixup_alpha}"
        + f"-ep{args.epochs}-bs{args.batch_size}"
    )
    args.record_path = os.path.join("outputs", "AFLFP_ABLATION", record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    set_up_datasets(args)
    _, _, train_dataloader, val_dataloader = get_dataloader(args)
    get_labelname(args)

    prompt_table = C2_AU_DESC if int(args.ablation_config) == 2 else AU_DESC
    label_prompts = [prompt_table[nm] for nm in args.label_nms]
    print(f"[{cfg_name}] AU label prompts: {label_prompts}")
    label_token = clip.tokenize(label_prompts)

    clip_model, _ = clip.load(args.clip_path, jit=False)
    model = build_ablation_model(args.ablation_config, args, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)
    auto_set_output_activation(model, train_dataloader, args)

    args.lr = float(getattr(args, "lr_head", args.lr))
    optimizer = build_optimizer(model, args.ablation_config, args)
    criterion = make_criterion(args)

    # EMA — per-step update inside engine
    ema = EMA(model, decay=args.ema_decay, warmup_steps=len(train_dataloader) * 2)
    args.ema = ema
    print(f"[EMA] decay={args.ema_decay}, warmup={len(train_dataloader)*2} steps")

    # Mixup
    if args.mixup_alpha > 0:
        args.mixup_fn = MixupRegression(alpha=args.mixup_alpha, prob=args.mixup_prob)
        print(f"[Mixup] alpha={args.mixup_alpha}, prob={args.mixup_prob}")
    else:
        args.mixup_fn = None

    train_rmse_list, train_mae_list, train_loss_list = [], [], []
    val_rmse_list, val_mae_list, val_loss_list = [], [], []
    ema_rmse_list = []

    best_rmse = float('inf')
    best_epoch = -1
    no_improve = 0
    patience  = int(getattr(args, "patience", 25))
    min_delta = float(getattr(args, "min_delta", 5e-5))

    for epoch in range(args.epochs):
        model.train()
        train_rmse, train_mae, train_loss = train(
            model, args, optimizer, criterion, train_dataloader, logger, label_token, epoch
        )

        model.eval()
        val_rmse, val_mae, val_loss = test(
            model, args, criterion, val_dataloader, logger, label_token, epoch
        )

        ema.apply_shadow()
        print("--- EMA EVALUATION ---")
        ema_rmse, ema_mae, _ = test(
            model, args, criterion, val_dataloader, logger, label_token, epoch
        )
        ema.restore()

        effective_rmse = min(val_rmse, ema_rmse)
        use_ema = ema_rmse < val_rmse

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow, 'best_rmse': best_rmse,
            'ablation_config': args.ablation_config,
        }, os.path.join(args.record_path, "last.pth"))

        if effective_rmse < best_rmse - min_delta:
            best_rmse, best_epoch, no_improve = effective_rmse, epoch, 0
            if use_ema: ema.apply_shadow()
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_rmse': best_rmse, 'used_ema': use_ema,
                'ablation_config': args.ablation_config,
            }, os.path.join(args.record_path, "best_model.pth"))
            if use_ema: ema.restore()
            tag = "(EMA)" if use_ema else "(normal)"
            logger.info(f"[BEST] {cfg_name} Epoch {epoch} | Val RMSE: {best_rmse:.4f} {tag}")
            print(f"[BEST] {cfg_name} Epoch {epoch} | Val RMSE: {best_rmse:.4f} {tag}")
        else:
            no_improve += 1

        train_rmse_list.append(train_rmse); train_mae_list.append(train_mae)
        train_loss_list.append(train_loss); val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae);       val_loss_list.append(val_loss)
        ema_rmse_list.append(ema_rmse)

        plot_metric(train_rmse_list, val_rmse_list, "RMSE",
                    os.path.join(args.record_path, "rmse_curve.png"), ema_rmse_list)
        plot_metric(train_mae_list, val_mae_list, "MAE",
                    os.path.join(args.record_path, "mae_curve.png"))
        plot_metric(train_loss_list, val_loss_list, "Loss",
                    os.path.join(args.record_path, "loss_curve.png"))

        if no_improve >= patience:
            logger.info(f"Early stopping @ epoch {epoch}. Best RMSE: {best_rmse:.4f} @ Epoch {best_epoch}")
            print(f"Early stopping @ epoch {epoch}. Best RMSE: {best_rmse:.4f} @ Epoch {best_epoch}")
            break

    logger.info(f"[{cfg_name}] Done. Best Val RMSE: {best_rmse:.4f} @ Epoch {best_epoch}")
    print(f"[{cfg_name}] Done. Best Val RMSE: {best_rmse:.4f} @ Epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFLFP Ablation Study")

    # ---- Ablation selector ----
    parser.add_argument("--ablation_config", type=int, default=4, choices=[1, 2, 3, 4],
                        help="1=NoText/NoMAD/NoAux, 2=NoMAD/NoAux, 3=NoAux, 4=Full original")

    # ---- Dataset ----
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='AFLFP')
    parser.add_argument('--data-path', type=str,
                        default='./data/AFLFP')
    parser.add_argument('--data_split_path', type=str,
                        default='./annotation')
    parser.add_argument("--clip-path", type=str,
                        default='./weights/ViT-B-16.pt')

    # ---- Training schedule ----
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=5e-5)

    # ---- Optimizer ----
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)

    # ---- Model ----
    parser.add_argument("--head_type", type=str, default="regression")
    parser.add_argument("--freeze_clip", action="store_true", default=True)
    parser.add_argument("--no_freeze_clip", action="store_false", dest="freeze_clip")
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--output_activation", type=str, default="auto",
                        choices=["auto", "none", "sigmoid", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.15)

    # ---- Augmentation / regularization ----
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--asym_loss_weight", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.997)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--mixup_prob", type=float, default=0.6)
    parser.add_argument("--label_noise_std", type=float, default=0.02)
    parser.add_argument("--rdrop_alpha", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
