"""Train the complete UniAU-FP model on MEEI."""

from train_AFLFP import main, parse_args


if __name__ == "__main__":
    main(parse_args(default_dataset="MEEI"))
