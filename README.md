# UniAU-FP: Unilateral Facial Action Unit Intensity Estimation

Official research code for **UniAU-FP**, a vision-language framework for side-specific facial Action Unit (AU) intensity estimation in facial paralysis assessment.

Facial paralysis can produce different muscle movements on the two sides of the face. UniAU-FP predicts the intensity of five AUs independently on the anatomical left and right sides:

`AU02`, `AU04`, `AU06`, `AU15`, and `AU43`

The model combines CLIP visual features, hierarchical bilateral asymmetry modeling, AU-aware text prompts, cross-modal attention, and an auxiliary asymmetry prediction branch.

> This repository contains research code. It is not a medical device and must not be used for clinical diagnosis without appropriate validation and regulatory review.

## Highlights

- Side-specific AU intensity regression with 10 outputs per face: 5 AUs x 2 sides.
- Unilateral AU annotations constructed from AFLFP and MEEI facial palsy benchmarks.
- Hierarchical Bilateral Asymmetry Fusion (HBAF) for upper, middle, and lower facial regions.
- Dual-stream cross-modal attention between AU prompts and visual region features.
- Asymmetry Difference Auxiliary Branch for explicit left-right difference supervision.
- Grad-CAM visualization for qualitative inspection of AU-related attention.

## Results

The reported single-run results from the accompanying paper are:

| Dataset | RMSE (lower is better) | MAE (lower is better) |
|---|---:|---:|
| AFLFP | **0.183** | **0.124** |
| MEEI | **0.144** | **0.109** |

These numbers should be interpreted together with the paper's limitations: the current release does not establish clinical validity, inter-rater reliability, or statistical significance across repeated runs.

## Datasets and labels

The experiments use two public facial palsy benchmarks:

- **AFLFP**: 3,460 labeled images, with 2,775 training images and 685 test images.
- **MEEI**: 2,395 labeled images, with 1,915 training images and 480 test images.

Together, the experiments contain 5,855 images and 58,550 unilateral AU labels. The repository does **not** redistribute the original facial images. Obtain AFLFP and MEEI from their official sources and follow their licenses and access conditions.

The `annotation/` files provide experiment split/index information. They are not a replacement for the original datasets and should not be redistributed separately from the applicable dataset permissions.

## Repository structure

```text
annotation/                         Experiment split files
clip/                               Local CLIP implementation and tokenizer
dataloader/                         AFLFP and MEEI data loaders
models/                             UniAU-FP and ablation model variants
utils/                              EMA, scheduling, mixup, and utilities
train_fer_first_stage_AFLFP_*.py   AFLFP training entry points
train_fer_first_stage_MEEI_*.py    MEEI training entry points
train_ablation_AFLFP.py             AFLFP ablation experiments
train_ablation_MEEI.py              MEEI ablation experiments
attention_heatmap_compare_*.py      Grad-CAM/attention visualization
UniAU-FP_camera_ready_revised.tex   Manuscript source
requirements.txt                    Python dependencies
```

## Installation

The commands below are for Windows PowerShell. On Linux or macOS, replace `py -3.10` with `python3` and activate the corresponding virtual environment.

```powershell
git clone https://github.com/peter-bob/UniAU-FP.git
cd UniAU-FP

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/>. The code uses a local CLIP ViT-B/16 checkpoint; keep it outside Git history, for example at:

```text
weights/ViT-B-16.pt
```

The checkpoint can also be supplied with `--clip-path`.

## Data layout

Place the legally obtained datasets outside the repository or under a local ignored `data/` directory:

```text
UniAU-FP/
  data/
    AFLFP/
    MEEI/
  annotation/
  weights/
    ViT-B-16.pt
```

Do not commit patient images, private annotations, pretrained weights, checkpoints, logs, or generated outputs.

## Training

### AFLFP

```powershell
python train_fer_first_stage_AFLFP_4.py `
  --data-path .\data\AFLFP `
  --data_split_path .\annotation `
  --clip-path .\weights\ViT-B-16.pt
```

### MEEI

```powershell
python train_fer_first_stage_MEEI_4.py `
  --data-path .\data\MEEI `
  --data_split_path .\annotation `
  --clip-path .\weights\ViT-B-16.pt
```

Common options include `--seed`, `--batch-size`, `--epochs`, `--workers`, `--unfreeze_last_n`, and `--device` where supported by the selected entry point. Training outputs are written to `outputs/`, which is ignored by Git.

## Ablation experiments

The ablation drivers use four configurations:

| Configuration | Meaning |
|---|---|
| C1 | Baseline without text, HBAF/MAD, and auxiliary asymmetry supervision |
| C2 | Adds the cross-modal interaction module |
| C3 | Adds bilateral asymmetry modeling |
| C4 | Full UniAU-FP model |

Example:

```powershell
python train_ablation_AFLFP.py --ablation_config 4
python train_ablation_MEEI.py --ablation_config 4
```

For a new study, use multiple random seeds and report confidence intervals. The cumulative C1-C4 comparison alone should not be treated as an isolated contribution estimate for each module.

## Attention visualization

`attention_heatmap_compare_FORCE5_BLUE.py` compares Grad-CAM maps from baseline and ablation checkpoints. A typical command is:

```powershell
python attention_heatmap_compare_FORCE5_BLUE.py `
  --project-root . `
  --dataset AFLFP `
  --data-path .\data\AFLFP `
  --data_split_path .\annotation `
  --clip-path .\weights\ViT-B-16.pt `
  --baseline-checkpoint .\outputs\AFLFP_ABLATION\C1_RUN\best_model.pth `
  --full-checkpoint .\outputs\AFLFP_ABLATION\C4_RUN\best_model.pth
```

Grad-CAM is qualitative: anatomically plausible attention does not, by itself, prove that the affected side has been identified.

## Reproducibility checklist

For each experiment, record:

- Git commit ID
- Dataset release/version and permission status
- Exact split files
- Random seed
- GPU and CUDA version
- PyTorch version
- Checkpoint filename and checksum
- Full command line and configuration

## Paper

The manuscript source is included in `UniAU-FP_camera_ready_revised.tex`. Please verify the final experimental tables against the archived logs before submitting or citing a camera-ready version.

## Citation

The formal citation will be added after the conference publication metadata is available.

## License and third-party materials

The original research code is released under the MIT License in `LICENSE`. AFLFP, MEEI, CLIP weights, and other third-party materials remain subject to their own licenses, access requirements, and attribution rules.

## Contact

For questions about the implementation, please open a GitHub issue with the operating system, Python/PyTorch versions, command line, and a complete error message. Do not upload patient images or private data to an issue.
