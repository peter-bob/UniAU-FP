# UniAU-FP

The official code for the paper 'UniAU-FP: Unilateral Facial Action Unit Intensity Estimation via Vision-Language Model for Muscle Movement Assessment in Facial Paralysis'.

This paper has been accepted by 2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM).

<!-- Figure placement: the paper framework figure is stored at assets/uniau_fp_framework.png. Replace this file with a higher-resolution version if needed. -->
<img width="1487" height="941" alt="uniau_fp_framework" src="https://github.com/user-attachments/assets/86e7c05e-f838-4106-8d7b-54308c7a62b3" />


> This repository contains research code. It is not a medical device and must not be used for clinical diagnosis without appropriate validation and regulatory review.

## Project Structure

```text
UniAU-FP/
|-- annotation/                         Experiment split and index files
|-- assets/                             Paper framework figure
|-- clip/                               Local CLIP implementation and tokenizer
|-- dataloader/                         AFLFP and MEEI data loaders
|-- models/                             UniAU-FP and ablation model definitions
|-- utils/                              Training utilities and learning-rate tools
|-- train_AFLFP.py             AFLFP training entry point
|-- train_MEEI.py              MEEI training entry point
|-- requirements.txt                    Python dependencies
|-- LICENSE                             MIT license
`-- README.md                           Project documentation
```

## Environment Setup

The commands below use Windows PowerShell. For Linux or macOS, replace `py -3.10` with the corresponding Python command.

```powershell
git clone https://github.com/peter-bob/UniAU-FP.git
cd UniAU-FP

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/>. The repository uses a local CLIP ViT-B/16 checkpoint. Keep it outside Git history and place it at:

```text
weights/ViT-B-16.pt
```

The checkpoint path can also be changed with `--clip-path`.

## Pretrained Weights

The CLIP checkpoint and trained UniAU-FP checkpoints are not included in this repository. Download or prepare them separately according to the applicable license and place them in local ignored directories:

```text
weights/
|-- ViT-B-16.pt
`-- <trained_checkpoint>.pth
```

Do not commit model weights, patient images, private annotations, logs, or generated outputs.

## Data

Due to licensing restrictions, we cannot provide the complete AFLFP and MEEI datasets. Users need to obtain the raw datasets from their respective official project pages. The dataset annotations (unilateral facial action unit intensity) are publicly available. Users may only use the dataset annotations for non‑commercial academic research. Before downloading the dataset annotations, users are required to send an application email with the signed End User License Agreement as an attachment to xiayifan@sdu.edu.cn using a valid academic or institutional email account.

## Create Output Directories

The training scripts create their experiment directories automatically. Before the first run, create the local data and weight directories if they do not exist:

```powershell
New-Item -ItemType Directory -Force -Path data\AFLFP, data\MEEI, weights
```

Training results are written under `outputs/`, which is ignored by Git.

## Single-Run Training and Validation

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

The `_3.py` and `_4.py` entry points expose closely related model/training variants. Check the selected script with `python <script>.py --help` before launching a long run.


## Common Parameters

| Parameter | Description | Typical value |
|---|---|---:|
| `--data-path` | Dataset root directory | `./data/AFLFP` or `./data/MEEI` |
| `--data_split_path` | Annotation and split directory | `./annotation` |
| `--clip-path` | CLIP ViT-B/16 checkpoint | `./weights/ViT-B-16.pt` |
| `--batch-size` | Training batch size | `128` |
| `--test-batch-size` | Evaluation batch size | `50` |
| `--epochs` | Number of training epochs | `100` |
| `--warmup_epochs` | Warm-up epochs | `5` |
| `--workers` | Data-loader workers | `8` |
| `--lr` | Learning rate | `1e-3` |
| `--weight_decay` | Weight decay | `0.05` |
| `--unfreeze_last_n` | Number of late CLIP blocks to unfreeze | `2` |
| `--seed` | Random seed | script-dependent |

Run `python <script>.py --help` because the exact option set can differ between entry points.


## License and Third-Party Materials

The original research code is released under the MIT License in `LICENSE`. AFLFP, MEEI, CLIP weights, and other third-party materials remain subject to their own licenses, access requirements, and attribution rules.

## Citation

The formal citation will be added after the conference publication metadata is available.


