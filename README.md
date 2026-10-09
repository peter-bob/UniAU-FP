# UniAU-FP

Official implementation of **UniAU-FP**, a vision-language framework for unilateral facial Action Unit (AU) intensity estimation in facial paralysis assessment.

UniAU-FP predicts five AUs independently on the anatomical left and right sides of the face: `AU02`, `AU04`, `AU06`, `AU15`, and `AU43`. The model combines CLIP visual features, hierarchical bilateral asymmetry modeling, AU-aware text prompts, cross-modal attention, and an auxiliary asymmetry prediction branch.

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
|-- train_fer_first_stage_AFLFP_3.py    AFLFP training entry point
|-- train_fer_first_stage_AFLFP_4.py    AFLFP training entry point
|-- train_fer_first_stage_MEEI_3.py     MEEI training entry point
|-- train_fer_first_stage_MEEI_4.py     MEEI training entry point
|-- train_ablation_AFLFP.py             AFLFP ablation experiments
|-- train_ablation_MEEI.py              MEEI ablation experiments
|-- attention_heatmap_compare_*.py      Attention and Grad-CAM visualization
|-- UniAU-FP_camera_ready_revised.tex   Manuscript source
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

## Data Preparation

The experiments use two facial palsy benchmarks:

| Dataset | Total images | Train | Test |
|---|---:|---:|---:|
| AFLFP | 3,460 | 2,775 | 685 |
| MEEI | 2,395 | 1,915 | 480 |

The original facial images are not redistributed here. Obtain AFLFP and MEEI from their official sources and follow their access conditions and licenses. The public snapshot currently contains the first split files, such as `annotation/AFLFP_1_train.txt` and `annotation/MEEI_1_train.txt`.

Place the locally obtained datasets as follows:

```text
UniAU-FP/
|-- data/
|   |-- AFLFP/
|   `-- MEEI/
|-- annotation/
`-- weights/
    `-- ViT-B-16.pt
```

The directory passed to `--data-path` must match the layout expected by the corresponding loader. The annotation files provide split/index information; they are not a replacement for the original datasets.

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

## Ablation Experiments

The ablation drivers compare four cumulative configurations:

| Configuration | Components |
|---|---|
| C1 | Baseline visual regression model |
| C2 | C1 plus cross-modal interaction |
| C3 | C2 plus bilateral asymmetry modeling |
| C4 | Full UniAU-FP with auxiliary asymmetry supervision |

Run the full configuration with:

```powershell
python train_ablation_AFLFP.py --ablation_config 4
python train_ablation_MEEI.py --ablation_config 4
```

Use `--ablation_config 1`, `2`, `3`, or `4` to select another configuration. For a new study, repeat each setting with multiple random seeds and report uncertainty across runs.

## Attention Visualization

`attention_heatmap_compare_FORCE5_BLUE.py` compares attention maps from baseline and full-model checkpoints. A typical AFLFP command is:

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

The visualization is qualitative. Anatomically plausible attention does not, by itself, establish clinical validity.

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

## Results

The following values are the reported single-run results from the accompanying manuscript:

| Dataset | RMSE (lower is better) | MAE (lower is better) |
|---|---:|---:|
| AFLFP | **0.183** | **0.124** |
| MEEI | **0.144** | **0.109** |

These values are reference results, not a guarantee of reproduction on a different hardware, software, or dataset release. Record the Git commit, dataset version, split files, random seed, CUDA/PyTorch versions, checkpoint checksum, and complete command line for each run.

## Paper

The manuscript source is included in `UniAU-FP_camera_ready_revised.tex`. Please verify the final experimental tables against the archived training logs before submitting or citing a camera-ready version.

## License and Third-Party Materials

The original research code is released under the MIT License in `LICENSE`. AFLFP, MEEI, CLIP weights, and other third-party materials remain subject to their own licenses, access requirements, and attribution rules.

## Citation

The formal citation will be added after the conference publication metadata is available.

## Contact

For implementation questions, open a GitHub issue with the operating system, Python/PyTorch versions, command line, and complete error message. Do not upload patient images or private data to an issue.
