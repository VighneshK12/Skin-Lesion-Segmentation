
````markdown
# Skin Lesion Segmentation with nnU-Net v2 (ISIC)

This repo contains the configuration and trainer patch used to train nnU-Net v2
on ISIC skin lesion segmentation data.

## Environment

- Python 3.9
- nnUNetv2
- PyTorch (CUDA build matching your GPU)

---

## 1) Clone repo + create environment

### Linux / macOS

```bash
git clone https://github.com/VighneshK12/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation

python3 -m venv .venv
source .venv/bin/activate

# install torch first (user can adjust CUDA version)
# Example: CUDA 11.8 build
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision

# then project deps
pip install --upgrade pip
pip install -r requirements.txt
````

### Windows (PowerShell)

```powershell
git clone https://github.com/VighneshK12/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation

py -m venv .venv
.\.venv\Scripts\Activate.ps1

# choose correct torch build (GPU or CPU-only)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Apply the ISICTrainer patch

```bash
# env already activated
python scripts/apply_isic_patch.py
```

---

## 3) Datasets: raw + preprocessed layout

* Download **raw ISIC data** (link…).
* Download **preprocessed ISIC nnU-Net data** (link…).

Create the following layout:

```text
/path/to/nnunet_data/
  nnUNet_raw/
    Dataset201_ISIC/...
  nnUNet_preprocessed/
    Dataset201_ISIC/...
  nnUNet_results/
```

**Recommended:** use the **preprocessed** archive (faster to start).
**Optional:** you can regenerate preprocessing from raw.

Example unzip instructions (Linux):

```bash
# Create the three standard nnU-Net folders
mkdir -p /path/to/nnunet_data/nnUNet_raw
mkdir -p /path/to/nnunet_data/nnUNet_preprocessed
mkdir -p /path/to/nnunet_data/nnUNet_results

# raw
unzip ISIC_raw_Dataset201.zip -d /path/to/nnunet_data/nnUNet_raw

# preprocessed
unzip ISIC_preprocessed_Dataset201.zip -d /path/to/nnunet_data/nnUNet_preprocessed
```

---

## 4) Set nnU-Net environment variables

These env vars must be set in the same shell where you run `nnUNetv2_*` commands.

### Linux / macOS (bash/zsh)

```bash
export nnUNet_raw=/path/to/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/path/to/nnunet_data/nnUNet_results
```

### Windows (PowerShell)

```powershell
$env:nnUNet_raw="C:\path\to\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results="C:\path\to\nnunet_data\nnUNet_results"
```

---

## 5) Optional: run preprocessing from raw

```bash
# env activated and nnUNet_* variables set
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity
```

* `-d 201` – dataset ID used for ISIC in this project
* `-c 2d` – 2D configuration (dermoscopy images)

---

## 6) Training with ISICTrainer (main command)

Once env, data, and patch are ready, this is the core command:

### Single fold example (fold 0, 2D config)

```bash
nnUNetv2_train 201 2d 0 -tr ISICTrainer
```

### All folds 0–4 (Linux)

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 201 2d $FOLD -tr ISICTrainer
done
```

Where:

* `201` – ISIC dataset ID
* `2d` – nnU-Net configuration (2D)
* `$FOLD` – cross-validation fold (0–4)
* `-tr ISICTrainer` – use the custom trainer defined by this repo’s patch

```

If you want, next step we can add a small “Results & Reproducibility” section with your Dice/IoU table and which fold/settings produced them.
```
