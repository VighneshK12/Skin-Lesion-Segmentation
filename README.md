

# Skin Lesion Segmentation with nnU-Net v2 (ISIC)

This repo contains the configuration and trainer patch used to train nnU-Net v2
on ISIC skin lesion segmentation data.

## Environment

- Python 3.9
- nnUNetv2
- PyTorch (CUDA build matching your GPU)

---

## 1) Clone repo

```bash
git clone https://github.com/VighneshK12/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation
```
## Speed (Concordia ENCS cluster)
```bash
# from your laptop
ssh yourid@speed.encs.concordia.ca

# on Speed
cd /nfs/speed-scratch/$USER
git clone https://github.com/VighneshK12/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation
```
---

### 2) Create environment & install dependencies

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

# install torch first (user can adjust CUDA / CPU build)
# Example: CUDA 11.8 build
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision

# then project deps
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# choose correct torch build (GPU or CPU-only)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision

pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## Speed (Concordia ENCS, tcsh)
```bash
# load Python module (adjust version if needed)
module load python/3.9

cd /nfs/speed-scratch/$USER/Skin-Lesion-Segmentation

# create venv in scratch
python -m venv .venv

# tcsh activate script
source .venv/bin/activate.csh

# install torch (CUDA build on GPU nodes; adjust index URL if needed)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision

# install project dependencies (using constraints for reproducibility)
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

---

## 3) Apply the ISICTrainer patch

```bash
# env already activated
python scripts/apply_isic_patch.py
```

This copies `patches/ISICTrainer.py` into the installed `nnUNetv2` package and removes any stale `.pyc`.

---
## Speed (tcsh)
```bash
cd /nfs/speed-scratch/$USER/Skin-Lesion-Segmentation
source .venv/bin/activate.csh
python scripts/apply_isic_patch.py
```

## 4) Prepare nnU-Net folders & datasets

1. Download **raw ISIC data** (link here).
2. Download **preprocessed ISIC nnU-Net data** (link here – recommended).

Choose a base directory for all nnU-Net data, e.g. `/path/to/nnunet_data`:

```bash
mkdir -p /path/to/nnunet_data/nnUNet_raw
mkdir -p /path/to/nnunet_data/nnUNet_preprocessed
mkdir -p /path/to/nnunet_data/nnUNet_results
```

Target layout:

```text
/path/to/nnunet_data/
  nnUNet_raw/
    Dataset201_ISIC/...
  nnUNet_preprocessed/
    Dataset201_ISIC/...
  nnUNet_results/
```

Example unzip commands (Linux):

```bash
# raw
unzip ISIC_raw_Dataset201.zip -d /path/to/nnunet_data/nnUNet_raw

# preprocessed
unzip ISIC_preprocessed_Dataset201.zip -d /path/to/nnunet_data/nnUNet_preprocessed
```

---

## Speed (tcsh example layout)

```bash
# choose a base dir on scratch
setenv NNUNET_BASE /nfs/speed-scratch/$USER/nnunet_data

mkdir -p $NNUNET_BASE/nnUNet_raw
mkdir -p $NNUNET_BASE/nnUNet_preprocessed
mkdir -p $NNUNET_BASE/nnUNet_results

# after downloading zips to Speed (e.g. into /nfs/speed-scratch/$USER/downloads):

cd /nfs/speed-scratch/$USER/downloads

# raw
unzip ISIC_raw_Dataset201.zip -d $NNUNET_BASE/nnUNet_raw

# preprocessed
unzip ISIC_preprocessed_Dataset201.zip -d $NNUNET_BASE/nnUNet_preprocessed
```


## 5) Set nnU-Net environment variables

These must be set in the same shell where you run `nnUNetv2_*` commands.

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

## Speed (tcsh)
```bash
# assuming you used NNUNET_BASE=/nfs/speed-scratch/$USER/nnunet_data
setenv nnUNet_raw        /nfs/speed-scratch/$USER/nnunet_data/nnUNet_raw
setenv nnUNet_preprocessed /nfs/speed-scratch/$USER/nnunet_data/nnUNet_preprocessed
setenv nnUNet_results    /nfs/speed-scratch/$USER/nnunet_data/nnUNet_results
```

---

## 6) (Optional) Re-run preprocessing from raw

```bash
# env activated and nnUNet_* variables set
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity
```

* `-d 201` – dataset ID used for ISIC in this project
* `-c 2d` – 2D configuration (dermoscopy images)

---
## Speed (tcsh)

```bash
cd /nfs/speed-scratch/$USER/Skin-Lesion-Segmentation
source .venv/bin/activate.csh

# nnUNet_* env vars already set as above
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity
```

## 7) Training with ISICTrainer

Once env, data, and patch are ready, use:

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

### Speed (tcsh, interactive example)

```bash
cd /nfs/speed-scratch/$USER/Skin-Lesion-Segmentation
source .venv/bin/activate.csh

# env vars:
setenv nnUNet_raw        /nfs/speed-scratch/$USER/nnunet_data/nnUNet_raw
setenv nnUNet_preprocessed /nfs/speed-scratch/$USER/nnunet_data/nnUNet_preprocessed
setenv nnUNet_results    /nfs/speed-scratch/$USER/nnunet_data/nnUNet_results

# single fold
nnUNetv2_train 201 2d 0 -tr ISICTrainer

# or all folds 0–4
foreach FOLD (0 1 2 3 4)
  nnUNetv2_train 201 2d $FOLD -tr ISICTrainer
end

```

### All Credits go to:
```
@article{peng2023u,
  title={U-Net v2: Rethinking the Skip Connections of U-Net for Medical Image Segmentation},
  author={Peng, Yaopeng and Sonka, Milan and Chen, Danny Z},
  journal={arXiv preprint arXiv:2311.17791},
  year={2023}
}

```
