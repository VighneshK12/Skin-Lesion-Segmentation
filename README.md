# Skin Lesion Segmentation with nnU-Net v2 (ISIC)

This repo contains the configuration and trainer patch used to train nnU-Net v2
on ISIC skin lesion segmentation data.

## Environment

- Python 3.9
- nnUNetv2
- PyTorch (CUDA build matching your GPU)

### Quick setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python scripts/apply_isic_patch.py
