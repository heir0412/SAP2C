# Cinemagraph

GitHub-ready inference code and paired segmentation dataset for the Cinemagraph project.

## Repository layout

```text
cinemagraph_github/
├── config/                 # Runtime configuration
├── data/                   # Data loaders used by the pipeline
├── dataset/
│   ├── train/
│   │   ├── image/          # 1,600 images
│   │   └── mask/           # 1,600 paired masks
│   ├── validation/
│   │   ├── image/          # 400 images
│   │   └── mask/           # 400 paired masks
│   └── test/
│       ├── image/          # 207 images
│       └── mask/           # 207 paired masks
├── img2img/                # Image-to-image inference code
├── model/
│   ├── dinov3/             # DINOv3-based segmentation module
│   ├── LISA/
│   ├── mask_fusion/
│   └── realistic_generation/
├── options/
├── util/
└── main.py
```

The train/validation split is deterministic (`seed=42`) and was generated only from the original `model/segdino/dataset/train` image and `orange_mask` pairs. The 207-pair test set is copied from the original validation images and `orange_mask` masks. The old `mask`/`color_mask` directories are not used.

## Mask fusion

The fuse stage uses only the trained per-pixel MLP in `model/mask_fusion/mlp_fusion.py`. Its four inputs are the LISA mask, DINOv3 mask, semantic map, and semantic boundary map. Configure the checkpoint and output threshold under `mask_fusion` in `config/config.yaml`.

## Excluded content

Model weights/checkpoints, API keys, caches, generated results, training code, and one-off experiment/evaluation scripts are intentionally excluded. Download required model weights separately and update `config/config.yaml` with local paths.

Set the DashScope key through the environment; never commit it:

```bash
cp .env.example .env
export DASHSCOPE_API_KEY="your-key"
```

## Dataset storage

The dataset is configured for Git LFS. Before committing it, install and initialize Git LFS:

```bash
git lfs install
git add .gitattributes dataset
```

## Run

Install the common inference dependencies first:

```bash
pip install -r requirements.txt
```

LISA and DINOv3 may require the additional dependencies listed in their own `requirements.txt` files. Review the local model paths in `config/config.yaml`, then run:

```bash
python main.py --help
```
