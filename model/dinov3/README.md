# DINOv3 Ancient Painting Stream

This directory contains the SAP2C wrapper for the Ancient Painting Stream.
It uses **DINOv3-Large (ViT-L/16)** to predict the ancient-domain fluid mask
`M_anc` and to save the structural feature tensor `F_anc` for the Mask
Generation Block.

`inference.py` writes the following files beside each processed input image:

```text
<name>_mask.jpg          Predicted ancient-domain mask (M_anc)
<name>_Sx.npy            Semantic probability map for SAFR/SGFS
<name>_anc_features.npy  DINOv3 structural feature tensor (F_anc)
```

Configure the local ViT-L/16 backbone and segmentation checkpoint paths in
`../../config/config.yaml`. The module is called by the root-level `main.py`
pipeline; it is not a standalone training release.
