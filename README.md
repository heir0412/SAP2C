# SAP2C

Source-code implementation for **SAP2C: Automated Eulerian Cinemagraph
Synthesis from a Single Ancient Painting**.

## Pipeline

1. **Image Transfer Block (ITB).** Qwen-VL-Max extracts a scene description
   from the input painting. A deterministic artistic-vocabulary lexicon removes
   style descriptors, and Qwen-Image-Edit-Plus uses the processed prompt to
   generate a photorealistic semantic proxy.
2. **Mask Generation Block (MGB).** DINOv3-Large (ViT-L/16) provides
   structural features from the ancient painting, while LISA-13B provides
   realistic-domain semantic features. A pixel-wise MLP fuses
   `[M_anc, F_anc, M_real, F_real]` through hidden layers of 256, 128, and 64
   units.
3. **Cinemagraph Generation Block (CGB).** Semantic-Aware Flow Refinement
   (SAFR) smooths motion within semantic regions. Semantic-Gated Feature
   Splatting (SGFS) bidirectionally aggregates warped features using
   semantic-consistency kernels.

## Demos

![Demo 00005](assets/demos/00005.gif)

![Demo 00001](assets/demos/00001.gif)

![Demo 00011](assets/demos/00011.gif)

## Important files

```text
config/config.yaml                         Runtime configuration
main.py                                    Pipeline entry point
model/realistic_generation/generation.py  Qwen-VL-Max and Qwen-Image-Edit-Plus proxy construction
model/realistic_generation/artistic_lexicon.json  Deterministic style-mapping lexicon
model/dinov3/                              Ancient Painting Stream
model/LISA/                                Realistic Image Stream
model/mask_fusion/mlp_fusion.py            Four-input feature-fusion MLP
video_inference.py                         SAFR and SGFS video synthesis
```

## Configuration

Set `DASHSCOPE_API_KEY` in your environment before invoking the image-transfer
stage. Configure local paths for the DINOv3-Large, LISA-13B, and MGB
checkpoints in `config/config.yaml`. The proxy-construction stage records the
intermediate `p_anc` and `p_real` prompts next to each generated proxy image.

```bash
export DASHSCOPE_API_KEY="your-key"
python main.py --help
```

The source code is publicly available at
<https://github.com/heir0412/SAP2C>.
