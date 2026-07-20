import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ==================== 工具函数 ====================

def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    """Tensor → BGR uint8 图像"""
    img = img_t.detach().cpu().float().clamp(0, 1)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_colormap(prob_map: np.ndarray, save_path: str):
    """概率图 S(x) → 彩色热力图"""
    prob_uint8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)
    return heatmap


def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将热力图与原图叠加"""
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


def find_input_image(sample_dir):
    folder_name = os.path.basename(sample_dir)
    for ext in IMAGE_EXTS:
        path = os.path.join(sample_dir, f"{folder_name}{ext}")
        if os.path.exists(path):
            return path

    candidates = []
    for name in sorted(os.listdir(sample_dir)):
        lower = name.lower()
        if not lower.endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(name)[0].lower()
        if any(token in stem for token in ["realistic", "mask", "overlay", "heat", "compare", "flow"]):
            continue
        candidates.append(os.path.join(sample_dir, name))
    return candidates[0] if candidates else None


# ==================== 单图推理 ====================

@torch.no_grad()
def infer_single(model, img_path, device, thr=0.5, size=(512, 512)):
    """对单张图像进行 DINOv3 分割推理"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)

    model.eval()
    logits, structural_features = model.forward_with_features(img_t)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > thr).float()

    Sx = probs[0, 0].detach().cpu().numpy()
    mask = (pred_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

    base_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    np.save(os.path.join(base_dir, f"{base_name}_Sx.npy"), Sx)
    np.save(
        os.path.join(base_dir, f"{base_name}_anc_features.npy"),
        structural_features[0].detach().cpu().numpy(),
    )
    cv2.imwrite(os.path.join(base_dir, f"{base_name}_mask.jpg"), mask)

    heatmap = save_colormap(Sx, os.path.join(base_dir, f"{base_name}_Sx_heat.jpg"))
    overlay = overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.5)
    cv2.imwrite(os.path.join(base_dir, f"{base_name}_overlay.jpg"), overlay)

    print(f"✅ 已处理: {img_path}")


# ==================== 模型加载 ====================

def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)


# ==================== 主函数 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch DINOv3 Semantic Segmentation (one image per sample folder)")
    parser.add_argument("--samples_root", type=str, default="../../samples",
                        help="样本根目录 (每个子目录包含一张同名图片)")
    parser.add_argument("--sample_name", type=str, default=None,
                        help="只处理指定样本子目录；默认处理 samples_root 下全部样本")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="DINOv3 模型权重 (.pth)")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="DINOv3 backbone 权重 (.pth)")
    parser.add_argument("--repo_dir", type=str, default="../../dinov3",
                        help="DINOv3 hubconf.py 所在目录")
    parser.add_argument("--dino_size", type=str, default="l", choices=["l", "b", "s"],
                        help="DINOv3 主干大小")
    parser.add_argument("--input_h", type=int, default=512)
    parser.add_argument("--input_w", type=int, default=512)
    parser.add_argument("--thr", type=float, default=0.5)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 加载 DINOv3 Backbone ---
    print(f"[Load DINOv3 backbone: {args.dino_size}]")
    if args.dino_size == "l":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitl16', source='local', weights=args.dino_ckpt)
    elif args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    # --- 构建模型 ---
    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone).to(device)
    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, map_location=device)

    # --- 批量遍历每个子文件夹 ---
    print(f"\n🔍 扫描样本目录: {args.samples_root}")
    if args.sample_name:
        sample_dirs = [os.path.join(args.samples_root, args.sample_name)]
    else:
        sample_dirs = sorted([
            os.path.join(args.samples_root, d)
            for d in os.listdir(args.samples_root)
            if os.path.isdir(os.path.join(args.samples_root, d)) and not d.startswith(".")
        ])

    if not sample_dirs:
        print("❌ 未找到任何子文件夹")
        return

    print(f"共发现 {len(sample_dirs)} 个样本目录，开始推理...\n")

    for sample_dir in tqdm(sample_dirs, desc="[DINOv3]"):
        target_img = find_input_image(sample_dir)

        if target_img is None or not os.path.exists(target_img):
            print(f"⚠️ 跳过: {sample_dir} 中未找到输入图像")
            continue

        infer_single(model, target_img, device, thr=args.thr, size=(args.input_w, args.input_h))

    print("\n🎯 所有样本处理完成。")


if __name__ == "__main__":
    main()
