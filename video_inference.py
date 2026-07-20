import os
import argparse
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import moviepy.editor
import tqdm

from options.test_options import TestOptions
from models.sync_batchnorm import convert_model
from models.models import create_model
import util.util as util
from util.util import load_image


def _drop_cli_option(argv, option, takes_value=True):
    cleaned = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            skip_next = takes_value
            continue
        if item.startswith(f"{option}="):
            continue
        cleaned.append(item)
    return cleaned


# ------------------------------
# 基础工具函数
# ------------------------------
def _to_device(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _load_flow_any(path):
    """加载光流（支持 .npy / .pt），返回 [1,2,H,W] float32"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        flow = np.load(path)
        if flow.ndim == 3 and flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))  # [2,H,W] -> [H,W,2]
        assert flow.ndim == 3 and flow.shape[2] == 2, f"Bad npy flow shape: {flow.shape}"
        flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
        return flow.float()
    elif ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "flow" in obj:
            obj = obj["flow"]
        if not isinstance(obj, torch.Tensor):
            raise ValueError("Unsupported .pt content. Expect Tensor or dict with key 'flow'.")
        flow = obj.float()
        if flow.dim() == 3 and flow.shape[0] == 2:  # [2,H,W]
            flow = flow.unsqueeze(0)  # [1,2,H,W]
        elif flow.dim() == 4 and flow.shape[1] == 2:  # [B,2,H,W]
            pass
        else:
            raise ValueError(f"Bad pt flow shape: {tuple(flow.shape)}")
        return flow
    else:
        raise ValueError(f"Unsupported flow extension: {ext}")


def _load_semantic_any(path):
    """
    加载语义概率图或标签图，返回 [1,C,H,W] float32
    """
    ext = os.path.splitext(path)[1].lower()
    if ext != ".npy":
        raise ValueError(f"Unsupported semantic format: {ext}")

    Sx = np.load(path)
    if Sx.ndim == 2:  # 单通道标签 [H,W]
        num_classes = int(Sx.max() + 1)
        one_hot = np.eye(num_classes, dtype=np.float32)[Sx.astype(int)]  # [H,W,C]
        Sx = np.transpose(one_hot, (2, 0, 1))  # [C,H,W]
    elif Sx.ndim == 3 and Sx.shape[0] < Sx.shape[1]:
        Sx = Sx.astype(np.float32)  # [C,H,W]
    else:
        raise ValueError(f"Unexpected Sx shape: {Sx.shape}")
    return torch.from_numpy(Sx).unsqueeze(0).float()  # [1,C,H,W]


# ------------------------------
# 语义引导光流平滑 (SAFR)
# ------------------------------
def semantic_guided_flow(flow, semantics, alpha=0.8, kernel_size=7):
    """
    语义引导光流平滑
    """
    C = semantics.shape[1]
    smoothed = torch.zeros_like(flow)
    weight_sum = torch.zeros_like(flow)

    pad = (kernel_size // 2,) * 4  # (l, r, t, b)
    for c in range(C):
        mask = semantics[:, c:c + 1, :, :]  # [1,1,H,W]
        masked_flow = flow * mask
        # 对 flow 和 mask 分别做反射填充，再平均池化
        smooth = F.avg_pool2d(F.pad(masked_flow, pad, mode='reflect'),
                              kernel_size=kernel_size, stride=1, padding=0)
        w = F.avg_pool2d(F.pad(mask, pad, mode='reflect'),
                         kernel_size=kernel_size, stride=1, padding=0)
        smoothed += smooth
        weight_sum += w

    weight_sum = torch.clamp(weight_sum, min=1e-5)
    flow_smooth = smoothed / weight_sum
    return alpha * flow + (1 - alpha) * flow_smooth


def laplacian_smooth_flow(flow, lam=0.05):
    """拉普拉斯平滑"""
    left = F.pad(flow, (1, 0, 0, 0), mode='replicate')[:, :, :, :-1]
    right = F.pad(flow, (0, 1, 0, 0), mode='replicate')[:, :, :, 1:]
    up = F.pad(flow, (0, 0, 1, 0), mode='replicate')[:, :, :-1, :]
    down = F.pad(flow, (0, 0, 0, 1), mode='replicate')[:, :, 1:, :]
    lap = -4 * flow + left + right + up + down
    return flow + lam * lap


def edge_smooth_flow(flow, edge_width=15):
    """边缘衰减"""
    _, _, H, W = flow.shape
    yy = torch.linspace(0, 1, steps=H, device=flow.device).view(1, 1, H, 1).expand(1, 1, H, W)
    xx = torch.linspace(0, 1, steps=W, device=flow.device).view(1, 1, 1, W).expand(1, 1, H, W)
    dist_min = torch.minimum(torch.minimum(xx, 1 - xx), torch.minimum(yy, 1 - yy))
    mask = torch.clamp(dist_min * (min(H, W) / (2.0 * edge_width)), 0.0, 1.0)
    return flow * mask


# ------------------------------
# 【新增核心模块】语义门控特征喷溅
# ------------------------------
class SemanticGatedFeatureSplatting(nn.Module):
    """
    实现公式: D_n(p) = [ sum( w * D_warped * Psi ) ] / [ sum( w * Psi ) ]
    其中 Psi(F_anc(p), F_anc(p')) 是语义一致性核函数。
    """

    def __init__(self, temperature=10.0, consistency_thresh=0.5):
        super().__init__()
        self.temperature = temperature
        self.thresh = consistency_thresh

    def get_consistency_kernel(self, sem_target, sem_warped):
        """Compute the semantic consistency kernel Psi(p, p'_k)."""
        similarity = torch.sum(sem_target * sem_warped, dim=1, keepdim=True)
        psi = torch.exp(self.temperature * (similarity - 1.0))
        return torch.where(psi >= self.thresh, psi, torch.zeros_like(psi))

    @staticmethod
    def _sample(tensor, flow):
        """Back-warp a tensor to the target grid for one source direction."""
        batch, _, height, width = tensor.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=tensor.device),
            torch.arange(width, device=tensor.device),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(batch, -1, -1, -1)
        source = base + flow
        source[:, 0] = 2.0 * source[:, 0] / max(width - 1, 1) - 1.0
        source[:, 1] = 2.0 * source[:, 1] / max(height - 1, 1) - 1.0
        return F.grid_sample(
            tensor, source.permute(0, 2, 3, 1), mode="bilinear",
            padding_mode="border", align_corners=False,
        )

    def forward(self, feature, flow, semantics):
        """
        Args:
            feature:   当前内容特征/图像 [B, C, H, W]
            flow:      光流场 [B, 2, H, W] (p -> p')
            semantics: 语义图 [B, S, H, W]
        Returns:
            Semantic-gated, bidirectionally splatted feature tensor.
        """
        numerator = torch.zeros_like(feature)
        denominator = torch.zeros_like(feature[:, :1])
        # Each direction supplies a source k with its own p'_k, semantic kernel,
        # and splatting weight omega_k.
        for source_flow in (flow, -flow):
            warped_feature = self._sample(feature, source_flow)
            warped_semantics = self._sample(semantics, source_flow)
            psi = self.get_consistency_kernel(semantics, warped_semantics)
            omega = torch.exp(-source_flow.norm(dim=1, keepdim=True))
            numerator = numerator + omega * psi * warped_feature
            denominator = denominator + omega * psi
        return numerator / denominator.clamp_min(1e-6)


# ------------------------------
# 主推理流程
# ------------------------------
def _default_semantics_path(image_path):
    root, _ = os.path.splitext(image_path)
    return f"{root}_Sx.npy"


def generate_video_single(opt, model, image_path, flow_path, n_frames=60, fps=30,
                          out_mp4_path=None, out_gif_path=None,
                          semantics_path=None, safr_alpha=0.8, safr_kernel=7,
                          lap_lambda=0.05, edge_width=15,
                          use_safr=True, use_sgfs=True):
    device = next(model.parameters()).device

    # 1️⃣ 读取图像
    image = load_image(image_path, target_width=opt.fineSize, target_height=opt.fineSize).unsqueeze(0)
    image = _to_device(image, device)

    # 2️⃣ 读取光流
    flow = _load_flow_any(flow_path)
    _, _, Hf, Wf = flow.shape
    motion_norm = getattr(opt, "motion_norm", 1.0)
    flow = flow * motion_norm
    flow = torch.clamp(flow, -1, 1)
    flow = flow * opt.speed

    # 将光流缩放到目标分辨率
    scale_xy = torch.tensor([opt.fineSize / Wf, opt.fineSize / Hf], dtype=torch.float32).view(1, 2, 1, 1)
    flow = flow * scale_xy
    flow = F.interpolate(flow, (opt.fineSize, opt.fineSize), mode="bilinear", align_corners=False)
    flow = _to_device(flow, device)

    # 3️⃣ 语义处理：平滑 + 准备门控模块
    sem_path = semantics_path or _default_semantics_path(image_path)
    has_semantics = os.path.exists(sem_path) and (use_safr or use_sgfs)

    # 初始化语义门控模块
    splatter = SemanticGatedFeatureSplatting(temperature=5.0, consistency_thresh=0.2).to(device)

    if has_semantics:
        semantics = _load_semantic_any(sem_path)
        if semantics.abs().sum() < 1e-6:
            print(f"[Warning] Semantic map {sem_path} is blank.")
            has_semantics = False
        else:
            semantics = F.interpolate(semantics, (opt.fineSize, opt.fineSize), mode="bilinear", align_corners=False)
            semantics = _to_device(semantics, device)

            # --- A. 语义感知光流优化 (SAFR) ---
            if use_safr:
                flow_refined = semantic_guided_flow(flow, semantics, alpha=safr_alpha, kernel_size=safr_kernel)
                flow_refined = laplacian_smooth_flow(flow_refined, lam=lap_lambda)
                flow_refined = edge_smooth_flow(flow_refined, edge_width=edge_width)

                if torch.isnan(flow_refined).any():
                    print("[Warning] invalid flow values, fallback to original.")
                else:
                    flow = flow_refined

            print(f"[Info] Semantic processing enabled for {os.path.basename(image_path)}")
    else:
        print(f"[Warning] No semantic map found for {image_path}")
        semantics = None

    # 4️⃣ 构造输入
    model_input = torch.cat([image, image, flow], dim=1)

    data = {
        "label": model_input,
        "inst": model_input,
        "image": model_input,
        "feat": 0,
        "path": [str(image_path)],
        "index": []
    }

    # 5️⃣ 逐帧生成
    frames = []
    for i in tqdm.tqdm(range(n_frames), desc="Generating"):
        data["index"] = torch.tensor([0, i, n_frames - 1], device=device).unsqueeze(0)

        with torch.no_grad():
            out = model.inference(data["label"], data["inst"], data["image"], index=data["index"])

        raw_frame_tensor = out.data

        # --- B. 语义门控特征喷溅 ---
        if use_sgfs and has_semantics and semantics is not None:
            t = i / float(max(n_frames - 1, 1))
            current_flow = flow * t
            splatted_features = splatter(raw_frame_tensor, current_flow, semantics)
            frame = util.tensor2im(splatted_features[0])
        else:
            frame = util.tensor2im(raw_frame_tensor[0])

        frames.append(frame)

    # 6️⃣ 保存视频
    if out_mp4_path:
        os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
        moviepy.editor.ImageSequenceClip(
            [np.clip(f, 0, 255).astype(np.uint8) for f in frames], fps=fps
        ).write_videofile(out_mp4_path, verbose=False, logger=None)

    if out_gif_path:
        os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)
        pil_frames = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / max(fps, 1))
        pil_frames[0].save(out_gif_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)

    print(f"✅ Saved: {out_mp4_path}")


# ------------------------------
# 主函数入口
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--no_instance', action='store_true')
    parser.add_argument('--label_nc', type=int, default=0)
    parser.add_argument('--input_nc', type=int, default=8)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--fineSize', type=int, default=512)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='frame')
    parser.add_argument('--netG', type=str, default='spadexattnunetsd')
    parser.add_argument('--speed', type=float, default=1.0)

    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--flow_path', type=str, required=True)
    parser.add_argument('--semantics_path', type=str, default=None)
    parser.add_argument('--n_frames', type=int, default=60)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='results/single_video')

    # 关键超参
    parser.add_argument('--safr_alpha', type=float, default=0.8, help='语义平滑强度')
    parser.add_argument('--safr_kernel', type=int, default=7, help='语义平滑核大小')
    parser.add_argument('--lap_lambda', type=float, default=0.05, help='Laplacian平滑')
    parser.add_argument('--edge_width', type=int, default=15, help='边缘衰减宽度')
    parser.add_argument('--disable_safr', action='store_true',
                        help='消融：禁用语义感知光流优化 (SAFR)')
    parser.add_argument('--disable_sgfs', action='store_true',
                        help='消融：禁用语义门控特征喷溅 (SGFS)')

    args = parser.parse_args()
    sys.argv = _drop_cli_option(sys.argv, "--semantics_path", takes_value=True)
    sys.argv = _drop_cli_option(sys.argv, "--disable_safr", takes_value=False)
    sys.argv = _drop_cli_option(sys.argv, "--disable_sgfs", takes_value=False)

    opt = TestOptions().parse(save=False)
    opt.name = args.name
    opt.no_instance = args.no_instance
    opt.label_nc = args.label_nc
    opt.input_nc = args.input_nc
    opt.output_nc = args.output_nc
    opt.fineSize = args.fineSize
    opt.batchSize = args.batchSize
    opt.dataset_name = args.dataset_name
    opt.netG = args.netG
    opt.speed = args.speed
    opt.nThreads = 1
    opt.serial_batches = True
    opt.no_flip = True
    if not hasattr(opt, 'motion_norm'):
        opt.motion_norm = 1.0

    model = create_model(opt)
    if hasattr(opt, "norm_G") and 'sync' in opt.norm_G:
        model = convert_model(model)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    base = os.path.splitext(os.path.basename(args.image_path))[0]
    out_mp4 = os.path.join(args.save_dir, f'{base}_video.mp4')
    out_gif = os.path.join(args.save_dir, f'{base}.gif')

    generate_video_single(
        opt=opt,
        model=model,
        image_path=args.image_path,
        flow_path=args.flow_path,
        n_frames=args.n_frames,
        fps=args.fps,
        out_mp4_path=out_mp4,
        out_gif_path=out_gif,
        semantics_path=args.semantics_path,
        safr_alpha=args.safr_alpha,
        safr_kernel=args.safr_kernel,
        lap_lambda=args.lap_lambda,
        edge_width=args.edge_width,
        use_safr=not args.disable_safr,
        use_sgfs=not args.disable_sgfs,
    )


if __name__ == '__main__':
    main()
