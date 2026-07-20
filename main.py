import argparse
import subprocess
import os
import shlex
import sys
import yaml
import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MASK_FUSION_DIRS = [
    os.path.join(PROJECT_ROOT, "models", "mask_fusion"),
    os.path.join(PROJECT_ROOT, "model", "mask_fusion"),
]

# ==========================================================
#                   通用工具函数
# ==========================================================

def q(value):
    return shlex.quote(str(value))


def resolve_repo_path(path, root_dir=PROJECT_ROOT, fix_model_dir=False):
    if path is None or not isinstance(path, str):
        return path
    if os.path.isabs(path):
        return path

    clean = path[2:] if path.startswith("./") else path
    candidates = [os.path.join(root_dir, clean)]
    if fix_model_dir and clean.startswith("models/"):
        candidates.append(os.path.join(root_dir, "model", clean[len("models/"):]))
    if fix_model_dir and clean.startswith("model/"):
        candidates.append(os.path.join(root_dir, "models", clean[len("model/"):]))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def normalize_config_paths(cfg, root_dir=PROJECT_ROOT):
    cfg["samples"]["root_dir"] = resolve_repo_path(cfg["samples"]["root_dir"], root_dir)
    cfg["samples"]["input_image"] = resolve_repo_path(cfg["samples"]["input_image"], root_dir)
    for key, value in list(cfg.get("paths", {}).items()):
        cfg["paths"][key] = resolve_repo_path(value, root_dir, fix_model_dir=True)
    return cfg


def run_python_script(script_path, args=""):
    """执行子模块 Python 脚本"""
    script_path = resolve_repo_path(script_path, PROJECT_ROOT, fix_model_dir=True)
    cmd = f"{q(sys.executable)} {q(script_path)} {args}"
    print(f"\n▶️ 运行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"❌ 模块执行失败: {script_path}")
    print(f"✅ 模块执行完成: {script_path}\n")


def semantic_boundary_weight(seg_mask):
    inv = 1 - (seg_mask > 0.7).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist = dist / (dist.max() + 1e-6)
    beta = np.exp(-((dist ** 2) / (2 * (0.12 ** 2))))
    return cv2.GaussianBlur(beta.astype(np.float32), (17, 17), 0)


def load_mlp_fusion_model(ckpt_path, feature_channels=256, hidden_channels=(256, 128, 64)):
    if not ckpt_path:
        raise ValueError("MLP fusion requires --fusion_mlp_ckpt or mask_fusion.mlp_ckpt in config.")
    ckpt_path = resolve_repo_path(ckpt_path, PROJECT_ROOT, fix_model_dir=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ MLP fusion checkpoint not found: {ckpt_path}")

    import torch

    mask_fusion_dir = next((p for p in MASK_FUSION_DIRS if os.path.isdir(p)), MASK_FUSION_DIRS[0])
    if mask_fusion_dir not in sys.path:
        sys.path.insert(0, mask_fusion_dir)
    from mlp_fusion import MaskFusionMLP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device)
    model = MaskFusionMLP(
        feature_channels=state.get("feature_channels", feature_channels),
        hidden_channels=tuple(state.get("hidden_channels", hidden_channels)),
    )
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    model.eval()
    return model, device


def spelling_aliases(name):
    aliases = [name]
    if "Image" in name:
        aliases.append(name.replace("Image", "Imgae"))
    if "Imgae" in name:
        aliases.append(name.replace("Imgae", "Image"))
    return list(dict.fromkeys(aliases))


def resolve_sample_dir(samples_root, sample_name):
    candidates = [os.path.join(samples_root, name) for name in spelling_aliases(sample_name)]
    for candidate in candidates:
        if os.path.isdir(candidate):
            if candidate != candidates[0]:
                print(f"⚠️ 样本目录自动纠正为: {candidate}")
            return candidate
    return candidates[0]


def find_sample_input_image(sample_dir, sample_name, configured_input):
    if os.path.exists(configured_input):
        return configured_input

    candidates = []
    configured_stem = os.path.splitext(os.path.basename(configured_input))[0]
    for stem in spelling_aliases(sample_name) + spelling_aliases(configured_stem):
        for ext in (".jpg", ".jpeg", ".png"):
            candidates.append(os.path.join(sample_dir, f"{stem}{ext}"))
    candidates = list(dict.fromkeys(candidates))

    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"⚠️ 配置输入不存在，自动改用: {candidate}")
            return candidate

    existing_images = []
    if os.path.isdir(sample_dir):
        existing_images = [
            os.path.join(sample_dir, name)
            for name in sorted(os.listdir(sample_dir))
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(existing_images) == 1:
            print(f"⚠️ 配置输入不存在，自动改用样本目录唯一图片: {existing_images[0]}")
            return existing_images[0]

    detail = "\n".join(f"   - {p}" for p in existing_images[:20])
    if not detail:
        detail = "   - 未找到任何 .jpg/.jpeg/.png"
    raise FileNotFoundError(
        f"❌ 输入图像不存在: {configured_input}\n"
        f"样本目录: {sample_dir}\n"
        "请检查 sample_name、input_image、目录名是否一致。\n"
        "当前样本目录下已有图片:\n"
        f"{detail}"
    )

# ==========================================================
#                   阶段函数定义
# ==========================================================

def stage_realistic(cfg, sample_dir, base_name, input_image, batch=False, overwrite=False):
    print("🪄 [Stage 1] Realistic Generation")
    output_image = os.path.join(sample_dir, f"{base_name}_realistic.jpg")
    if batch:
        args = f"--samples_root {q(cfg['samples']['root_dir'])}"
        if overwrite:
            args += " --overwrite"
    else:
        args = f"--input {q(input_image)} --output {q(output_image)}"
    realistic_cfg = cfg.get("realistic_generation", {})
    args += (
        f" --vision_language_model {q(realistic_cfg.get('vision_language_model', 'qwen-vl-max'))}"
        f" --image_edit_model {q(realistic_cfg.get('image_edit_model', 'qwen-image-edit-plus'))}"
        f" --lexicon_path {q(resolve_repo_path(realistic_cfg['lexicon_path']))}"
        f" --fixed_suffix {q(realistic_cfg.get('fixed_suffix', 'realistic photograph'))}"
    )
    run_python_script(cfg["paths"]["realistic_generation"], args)
    return output_image


def stage_lisa(cfg, sample_dir, base_name, realistic_image):
    print("🎭 [Stage 2] LISA Segmentation")
    lisa_script = cfg["paths"]["lisa_segmentation"]
    output_mask = os.path.join(sample_dir, f"{base_name}_realistic_mask.jpg")
    output_features = os.path.join(sample_dir, f"{base_name}_realistic_features.npy")
    lisa_cfg = cfg.get("lisa", {})
    run_python_script(
        lisa_script,
        f"--input {q(realistic_image)} --output {q(output_mask)} "
        f"--feature_output {q(output_features)} --version {q(lisa_cfg['version'])}",
    )
    return output_mask


def stage_dinov3(cfg, sample_dir, base_name, input_image):
    """
    Stage 3: 调用 DINOv3 语义分割模块（批量模式）
    """
    print("🧩 [Stage 3] DINOv3 Segmentation")

    seg_script = cfg["paths"]["dinov3_segmentation"]
    seg_ckpt = cfg["paths"]["ckpt"]
    dino_ckpt = cfg["paths"]["dino_ckpt"]
    repo_dir = cfg["paths"]["repo_dir"]

    # 从 config 中获取输入尺寸与阈值
    input_h, input_w = cfg["dinov3"]["input_size"]
    thr = cfg["dinov3"]["dice_thr"]

    # ✅ 关键：改为批量根目录
    samples_root = cfg["samples"]["root_dir"]

    # 构建命令参数
    cmd_args = (
        f"--samples_root {q(samples_root)} "
        f"--sample_name {q(cfg['samples']['sample_name'])} "
        f"--ckpt {q(seg_ckpt)} "
        f"--dino_ckpt {q(dino_ckpt)} "
        f"--repo_dir {q(repo_dir)} "
        f"--input_h {input_h} "
        f"--input_w {input_w} "
        f"--thr {thr}"
        f" --dino_size {q(cfg['dinov3'].get('backbone', 'l'))}"
    )

    # 调用 DINOv3 模块
    run_python_script(seg_script, cmd_args)

    # 返回掩码路径（当前样本的）
    seg_mask_path = os.path.join(sample_dir, f"{base_name}_mask.jpg")
    return seg_mask_path


def _load_feature_tensor(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} feature tensor: {path}")
    feature = np.load(path).astype(np.float32)
    if feature.ndim == 2:
        feature = feature[None, ...]
    if feature.ndim != 3:
        raise ValueError(f"{name} feature tensor must have shape [C, H, W], got {feature.shape}")
    return feature[None, ...]


def stage_fuse(sample_dir, base_name, lisa_mask, dinov3_mask, fusion_mlp_ckpt,
               fusion_threshold=0.5, feature_channels=256, hidden_channels=(256, 128, 64)):
    """Fuse M_anc, F_anc, M_real, and F_real with the MGB feature MLP."""
    print("🔗 [Stage 4] MLP Mask Fusion")

    # ===== 读取输入掩码 =====
    m1 = cv2.imread(lisa_mask, cv2.IMREAD_GRAYSCALE)    # LISA 掩码
    m2 = cv2.imread(dinov3_mask, cv2.IMREAD_GRAYSCALE) # DINOv3 掩码
    if m1 is None or m2 is None:
        raise FileNotFoundError("❌ 掩码读取失败，请检查路径")

    m1 = m1.astype(np.float32) / 255.0
    m2 = m2.astype(np.float32) / 255.0
    if m1.shape != m2.shape:
        m1 = cv2.resize(m1, (m2.shape[1], m2.shape[0]), interpolation=cv2.INTER_LINEAR)

    import torch

    model, device = load_mlp_fusion_model(
        fusion_mlp_ckpt, feature_channels=feature_channels, hidden_channels=hidden_channels
    )
    anc_feature_path = os.path.join(sample_dir, f"{base_name}_anc_features.npy")
    real_feature_path = os.path.join(sample_dir, f"{base_name}_realistic_features.npy")
    f_anc = _load_feature_tensor(anc_feature_path, "Ancient Painting Stream")
    f_real = _load_feature_tensor(real_feature_path, "Realistic Image Stream")
    m_anc = m2[None, None, ...].astype(np.float32)
    m_real = m1[None, None, ...].astype(np.float32)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(m_anc).to(device),
            torch.from_numpy(f_anc).to(device),
            torch.from_numpy(m_real).to(device),
            torch.from_numpy(f_real).to(device),
        )
        fused = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    # ===== 二值化输出 =====
    threshold = float(fusion_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"mask_fusion.threshold must be in [0, 1], got {threshold}")
    fused_bin = (fused > threshold).astype(np.uint8) * 255
    # 保存融合结果
    fused_bin_path = os.path.join(sample_dir, f"{base_name}_fused_mask.jpg")
    if not cv2.imwrite(fused_bin_path, fused_bin):
        raise RuntimeError(f"❌ 融合掩码保存失败: {fused_bin_path}")
    print(f"✅ 二值化掩码已保存: {fused_bin_path}")

    return fused_bin_path


def stage_motion(cfg, sample_dir, base_name, realistic_image, selected_mask,
                 motion_prompt=None, use_prompt=True):
    print("🌊 [Stage 5] Motion Generation")

    # 提取路径与配置
    single_image_path = realistic_image
    single_mask_path = selected_mask
    tokenizer_path = cfg["paths"]["tokenizer_path"]

    # 构造命令
    cmd_args = (
        f"--name {cfg['optical_flow']['model_name']} "
        f"--which_epoch latest "
        f"--inference_only "
        f"--single_image {q(single_image_path)} "
        f"--single_mask {q(single_mask_path)} "
        f"--input_nc {cfg['optical_flow']['input_nc']} "
        f"--output_nc {cfg['optical_flow']['output_nc']} "
        f"--fineSize {cfg['optical_flow']['fineSize']} "
        f"--out_size 512 "
        f"--no_instance "
        f"--label_nc 0 "
        f"--netG {cfg['optical_flow']['netG']} "
        f"--batchSize 16"
    )
    if use_prompt:
        cmd_args += f" --tokenizer_path {q(tokenizer_path)} --use_prompts"
        if motion_prompt is not None:
            cmd_args += f" --single_prompt {q(motion_prompt)}"

    # 执行推理脚本
    motion_script = os.path.join(cfg["paths"]["motion_generation"], "test_motion.py")
    run_python_script(motion_script, cmd_args)

    # 自动匹配输出文件（和 test_motion.py 一致）
    base_name_no_ext = os.path.splitext(os.path.basename(single_image_path))[0]
    flow_file = os.path.join(os.path.dirname(single_image_path), f"{base_name_no_ext}_Motion.pt")
    flow_vis = os.path.join(os.path.dirname(single_image_path), f"{base_name_no_ext}_flow_masked.jpg")

    print(f"✅ Motion flow tensor: {flow_file}")
    print(f"✅ Motion flow visualization: {flow_vis}")
    return flow_file


def stage_video(cfg, sample_dir, base_name, input_image, flow_file,
                disable_safr=False, disable_sgfs=False):
    print("🎞 [Stage 6] Video Generation")
    video_script = cfg["paths"]["video_generation"]

    # ✅ 改参数名与 test_s_video.py 保持一致
    cmd = (
        f"--name video-pretrained "
        f"--no_instance "
        f"--label_nc 0 "
        f"--input_nc 8 "
        f"--output_nc 3 "
        f"--fineSize 512 "
        f"--batchSize 1 "
        f"--dataset_name frame "
        f"--netG spadeunet4softmaxsplating "
        f"--speed 1.0 "
        f"--image_path {q(input_image)} "
        f"--flow_path {q(flow_file)} "
        f"--n_frames 60 "
        f"--fps 30 "
        f"--save_dir {q(sample_dir)}"
    )

    if disable_safr:
        cmd += " --disable_safr"
    if disable_sgfs:
        cmd += " --disable_sgfs"
    run_python_script(video_script, cmd)
    return os.path.join(sample_dir, f"{base_name}_video.mp4")



# ==========================================================
#                   主控制流程
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Cinemagraph 全流程主控程序（支持阶段执行）")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["realistic", "lisa", "dinov3", "fuse", "motion", "video", "all"],
                        help="指定执行阶段")
    parser.add_argument("--config", type=str, default=None,
                        help="自定义配置文件路径（可选）")
    parser.add_argument("--mask_mode", type=str, default="fused",
                        choices=["dinov3", "lisa", "fused"],
                        help="motion 阶段使用的掩码来源")
    parser.add_argument("--fusion_mlp_ckpt", type=str, default=None,
                        help="覆盖 config 中的 MLP fusion checkpoint 路径")
    parser.add_argument("--motion_prompt", type=str, default=None,
                        help="motion 阶段使用的文本 prompt")
    parser.add_argument("--no_prompt", action="store_true",
                        help="motion 阶段使用 neutral prompt 做消融；不设置时由 test_motion.py 使用内置通用 prompt")
    parser.add_argument("--neutral_prompt", type=str, default="motion",
                        help="--no_prompt 消融时使用的 neutral prompt")
    parser.add_argument("--motion_image_mode", choices=["realistic", "original"],
                        default="realistic",
                        help="motion 阶段使用写实图或原图")
    parser.add_argument("--video_image_mode", choices=["original", "realistic"],
                        default="original",
                        help="video 阶段的渲染底图；正式结果应使用 original")
    parser.add_argument("--disable_safr", action="store_true",
                        help="video 阶段禁用 SAFR")
    parser.add_argument("--disable_sgfs", action="store_true",
                        help="video 阶段禁用 SGFS")
    parser.add_argument("--realistic_batch", action="store_true",
                        help="realistic 阶段处理 samples.root_dir 下所有样本文件夹")
    parser.add_argument("--realistic_overwrite", action="store_true",
                        help="realistic 批量模式下覆盖已有 *_realistic.jpg")
    parser.add_argument("--skip_lisa", action="store_true",
                        help="跳过 LISA 推理，直接使用样本目录中已有的 *_realistic_mask.jpg")
    parser.add_argument("--skip_realistic", action="store_true",
                        help="跳过写实图生成，直接使用样本目录中已有的 *_realistic.jpg")
    parser.add_argument("--skip_dinov3", action="store_true",
                        help="跳过 DINOv3 推理，直接使用样本目录中已有的 *_mask.jpg 和 *_Sx.npy")
    args = parser.parse_args()

    # ===================== 自动定位配置文件 =====================
    root_dir = PROJECT_ROOT
    default_config_path = os.path.join(root_dir, "config", "config.yaml")
    config_path = args.config if args.config else default_config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 未找到配置文件: {config_path}\n"
                                f"请确保 config.yaml 位于 {default_config_path}")
    print(f"🗂️ 使用配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = normalize_config_paths(cfg, root_dir)
    # ===========================================================

    # 样本路径设置
    sample_dir = resolve_sample_dir(cfg["samples"]["root_dir"], cfg["samples"]["sample_name"])
    if args.stage == "realistic" and args.realistic_batch:
        input_image = cfg["samples"]["input_image"]
    else:
        input_image = find_sample_input_image(
            sample_dir,
            cfg["samples"]["sample_name"],
            cfg["samples"]["input_image"]
        )
    base_name = os.path.splitext(os.path.basename(input_image))[0]

    print(f"\n🎬 Cinemagraph Pipeline 启动")
    print(f"当前阶段: {args.stage.upper()}")
    print(f"输入图像: {input_image}\n")

    # 各阶段生成的文件
    realistic_path = os.path.join(sample_dir, f"{base_name}_realistic.jpg")
    lisa_mask = os.path.join(sample_dir, f"{base_name}_realistic_mask.jpg")
    dinov3_mask = os.path.join(sample_dir, f"{base_name}_mask.jpg")
    fused_mask = os.path.join(sample_dir, f"{base_name}_fused_mask.jpg")
    motion_base_name = base_name if args.motion_image_mode == "original" else f"{base_name}_realistic"
    flow_file = os.path.join(sample_dir, f"{motion_base_name}_Motion.pt")
    video_path = os.path.join(sample_dir, f"{base_name}_video.mp4")

    # ===================== 分阶段执行 =====================

    # Step 1
    if args.stage in ["realistic", "all"]:
        if args.skip_realistic:
            if not os.path.exists(realistic_path):
                raise FileNotFoundError(
                    f"⚠️ 已设置 --skip_realistic，但未找到已有写实图: {realistic_path}\n"
                    "请先把外部写实生成结果保存为该文件名。"
                )
            print(f"⏭️  跳过写实图生成，使用已有图像: {realistic_path}")
        else:
            realistic_path = stage_realistic(
                cfg, sample_dir, base_name, input_image,
                batch=args.realistic_batch,
                overwrite=args.realistic_overwrite
            )

    # Step 2
    if args.stage in ["lisa", "all"]:
        if args.skip_lisa:
            if not os.path.exists(lisa_mask):
                raise FileNotFoundError(
                    f"⚠️ 已设置 --skip_lisa，但未找到已有 LISA mask: {lisa_mask}\n"
                    "请先把外部 LISA 输出保存为该文件名。"
                )
            print(f"⏭️  跳过 LISA 推理，使用已有 mask: {lisa_mask}")
        else:
            if not os.path.exists(realistic_path):
                raise FileNotFoundError("⚠️ 未找到真实感图像，请先执行 --stage realistic")
            lisa_mask = stage_lisa(cfg, sample_dir, base_name, realistic_path)

    # Step 3
    if args.stage in ["dinov3", "all"]:
        if args.skip_dinov3:
            if not os.path.exists(dinov3_mask):
                raise FileNotFoundError(
                    f"⚠️ 已设置 --skip_dinov3，但未找到已有 DINOv3 mask: {dinov3_mask}\n"
                    "请先把 DINOv3 输出保存为该文件名，或使用 Python>=3.10 环境运行 DINOv3。"
                )
            print(f"⏭️  跳过 DINOv3 推理，使用已有 mask: {dinov3_mask}")
        else:
            dinov3_mask = stage_dinov3(cfg, sample_dir, base_name, input_image)


    # Step 4
    if args.stage in ["fuse", "all"]:
        fusion_cfg = cfg.get("mask_fusion", {})
        fused_mask = stage_fuse(
            sample_dir, base_name, lisa_mask, dinov3_mask,
            fusion_mlp_ckpt=args.fusion_mlp_ckpt or fusion_cfg.get("mlp_ckpt"),
            fusion_threshold=fusion_cfg.get("threshold", 0.5),
            feature_channels=fusion_cfg.get("feature_channels", 256),
            hidden_channels=tuple(fusion_cfg.get("hidden_channels", [256, 128, 64])),
        )

    # Step 5
    if args.stage in ["motion", "all"]:
        mask_by_mode = {
            "dinov3": dinov3_mask,
            "lisa": lisa_mask,
            "fused": fused_mask,
        }
        selected_mask = mask_by_mode[args.mask_mode]
        if not os.path.exists(selected_mask):
            raise FileNotFoundError(f"⚠️ 未找到 {args.mask_mode} mask: {selected_mask}")
        motion_prompt = args.neutral_prompt if args.no_prompt else args.motion_prompt
        motion_image = input_image if args.motion_image_mode == "original" else realistic_path
        if not os.path.exists(motion_image):
            raise FileNotFoundError(f"⚠️ 未找到 motion 输入图: {motion_image}")
        flow_file = stage_motion(
            cfg, sample_dir, base_name, motion_image, selected_mask,
            motion_prompt=motion_prompt,
            use_prompt=True
        )

    # Step 6
    if args.stage in ["video", "all"]:
        if not (os.path.exists(flow_file) and os.path.exists(dinov3_mask)):
            raise FileNotFoundError("⚠️ 请先执行 motion 与 dinov3 阶段")
        video_image = input_image if args.video_image_mode == "original" else realistic_path
        if not os.path.exists(video_image):
            raise FileNotFoundError(f"⚠️ 未找到 video 渲染底图: {video_image}")
        video_path = stage_video(
            cfg, sample_dir, base_name, video_image, flow_file,
            disable_safr=args.disable_safr,
            disable_sgfs=args.disable_sgfs,
        )

    print("\n✅ 当前阶段执行完成！")
    print(f"📦 输出目录: {sample_dir}")
    print("✨ 生成文件：")
    for p in [realistic_path, lisa_mask, dinov3_mask, fused_mask, flow_file, video_path]:
        if os.path.exists(p):
            print(f"   - {p}")


if __name__ == "__main__":
    main()
