import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Semantic Segmentation")

    # 输入输出路径
    parser.add_argument("--input", required=False, type=str, default=None, help="输入图像路径")
    parser.add_argument("--output", required=False, type=str, default=None, help="输出掩码路径")
    parser.add_argument("--feature_output", required=False, type=str, default=None,
                        help="保存 Realistic Image Stream 特征张量 (.npy)")

    # 模型与版本
    parser.add_argument("--version", default="./model/LISA/LISA-13B-v1")

    # ✅ 自动根据输入路径选择 vis_save_path
    default_vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vis_output"))
    parser.add_argument("--vis_save_path", default=default_vis_path, type=str,
                        help="保存可视化输出的路径 (默认: 根据输入图像所在目录自动设置)")

    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"], help="precision for inference")
    parser.add_argument("--image_size", default=512, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--vision-tower", default="/mnt/f/survey_animate/cinemagraph/models/LISA/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1",
                        type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)



def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def run_inference(args, image_path, output_mask_path=None):
    os.makedirs(args.vis_save_path, exist_ok=True)

    # === 安全加载本地模型 ===
    version_path = os.path.abspath(args.version)
    if not os.path.isdir(version_path):
        raise FileNotFoundError(f"❌ 本地模型路径无效: {version_path}")

    print(f"🔹 正在加载本地模型: {version_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=version_path,
        local_files_only=True,  # ✅ 禁止联网
        trust_remote_code=True,  # ✅ 允许加载本地自定义模型结构
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Precision
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    image_clip = image_clip.to(dtype=torch_dtype)

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().to(dtype=torch_dtype)

    prompt = "Segment all water regions in the image, including rivers, oceans, lakes and  yellow water , ocean and lakes"
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    print(f"segment prompt:{prompt}")
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip, image, input_ids, resize_list, original_size_list,
        max_new_tokens=512, tokenizer=tokenizer,
    )

    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    print("text_output:", text_output)

    os.makedirs(args.vis_save_path, exist_ok=True)
    if pred_masks:
        pred_mask = pred_masks[0].detach().cpu().numpy()[0]
        prob_map = 1 / (1 + np.exp(-pred_mask))
        pred_mask_bin = (prob_map > 0.5).astype(np.uint8) * 255

        if output_mask_path:
            cv2.imwrite(output_mask_path, pred_mask_bin)
            print(f"✅ 掩码已保存: {output_mask_path}")
        else:
            default_save = os.path.join(args.vis_save_path, os.path.basename(image_path).replace('.jpg', '_mask.jpg'))
            cv2.imwrite(default_save, pred_mask_bin)
            print(f"✅ 掩码已保存: {default_save}")

        if args.feature_output:
            # The calibrated LISA probability map is retained as the
            # realistic-domain semantic feature tensor consumed by MGB.
            np.save(args.feature_output, prob_map.astype(np.float32)[None, ...])
            print(f"✅ Realistic Image Stream features saved: {args.feature_output}")

        return output_mask_path or default_save
    else:
        print("⚠️ 模型未返回掩码")
        return None


def main(raw_args=None):
    args = parse_args(raw_args or sys.argv[1:])
    image_path = args.input if args.input else "./imgs/00003.jpg"
    output_path = args.output if args.output else None
    run_inference(args, image_path, output_path)


if __name__ == "__main__":
    main()
