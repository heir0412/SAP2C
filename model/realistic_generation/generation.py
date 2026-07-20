import os
import argparse
import base64
import mimetypes
import requests
import cv2
from dashscope import MultiModalConversation
import dashscope

# ==================== Dashscope API 基础配置 ====================
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


# --- 从环境变量读取 API 密钥 ---
def read_api_key():
    """Read the DashScope API key from DASHSCOPE_API_KEY."""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is not set. Configure it in your environment.")
    return key


# --- 将图像编码为 Base64 ---
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


# --- 下载生成结果并统一尺寸 ---
def download_image(image_url, save_path, target_size=(512, 512)):
    """下载图像并调整为固定大小"""
    response = requests.get(image_url)
    if response.status_code == 200:
        # 临时保存
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        # 读取并调整大小
        img = cv2.imread(tmp_path)
        if img is not None:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(save_path, resized)
            os.remove(tmp_path)
            print(f"✅ 图像已保存（512×512）: {save_path}")
            return True
        else:
            print(f"⚠️ 无法读取下载的图像内容: {image_url}")
            return False
    else:
        print(f"❌ 图像下载失败，状态码: {response.status_code}")
        return False


# --- 拼接原图与生成图 ---
def create_comparison(original_path, generated_path, save_path):
    try:
        orig = cv2.imread(original_path)
        gen = cv2.imread(generated_path)

        if orig is None or gen is None:
            print(f"⚠️ 无法读取拼接图像: {original_path} 或 {generated_path}")
            return

        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]))
        comparison = cv2.hconcat([orig, gen])
        cv2.imwrite(save_path, comparison)
        print(f"📷 生成对比图: {save_path}")
    except Exception as e:
        print(f"❌ 拼接图像时出错: {str(e)}")


def process_single_image(image_path, api_key, output_path=None, prompt=None, skip_existing=False):
    """处理单张图片，支持 jpg/jpeg/png。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 输入图像不存在: {image_path}")

    image_dir = os.path.dirname(image_path)
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = output_path or os.path.join(image_dir, f"{image_stem}_realistic.jpg")
    cmp_path = os.path.join(image_dir, f"{image_stem}_compare.jpg")

    if skip_existing and os.path.exists(output_path):
        print(f"⏭️ 已存在，跳过: {output_path}")
        return output_path

    print(f"\n🎨 正在处理图像: {image_path}")
    try:
        image_base64 = encode_file(image_path)

        # === Prompt 设置 ===
        prompt = prompt or "realistic high-resolution photograph. "

        negative_prompt = (
            "painting, brush strokes, illustration, cartoon, sketch, unreal, oil texture, low resolution, "
            "digital art, drawing, blur, noise, fake, distorted colors, fantasy, anime"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_base64},
                    {"text": prompt}
                ]
            }
        ]

        # 调用 Dashscope API
        response = MultiModalConversation.call(
            api_key=api_key,
            model="qwen-image-edit",
            messages=messages,
            stream=False,
            watermark=False,
            negative_prompt=negative_prompt
        )

        if response.status_code == 200:
            output_url = response.output.choices[0].message.content[0]["image"]

            # 下载并统一尺寸
            if download_image(output_url, output_path, target_size=(512, 512)):
                create_comparison(image_path, output_path, cmp_path)

            return output_path

        else:
            print(f"❌ API 调用错误: HTTP {response.status_code}, 信息: {response.message}")
            return None

    except Exception as e:
        print(f"⚠️ 处理 {image_path} 时出错: {str(e)}")
        return None


def find_folder_image(folder_path):
    folder_name = os.path.basename(folder_path)
    for ext in (".jpg", ".jpeg", ".png"):
        image_path = os.path.join(folder_path, f"{folder_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


# --- 处理单个文件夹（同名图片） ---
def process_single_folder(folder_path, api_key, prompt=None, skip_existing=False):
    folder_name = os.path.basename(folder_path)
    image_path = find_folder_image(folder_path)

    if not image_path:
        expected = ", ".join(os.path.join(folder_path, f"{folder_name}{ext}") for ext in (".jpg", ".jpeg", ".png"))
        print(f"⚠️ 跳过 {folder_name}: 未找到同名图片，期望其中之一: {expected}")
        return None

    return process_single_image(image_path, api_key, prompt=prompt, skip_existing=skip_existing)


# --- 主控制逻辑 ---
def process_all_samples(samples_root, api_key, prompt=None, skip_existing=True):
    """遍历 samples 下所有子文件夹，处理同名 jpg/jpeg/png。"""
    if not os.path.exists(samples_root):
        print(f"❌ 路径不存在: {samples_root}")
        return

    subfolders = [os.path.join(samples_root, d) for d in os.listdir(samples_root)
                  if os.path.isdir(os.path.join(samples_root, d)) and not d.startswith(".")]
    if not subfolders:
        print(f"❌ 未在 {samples_root} 下找到子文件夹。")
        return

    print(f"📁 共找到 {len(subfolders)} 个文件夹，开始批量生成...\n")
    ok_count = 0
    fail_count = 0
    for folder in sorted(subfolders):
        result = process_single_folder(folder, api_key, prompt=prompt, skip_existing=skip_existing)
        if result:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n✅ 批量处理完成：成功/已存在 {ok_count}，失败/跳过 {fail_count}")


# ==================== 主函数入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DashScope realistic image generation")
    parser.add_argument("--input", type=str, default=None, help="单张输入图片路径")
    parser.add_argument("--output", type=str, default=None, help="单张输出图片路径")
    parser.add_argument("--samples_root", type=str, default=None, help="批量样本根目录")
    parser.add_argument("--prompt", type=str, default=None, help="写实生成 prompt")
    parser.add_argument("--overwrite", action="store_true", help="批量模式下覆盖已有 *_realistic.jpg")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))  # model/realistic_generation/
    samples_root = args.samples_root or os.path.abspath(os.path.join(current_dir, "../../sample"))

    if args.input:
        print(f"📄 单图输入: {args.input}")
    else:
        print(f"📂 样本根目录: {samples_root}")

    api_key = read_api_key()

    if args.input:
        result = process_single_image(args.input, api_key, output_path=args.output, prompt=args.prompt)
        if not result:
            raise SystemExit(1)
    else:
        process_all_samples(samples_root, api_key, prompt=args.prompt, skip_existing=not args.overwrite)
