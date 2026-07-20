"""MLLM-guided construction of a photorealistic semantic proxy.

The pipeline follows the manuscript: Qwen-VL-Max first describes the input
painting, a deterministic lexicon removes painting-style expressions, and
Qwen-Image-Edit-Plus edits the original image using the resulting prompt.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path

import cv2
import dashscope
import numpy as np
import requests
from dashscope import MultiModalConversation

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

DEFAULT_PARSE_INSTRUCTION = (
    "Describe this Chinese landscape painting for image editing. Identify the "
    "main objects, water or other fluid regions, surrounding elements, spatial "
    "relationships, and composition. Preserve observable content, but do not "
    "invent objects. Return one concise scene description."
)
DEFAULT_LEXICON = {
    "ink wash": "",
    "ink-and-wash": "",
    "brushwork": "",
    "brush strokes": "",
    "traditional Chinese painting": "",
    "Chinese painting": "",
    "landscape painting": "landscape",
    "xuan paper": "",
    "rice paper": "",
    "calligraphic": "",
    "freehand": "",
    "literati": "",
}


def read_api_key():
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is not set. Configure it in your environment.")
    return key


def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image file: {file_path}")
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def load_lexicon(lexicon_path=None):
    lexicon = dict(DEFAULT_LEXICON)
    if lexicon_path:
        path = Path(lexicon_path)
        if not path.exists():
            raise FileNotFoundError(f"Artistic-vocabulary lexicon not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            custom = json.load(handle)
        if not isinstance(custom, dict):
            raise ValueError("The artistic-vocabulary lexicon must be a JSON object.")
        lexicon.update({str(k): str(v) for k, v in custom.items()})
    return lexicon


def style_map(description, lexicon):
    """Deterministically remove or replace artistic-style descriptors."""
    mapped = description
    for source, target in sorted(lexicon.items(), key=lambda item: len(item[0]), reverse=True):
        mapped = re.sub(rf"\b{re.escape(source)}\b", target, mapped, flags=re.IGNORECASE)
    mapped = re.sub(r"\s+", " ", mapped)
    mapped = re.sub(r"\s*([,.;:])\s*", r"\1 ", mapped)
    return mapped.strip(" ,.;:")


def response_text(response):
    """Extract text from DashScope's list- or dictionary-style response."""
    content = response.output.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts).strip()
    return str(content).strip()


def parse_scene(image_base64, api_key, model, instruction):
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": [{"image": image_base64}, {"text": instruction}]}],
        stream=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Visual semantic parsing failed: HTTP {response.status_code}; {response.message}")
    description = response_text(response)
    if not description:
        raise RuntimeError("Qwen-VL-Max returned an empty scene description.")
    return description


def edit_image(image_base64, prompt, api_key, model):
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": [{"image": image_base64}, {"text": prompt}]}],
        stream=False,
        watermark=False,
        negative_prompt=(
            "painting, brush strokes, illustration, cartoon, sketch, oil texture, "
            "digital art, drawing, blur, noise, distorted colors, fantasy, anime"
        ),
    )
    if response.status_code != 200:
        raise RuntimeError(f"Photorealistic image editing failed: HTTP {response.status_code}; {response.message}")
    content = response.output.choices[0].message.content
    if not isinstance(content, list) or not content or "image" not in content[0]:
        raise RuntimeError("Qwen-Image-Edit-Plus did not return an image URL.")
    return content[0]["image"]


def download_image(image_url, save_path, target_size=(512, 512)):
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()
    data = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise RuntimeError("Unable to decode the image returned by Qwen-Image-Edit-Plus.")
    data = cv2.resize(data, target_size, interpolation=cv2.INTER_CUBIC)
    if not cv2.imwrite(save_path, data):
        raise RuntimeError(f"Unable to save proxy image: {save_path}")


def process_single_image(
    image_path,
    api_key,
    output_path=None,
    vision_language_model="qwen-vl-max",
    image_edit_model="qwen-image-edit-plus",
    lexicon_path=None,
    fixed_suffix="realistic photograph",
    parse_instruction=DEFAULT_PARSE_INSTRUCTION,
    skip_existing=False,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image_stem = Path(image_path).stem
    output_path = output_path or str(Path(image_path).with_name(f"{image_stem}_realistic.jpg"))
    if skip_existing and os.path.exists(output_path):
        return output_path

    image_base64 = encode_file(image_path)
    description = parse_scene(image_base64, api_key, vision_language_model, parse_instruction)
    prompt = f"{style_map(description, load_lexicon(lexicon_path))}, {fixed_suffix}".strip(", ")
    image_url = edit_image(image_base64, prompt, api_key, image_edit_model)
    download_image(image_url, output_path)

    # Store the deterministic intermediate prompt beside the proxy for auditing.
    Path(output_path).with_suffix(".prompt.txt").write_text(
        f"p_anc:\n{description}\n\np_real:\n{prompt}\n", encoding="utf-8"
    )
    return output_path


def find_folder_image(folder_path):
    folder_name = os.path.basename(folder_path)
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(folder_path, f"{folder_name}{extension}")
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="MLLM-guided realistic semantic-proxy construction")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--samples_root", type=str, default=None)
    parser.add_argument("--vision_language_model", default="qwen-vl-max")
    parser.add_argument("--image_edit_model", default="qwen-image-edit-plus")
    parser.add_argument("--lexicon_path", default=None)
    parser.add_argument("--fixed_suffix", default="realistic photograph")
    parser.add_argument("--parse_instruction", default=DEFAULT_PARSE_INSTRUCTION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = read_api_key()
    common = dict(
        api_key=api_key,
        vision_language_model=args.vision_language_model,
        image_edit_model=args.image_edit_model,
        lexicon_path=args.lexicon_path,
        fixed_suffix=args.fixed_suffix,
        parse_instruction=args.parse_instruction,
    )
    if args.input:
        process_single_image(args.input, output_path=args.output, skip_existing=not args.overwrite, **common)
        return
    if not args.samples_root:
        parser.error("provide --input or --samples_root")
    for folder in sorted(Path(args.samples_root).iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            image_path = find_folder_image(str(folder))
            if image_path:
                process_single_image(image_path, skip_existing=not args.overwrite, **common)


if __name__ == "__main__":
    main()
