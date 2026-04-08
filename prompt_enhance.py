#!/usr/bin/env python3
"""Prompt enhancement script using Ollama with multimodal model.

Reads image+txt pairs from input_dir, sends each to Ollama (with image as input)
to enhance the video generation prompt, saves the enhanced version back to the txt.
Original txt is moved to backup_dir.
"""
import os
os.environ['all_proxy']=''
os.environ['all_proxy']=''


import argparse
import base64
import io
import logging
import shutil
import sys
from pathlib import Path

import ollama
from PIL import Image


MAX_IMAGE_SIZE = (512, 512)
SYSTEM_PROMPT = (
    "You are a video generation prompt engineer. "
    "Based on the input image and the current video generation prompt, enhance and expand it into a richer, more detailed prompt that will produce a better video. "
    "Keep the enhanced prompt concise but vivid. "
    "Only output the enhanced prompt — no commentary or formatting."
)


def resize_image(img_path: Path, max_size: tuple[int, int] = MAX_IMAGE_SIZE) -> Image.Image:
    img = Image.open(img_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def pil_to_base64(image: Image.Image) -> str:
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='JPEG')
    byte_stream.seek(0)
    return base64.b64encode(byte_stream.read()).decode('utf-8')


def enhance_prompt(client: ollama.Client, model: str, img_path: Path, prompt: str) -> str:
    img = resize_image(img_path)
    b64_img = pil_to_base64(img)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [b64_img]},
        ],
    )
    return response["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance video generation prompts using Ollama.")
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Directory containing images and .txt prompts.")
    parser.add_argument("--backup-dir", type=str, default="./prompt_back",
                        help="Directory to store original prompts (default: ./prompt_back).")
    parser.add_argument("--model", type=str, default="gemma4:31b",
                        help="Ollama model to use (default: gemma4:31b).")
    parser.add_argument("--extensions", type=str, default="png,jpg,jpeg",
                        help="Comma-separated image extensions (default: png,jpg,jpeg).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be enhanced without writing anything.")
    return parser.parse_args()


def find_image_txt_pairs(input_dir: Path, extensions: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for ext in extensions:
        for img_path in input_dir.rglob(f"*.{ext}"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                pairs.append((img_path, txt_path))
    pairs.sort()
    return pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    log = logging.getLogger(__name__)

    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    backup_dir = Path(args.backup_dir).expanduser().resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    extensions = [e.strip().lstrip(".") for e in args.extensions.split(",")]
    pairs = find_image_txt_pairs(input_dir, extensions)

    if not pairs:
        log.warning("No image+.txt pairs found in %s", input_dir)
        sys.exit(0)

    valid_pairs = [(img, txt) for img, txt in pairs if len(txt.read_text().strip()) >= 5]
    skipped = len(pairs) - len(valid_pairs)
    if skipped:
        log.info("Skipping %d pairs with empty/short prompts", skipped)

    client = ollama.Client(host='http://127.0.0.1:11434')
    log.info("Enhancing %d prompt(s) with %s", len(valid_pairs), args.model)

    for img_path, txt_path in valid_pairs:
        bk_path = str(backup_dir / txt_path.name)
        if os.path.exists(bk_path):
            print("already enhanced", bk_path)
            continue

        original = txt_path.read_text().strip()
        log.info("  [ ] %s", img_path.name)

        try:
            enhanced = enhance_prompt(client, args.model, img_path, original)
            log.info("  [x] %s -> %r", img_path.name, enhanced[:80])
        except Exception as e:
            log.error("  [!] %s failed: %s", img_path.name, e)
            continue

        if args.dry_run:
            continue

        shutil.move(str(txt_path), bk_path)
        txt_path.write_text(enhanced)


if __name__ == "__main__":
    main()
