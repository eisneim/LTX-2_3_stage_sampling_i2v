#!/usr/bin/env python3
"""Batch image-to-video generation script using three-stage pipeline.

Scans input_dir recursively for images (.png, .jpg, .jpeg) that have a matching
.txt file with the same name. Each pair is run through the TI2VidTripleStagesPipeline
to produce a video — based on the ComfyUI three-stage workflow.
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image-to-video generation (three-stage).")
    # I/O
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Directory containing images and .txt prompts.")
    parser.add_argument("-o", "--output-dir", type=str, default="./scene_image_NB_output_triple",
                        help="Directory for output videos.")
    parser.add_argument("--extensions", type=str, default="png,jpg,jpeg",
                        help="Comma-separated list of image extensions (default: png,jpg,jpeg).")
    # Model paths
    parser.add_argument("--checkpoint-path", type=str,
                        # default="ltx_2.3_ckpt/ltx-2.3-22b-distilled.safetensors",
                        default="ltx_2.3_ckpt/ltx-2.3-22b-dev.safetensors",
                        help="Path to LTX-2 model checkpoint.")
    parser.add_argument("--distilled-lora", type=str,
                        default="ltx_2.3_ckpt/ltx-2.3-22b-distilled-lora-384.safetensors",
                        help="Path to distilled LoRA.")
    parser.add_argument("--distilled-lora-strength", type=float, default=0.8,
                        help="Strength for the distilled LoRA (default: 0.8).")
    parser.add_argument("--spatial-upsampler-path", type=str,
                        default="ltx_2.3_ckpt/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                        help="Path to spatial upsampler.")
    parser.add_argument("--gemma-root", type=str, default="./gemma_ckpt",
                        help="Path to Gemma text encoder root.")
    # Generation params — three-stage defaults from ComfyUI workflow
    parser.add_argument("--width", type=int, default=1536,
                        help="Final output width, multiple of 64 (default: 1536).")
    parser.add_argument("--height", type=int, default=1024,
                        help="Final output height, multiple of 64 (default: 1024).")
    parser.add_argument("--num-frames", type=int, default=96,
                        help="Number of frames; must satisfy (8*K)+1 (default: 121).")
    parser.add_argument("--frame-rate", type=float, default=24.0,
                        help="Output video frame rate (default: 24.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of variants per image (default: 3).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all pairs and exit without generating.")
    parser.add_argument("--force-regenerate", action="store_true", default=False,
                        help="Regenerate even if output video already exists.")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None,
                        help="Layer streaming prefetch count.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Max batch size per transformer forward pass (default: 1).")
    parser.add_argument("--stage1-steps", type=int, default=30,
                        help="Stage 1 denoising steps (default: 16). ComfyUI uses 8.")
    parser.add_argument("--stage2-steps", type=int, default=6,
                        help="Stage 2 & 3 denoising steps (default: 8). ComfyUI uses 3.")
    parser.add_argument("--image-strength", type=float, default=0.8,
                        help="Image conditioning strength (default: 1.0). ComfyUI uses 0.7.")
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


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    log = logging.getLogger(__name__)

    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = [e.strip().lstrip(".") for e in args.extensions.split(",")]
    pairs = find_image_txt_pairs(input_dir, extensions)

    if not pairs:
        log.warning("No image+.txt pairs found in %s", input_dir)
        sys.exit(0)

    log.info("Found %d pair(s), starting generation...", len(pairs))

    if args.dry_run:
        log.info("--dry-run: exiting without generation.")
        sys.exit(0)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ti2vid_triple_stages import TI2VidTripleStagesPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF
    from ltx_pipelines.utils.media_io import encode_video

    distilled_lora = [
        LoraPathStrengthAndSDOps(args.distilled_lora, args.distilled_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP),
    ]

    log.info("Loading three-stage pipeline...")
    pipeline = TI2VidTripleStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=[],
    )
    log.info("Pipeline loaded.")

    # Guidance params: cfg=1 from ComfyUI workflow, but keep STG for movement consistency
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0,
        stg_scale=2.0,
        rescale_scale=0.0,
        modality_scale=1.0,
        skip_step=0,
        stg_blocks=[28],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0,
        stg_scale=2.0,
        rescale_scale=0.0,
        modality_scale=1.0,
        skip_step=0,
        stg_blocks=[28],
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    for img_path, txt_path in pairs:
        stem = img_path.stem
        prompt = txt_path.read_text().strip()
        if len(prompt) < 5:
            log.warning("  [!] prompt too short, skip: %s", img_path.name)
            continue

        existing = list(output_dir.glob(f"{stem}_*.mp4"))
        n_existing = len(existing)

        if not args.force_regenerate and n_existing >= args.batch_size:
            log.info("  [SKIP] %s (%d/%d variants exist)", img_path.name, n_existing, args.batch_size)
            continue

        for variant_idx in range(n_existing, args.batch_size):
            seed = args.seed + variant_idx
            output_name = f"{stem}_seed{seed}.mp4"
            output_path = output_dir / output_name

            log.info("[ ] %s (variant %d/%d)", img_path.name, variant_idx + 1, args.batch_size)

            try:
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt="",
                    seed=seed,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    video_guider_params=video_guider_params,
                    audio_guider_params=audio_guider_params,
                    images=[ImageConditioningInput(
                        path=str(img_path),
                        frame_idx=0,
                        strength=args.image_strength,
                        crf=10,  # ComfyUI LTXVPreprocess uses img_compression=10
                    )],
                    tiling_config=tiling_config,
                    streaming_prefetch_count=args.streaming_prefetch_count,
                    max_batch_size=args.max_batch_size,
                    stage1_steps=args.stage1_steps,
                    stage2_steps=args.stage2_steps,
                )

                encode_video(
                    video=video,
                    fps=args.frame_rate,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks_number,
                )
                log.info("  [x] %s (variant %d/%d) -> %s",
                         img_path.name, variant_idx + 1, args.batch_size, output_name)

            except Exception:
                raise


if __name__ == "__main__":
    main()
