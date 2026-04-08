#!/usr/bin/env python3
"""Batch image-to-video generation script.

Scans input_dir recursively for images (.png, .jpg, .jpeg) that have a matching
.txt file with the same name. Each pair is run through the TI2VidTwoStagesPipeline
to produce a video.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch


def parse_args(params) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image-to-video generation.")
    # I/O
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Directory containing images and .txt prompts.")
    parser.add_argument("-o", "--output-dir", type=str, default="./scene_image_NB_output",
                        help="Directory for output videos.")
    parser.add_argument("--extensions", type=str, default="png,jpg,jpeg",
                        help="Comma-separated list of image extensions (default: png,jpg,jpeg).")
    # Model paths (defaults from t2v.sh)
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
    # Generation params — defaults come from detect_params
    parser.add_argument("--width", type=int, default=1600, #params.stage_2_width,
                        help=f"Output video width, multiple of 64 (default: {params.stage_2_width}).")
    parser.add_argument("--height", type=int, default=896, #params.stage_2_height,
                        help=f"Output video height, multiple of 64 (default: {params.stage_2_height}).")
    parser.add_argument("--num-frames", type=int, default=params.num_frames,
                        help=f"Number of frames; must satisfy (8*K)+1 (default: {params.num_frames}).")
    parser.add_argument("--num-inference-steps", type=int, default=40, #params.num_inference_steps
                        help=f"Number of denoising steps (default: {params.num_inference_steps}).")
    parser.add_argument("--frame-rate", type=float, default=params.frame_rate,
                        help=f"Output video frame rate (default: {params.frame_rate}).")
    parser.add_argument("--seed", type=int, default=params.seed,
                        help=f"Random seed for reproducibility (default: {params.seed}).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of variants per image (default: 4).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all pairs and exit without generating.")
    parser.add_argument("--force-regenerate", action="store_true", default=False,
                        help="Regenerate even if output video already exists.")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None,
                        help="Layer streaming prefetch count.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Max batch size per transformer forward pass (default: 1).")
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

    # Deferred imports for logging setup
    from ltx_pipelines.utils.constants import detect_params

    # Detect params from checkpoint first, before parsing args
    import argparse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--checkpoint-path", type=str,
                            default="ltx_2.3_ckpt/ltx-2.3-22b-distilled.safetensors")
    pre_args, _ = pre_parser.parse_known_args()
    params = detect_params(pre_args.checkpoint_path)
    print("params:", params)

    args = parse_args(params)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = [e.strip().lstrip(".") for e in args.extensions.split(",")]
    pairs = find_image_txt_pairs(input_dir, extensions)

    if not pairs:
        log.warning("No image+.txt pairs found in %s", input_dir)
        sys.exit(0)

    log.info("Found %d image+.txt pair(s) in %s", len(pairs), input_dir)
    for img_path, txt_path in pairs:
        log.info("  [ ] %s -> %s", img_path.relative_to(input_dir), txt_path.relative_to(input_dir))

    if args.dry_run:
        log.info("--dry-run: exiting without generation.")
        sys.exit(0)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    # from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF
    from ltx_pipelines.utils.media_io import encode_video

    distilled_lora = [
        LoraPathStrengthAndSDOps(args.distilled_lora, args.distilled_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP),
    ]

    log.info("Loading pipeline...")
    # pipeline = TI2VidTwoStagesHQPipeline(
    #     checkpoint_path=args.checkpoint_path,
    #     distilled_lora=distilled_lora,
    #     distilled_lora_strength_stage_1=0.25,
    #     distilled_lora_strength_stage_2=0.5,
    #     spatial_upsampler_path=args.spatial_upsampler_path,
    #     gemma_root=args.gemma_root,
    #     loras=[],
    # )

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=[],
    )
    log.info("Pipeline loaded.")

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0, stg_scale=2.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=params.video_guider_params.stg_blocks,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=1.0, stg_scale=2.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=params.audio_guider_params.stg_blocks,
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
                    negative_prompt="screen overlays, artifacts",
                    seed=seed,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    num_inference_steps=args.num_inference_steps,
                    video_guider_params=video_guider_params,
                    audio_guider_params=audio_guider_params,
                    images=[ImageConditioningInput(path=str(img_path), frame_idx=0, strength=1.0, crf=DEFAULT_IMAGE_CRF)],
                    tiling_config=tiling_config,
                    streaming_prefetch_count=args.streaming_prefetch_count,
                    max_batch_size=args.max_batch_size,
                )

                encode_video(
                    video=video,
                    fps=args.frame_rate,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks_number,
                )
                log.info("  [x] %s (variant %d/%d) -> %s", img_path.name, variant_idx + 1, args.batch_size, output_name)

            except Exception as e:
                log.error("  [!] %s (variant %d/%d) failed: %s", img_path.name, variant_idx + 1, args.batch_size, e)
                continue


if __name__ == "__main__":
    main()
