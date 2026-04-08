# LTX 2.3 Three-Stage Image-to-Video Pipeline

> Inspired by: [Triple sampler results](https://www.reddit.com/r/StableDiffusion/comments/1rneluh/ltx_23_triple_sampler_results_are_awesome/) · [Triple stage sampling for LTX2](https://www.reddit.com/r/StableDiffusion/comments/1rn3fjv/for_ltx2_use_triple_stage_sampling/)


## Setup

### 1. Download model files

```bash
# Download all required checkpoints (LTX-2.3 model + Gemma text encoder)
python dl_ltx.py
```

This saves files to:
- `ltx_2.3_ckpt/` — LTX-2.3 model, LoRA, and upscaler files
- `gemma_ckpt/` — Gemma 3 text encoder

### 2. Install dependencies

```bash
uv sync --frozen
source .venv/bin/activate
```

### 3. Prepare input images

Place images (`.png`, `.jpg`, `.jpeg`) in a folder. Each image needs a matching `.txt` file with the same name containing the prompt:

```
my_images/
├── scene_001.png
├── scene_001.txt   ← prompt text
├── scene_002.jpg
└── scene_002.txt
```

## Usage Batch generation default batch_size is 2

```bash
CUDA_VISIBLE_DEVICES=0 python batch_gen.py \
  --input-dir /path/to/my_images \
  --checkpoint-path ltx_2.3_ckpt/ltx-2.3-22b-dev.safetensors \
  --distilled-lora ltx_2.3_ckpt/ltx-2.3-22b-distilled-lora-384.safetensors \
  --spatial-upsampler-path ltx_2.3_ckpt/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root ./gemma_ckpt \
  --num-frames 96 \
  --stage1-steps 30 \
  --stage2-steps 6
```

Output videos are saved to `./scene_image_NB_output_triple/` by default.

### Key parameters

- `--stage1-steps` — Stage 1 denoising steps at half resolution (default: 30)
- `--stage2-steps` — Stage 2 & 3 denoising steps at full resolution (default: 6)
- `--image-strength` — Image conditioning strength 0.0–1.0 (default: 0.8)
- `--num-frames` — Frame count, must satisfy (8×K)+1 (e.g. 97, 121, 161...)
- `--batch-size` — Number of variants to generate per image (default: 2)

### Single image generation (programmatic)

```python
import torch
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_triple_stages import TI2VidTripleStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

pipeline = TI2VidTripleStagesPipeline(
    checkpoint_path="ltx_2.3_ckpt/ltx-2.3-22b-dev.safetensors",
    distilled_lora=[LoraPathStrengthAndSDOps(
        "ltx_2.3_ckpt/ltx-2.3-22b-distilled-lora-384.safetensors",
        0.8, LTXV_LORA_COMFY_RENAMING_MAP)],
    spatial_upsampler_path="ltx_2.3_ckpt/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    gemma_root="./gemma_ckpt",
    loras=[],
)

video_guider_params = MultiModalGuiderParams(
    cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
    modality_scale=1.0, skip_step=0, stg_blocks=[28],
)
audio_guider_params = MultiModalGuiderParams(
    cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
    modality_scale=1.0, skip_step=0, stg_blocks=[28],
)

video, audio = pipeline(
    prompt="your prompt here",
    negative_prompt="",
    seed=42,
    height=1024,
    width=1536,
    num_frames=97,
    frame_rate=24.0,
    video_guider_params=video_guider_params,
    audio_guider_params=audio_guider_params,
    images=[ImageConditioningInput(
        path="image.png", frame_idx=0, strength=0.8, crf=10)],
    tiling_config=TilingConfig.default(),
    streaming_prefetch_count=None,
    max_batch_size=1,
    stage1_steps=30,
    stage2_steps=6,
)

encode_video(
    video=video,
    fps=24.0,
    audio=audio,
    output_path="output.mp4",
    video_chunks_number=get_video_chunks_number(97, TilingConfig.default()),
)
```

---

For all other information about LTX-2, please refer to the [official LTX-2 repository](https://github.com/Lightricks/LTX-2).
