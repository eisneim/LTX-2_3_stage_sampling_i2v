

python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path ltx_2.3_ckpt/ltx-2.3-22b-distilled.safetensors \
    --distilled-lora ltx_2.3_ckpt/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path ltx_2.3_ckpt/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root gemma_ckpt \
    --prompt "walking into the meeting room; camera follow shot" \
    --image '/home/teli/www/video_gen/LTX-2.3/scene_image_NB/0走进会议室.png' 0 0.8 \
    --num-inference-steps 40 \
    --output-path ./output/output_walk.mp4