from huggingface_hub import snapshot_download


snapshot_download(repo_id="Lightricks/LTX-2.3", local_dir="./ltx_2.3_ckpt", ignore_patterns=[])
snapshot_download(repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized", local_dir="./gemma_ckpt", ignore_patterns=None)
