from huggingface_hub import snapshot_download
import sys

print(">>> Starting download of NousResearch/Meta-Llama-3-8B (~16 GB)...")
local_dir = snapshot_download(
    repo_id="NousResearch/Meta-Llama-3-8B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    resume_download=True,
)
print(">>> Download completed! Stored at:", local_dir)
