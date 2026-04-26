from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./models/all-MiniLM-L6-v2",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("✅ Model downloaded successfully to ./models/all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("\nPossible solutions:")
    print("1. Check your internet connection")
    print("2. If behind a proxy, set environment variables HTTP_PROXY and HTTPS_PROXY")
    print("3. Temporarily disable firewall/antivirus")
    print("4. Use a VPN or change DNS to 8.8.8.8")