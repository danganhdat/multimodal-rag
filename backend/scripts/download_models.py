"""Download all required models to the HuggingFace cache.

Run once with internet access:
    python -m backend.scripts.download_models

After that, the app can run offline with HF_HUB_OFFLINE=1.

Models downloaded:
  - google/siglip2-base-patch16-512  (text encoder, ~1.5GB)
  - BAAI/bge-reranker-v2-m3         (cross-encoder reranker, ~5.6GB)
  - openai/clip-vit-base-patch16     (OpenAI CLIP for reference, ~600MB)
"""

import open_clip
from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Downloading models to HuggingFace cache...")
    print("=" * 60)

    print("\n[1/3] OpenCLIP ViT-B-16 (open_clip registry)...")
    open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    print("  Done.")

    print("\n[2/3] google/siglip2-base-patch16-512...")
    snapshot_download("google/siglip2-base-patch16-512")
    print("  Done.")

    print("\n[3/3] BAAI/bge-reranker-v2-m3...")
    snapshot_download("BAAI/bge-reranker-v2-m3")
    print("  Done.")

    print("\n" + "=" * 60)
    print("All models cached. You can now run offline with:")
    print("  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    print("=" * 60)


if __name__ == "__main__":
    main()
