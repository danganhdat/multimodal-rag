"""Download required models to the HuggingFace cache.

Run once with internet access:
    python -m backend.scripts.download_models

After that, the app can run offline with HF_HUB_OFFLINE=1.

Models downloaded:
  - google/siglip2-base-patch16-512  (text + image encoder, ~1.5GB)
  - OpenCLIP ViT-B-16 architecture   (for SketchEncoder; weights from TSBIR checkpoint)
"""

import open_clip
from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Downloading models to HuggingFace cache...")
    print("=" * 60)

    print("\n[1/2] OpenCLIP ViT-B-16 architecture (no pretrained weights)...")
    open_clip.create_model_and_transforms("ViT-B-16", pretrained=None, quick_gelu=True)
    print("  Done. (TSBIR checkpoint loaded separately from tsbir_model_weight/)")

    print("\n[2/2] google/siglip2-base-patch16-512...")
    snapshot_download("google/siglip2-base-patch16-512")
    print("  Done.")

    print("\n" + "=" * 60)
    print("All models cached. You can now run offline with:")
    print("  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    print("=" * 60)


if __name__ == "__main__":
    main()
