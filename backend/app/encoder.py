import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import open_clip
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class SketchEncoder:
    """TSBIR/TASK-former vision tower only — for sketch → clip_vector search.

    Uses QuickGELU activation to match the TSBIR training config.
    Only loads the visual.* keys from the checkpoint.
    """

    def __init__(self, model_name: str = "ViT-B-16", checkpoint_path: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading SketchEncoder %s (QuickGELU)...", self.model_name)
        prev_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=None, quick_gelu=True,
        )
        logging.disable(prev_level)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            visual_sd = {k: v for k, v in sd.items() if k.startswith("visual.")}
            model_visual = {k: v for k, v in self._model.state_dict().items() if k.startswith("visual.")}
            missing = set(model_visual) - set(visual_sd)
            if missing:
                logger.warning("%d missing visual keys: %s", len(missing), list(missing)[:5])
            self._model.load_state_dict(visual_sd, strict=False)
            logger.info("TSBIR visual checkpoint loaded (%d keys)", len(visual_sd))
        elif self.checkpoint_path:
            logger.warning("Checkpoint not found at %s", self.checkpoint_path)

        # Only keep visual tower, free text tower memory
        self._model.to(self.device).eval()
        if hasattr(self._model, "transformer"):
            del self._model.transformer
        if hasattr(self._model, "token_embedding"):
            del self._model.token_embedding
        torch.cuda.empty_cache()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def encode_image(self, image: PILImage.Image) -> list[float]:
        self._load()
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()


class SigLIP2Encoder:
    """SigLIP2 text + image encoder for text_vector search and image-by-example."""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-512", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading SigLIP2 model %s...", self.model_name)
        from transformers import AutoModel, AutoProcessor
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
        logger.info("SigLIP2 model loaded.")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def encode_text(self, text: str) -> list[float]:
        """Encode a single text query → 768-d vector."""
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            features = self._model.text_model(**text_inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of text queries → list of 768-d vectors."""
        self._load()
        inputs = self._processor(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            features = self._model.text_model(**text_inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    def encode_image(self, image: PILImage.Image) -> list[float]:
        """Encode a single image → 768-d vector (for image-by-example search)."""
        self._load()
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_inputs = {k: v.to(self.device) for k, v in inputs.items() if k.startswith("pixel")}
        with torch.no_grad():
            features = self._model.vision_model(**pixel_inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()
