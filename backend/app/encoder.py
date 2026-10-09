import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import open_clip
from PIL import Image as PILImage


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-16", checkpoint_path: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return
        print(f"Loading CLIP model {self.model_name}...")
        import logging
        logging.disable(logging.WARNING)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=None)
        logging.disable(logging.NOTSET)
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model_keys = set(self._model.state_dict().keys())
            filtered = {k: v for k, v in sd.items() if k in model_keys}
            self._model.load_state_dict(filtered, strict=False)
            print(f"TSBIR checkpoint loaded ({len(filtered)} keys) from {self.checkpoint_path}")
        elif self.checkpoint_path:
            print(f"WARNING: checkpoint not found at {self.checkpoint_path}")
        self._model.to(self.device).eval()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        if self._model is None:
            return
        del self._model, self._tokenizer, self._preprocess
        self._model = self._tokenizer = self._preprocess = None
        torch.cuda.empty_cache()
        print("CLIP model unloaded.")

    def encode_text(self, text: str) -> list[float]:
        self._load()
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()

    def encode_image(self, image: PILImage.Image) -> list[float]:
        self._load()
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()


class SigLIP2Encoder:
    def __init__(self, model_name: str = "google/siglip2-base-patch16-512", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        print(f"Loading SigLIP2 model {self.model_name}...")
        from transformers import AutoModel, AutoProcessor
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        print("SigLIP2 model loaded.")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        if self._model is None:
            return
        del self._model, self._processor
        self._model = self._processor = None
        torch.cuda.empty_cache()
        print("SigLIP2 model unloaded.")

    def encode_text(self, text: str) -> list[float]:
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            features = self._model.text_model(**text_inputs).pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()
