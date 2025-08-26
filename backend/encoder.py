import os
import sys
import json
import torch

from PIL import Image
from transformers import AutoModel, AutoProcessor

# make local CLIP package importable
CODE_PATH = os.path.abspath('../tsbir/code')
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

from clip.model import CLIP
from clip.clip import tokenize, _transform
    
class SigLIPEncoder:

    def __init__(self, model_name="google/siglip2-base-patch16-512", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
            embedding /= embedding.norm(dim=-1, keepdim=True) # normalize
        return embedding.squeeze().tolist()
    
    def encode_text(self, text: str):
        inputs = self.processor(text=text, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
            embedding /= embedding.norm(dim=-1, keepdim=True) # normalize
        return embedding.squeeze().tolist()


class TaskFormerEncoder:

    def __init__(self, model_file = '../tsbir/model/tsbir_model_final.pt', device=None):
        gpu = 0
        model_config_file = '../tsbir/code/training/model_configs/ViT-B-16.json'
        
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        self.model = CLIP(**model_info)
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.to(self.device).eval()
        self.transformer = _transform(self.model.visual.input_resolution, is_train=False)

    def get_feature(self, query_sketch, query_text):
        sketch = Image.open(query_sketch).convert("RGB")
        img1 = self.transformer(sketch).unsqueeze(0).to(self.device)
        txt = tokenize([str(query_text)]).to(self.device)
        with torch.no_grad():
            sketch_feature = self.model.encode_sketch(img1)
            text_feature = self.model.encode_text(txt)
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.model.feature_fuse(sketch_feature, text_feature).squeeze().tolist()


