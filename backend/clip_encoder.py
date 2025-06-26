import torch
import clip
from PIL import Image

class ClipEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    # Define a function to encode images
    def encode_image(self, image_path: str) -> list[float]:
        image = self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(
                dim=-1, keepdim=True
            )  # Normalize the image features
        return image_features.squeeze().tolist()

    # Define a function to encode text
    def encode_text(self, text: str) -> list[float]:
        text_tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(
                dim=-1, keepdim=True
            )  # Normalize the text features
        return text_features.squeeze().tolist()