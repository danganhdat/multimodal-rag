from sentence_transformers import SentenceTransformer
from PIL import Image


class CLIPEncoder:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        embedding = self.model.encode(image, normalize_embeddings=True)
        return embedding.tolist()
