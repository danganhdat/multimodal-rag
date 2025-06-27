import pickle
from pymilvus import MilvusClient

class MilvusClientWrapper:
    def __init__(self, uri: str = "http://localhost:19530", collection_name: str = "clip_image_collection", dimension: int = 512):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension
        self._init_collection()

    def _init_collection(self, drop_if_exists: bool = True):
        if drop_if_exists and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            auto_id=True,
            enable_dynamic_field=True,
        )

    def ingest_from_pickle(self, pkl_path: str = "clip_embed.pkl") -> int:
        """Load list of dicts (vector + filepath) and insert into Milvus."""
        with open(pkl_path, 'rb') as f:
            clip_embed_data = pickle.load(f)
        return self.client.insert(collection_name=self.collection_name ,data=clip_embed_data)

    def search(self, query_embedding: list[float], limit: int = 10, output_fields: list[str] = None) -> list[dict]:
        """Perform similarity search and return list of result dicts."""
        output_fields = output_fields or []
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=output_fields
        )
        return results