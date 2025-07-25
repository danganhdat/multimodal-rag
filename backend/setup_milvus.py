import os
import pickle
from pymilvus import MilvusClient

class MilvusCollection:
    def __init__(self, client: MilvusClient, collection_name: str, encoder_dim: int = 512):
        self.client = client
        self.collection_name = collection_name
        self.encoder_dim = encoder_dim

    def setup(self, embeddings_dir: str):
        """Ensure collection exists and is populated. Import only if missing or empty."""
        if not self.client.has_collection(self.collection_name):
            print(f"[Milvus] Creating collection '{self.collection_name}' and importing embeddings...")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.encoder_dim,
                auto_id=True
            )
            self.import_embeddings(embeddings_dir)
        else:
            stats = int(self.client.get_collection_stats(self.collection_name)["row_count"])
            print(f"[Milvus] Collection '{self.collection_name}' exists with {stats} entities.")
            if stats == 0:
                print(f"[Milvus] Collection is empty. Importing embeddings...")
                self.import_embeddings(embeddings_dir)
            else:
                print(f"[Milvus] No import needed.")

    def import_embeddings(self, embeddings_dir: str):
        """Import all .pkl embeddings from a directory."""
        files = [f for f in sorted(os.listdir(embeddings_dir)) if f.endswith(".pkl")]
        if not files:
            print(f"[Milvus] No .pkl files found in {embeddings_dir}.")
            return
        for filename in files:
            pkl_path = os.path.join(embeddings_dir, filename)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.client.insert(collection_name=self.collection_name, data=data)
            print(f"[Milvus] Imported: {pkl_path}")

    def entity_count(self) -> int:
        """Return number of entities in the collection."""
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats["row_count"])

    def drop(self):
        """Drop the collection (useful for admin/testing)."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"[Milvus] Collection '{self.collection_name}' dropped.")
