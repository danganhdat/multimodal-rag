from pymilvus import MilvusClient, DataType
import pickle
import math

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

COLLECTION_NAME = "my_collection"

collections = client.list_collections()

for name in collections:
    client.drop_collection(name)
    print(f"[Milvus] Collection {name} dropped.")

print("[Milvus] All collections dropped.")

# --- SCHEMA ---
# enable_dynamic_field=True lets you still insert extra scalar fields if present in your data.
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

# Vectors
schema.add_field(
    field_name="text_dense",
    datatype=DataType.FLOAT_VECTOR,
    dim=768,
    description="text dense embedding"
)

schema.add_field(
    field_name="sketch_dense",
    datatype=DataType.FLOAT_VECTOR,
    dim=512,
    description="sketch dense embedding"
)

# Primary key
schema.add_field(
    field_name="path",
    datatype=DataType.VARCHAR,
    is_primary=True,
    max_length=64,
    enable_analyzer=True,
    description="path to image file"
)

# OCR text
schema.add_field(
    field_name="ocr",
    datatype=DataType.VARCHAR,
    max_length=4096,
    enable_analyzer=True,
    description="characters in the image"
)

# Objects
schema.add_field(
    field_name="objects",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=64,
    max_length=32,
    nullable=True,
    description="objects in the image"
)

# Colours: array of VARCHAR (dominant/quantized colour names/labels)
schema.add_field(
    field_name="colours",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=32,
    max_length=16,
    nullable=True,
    description="colour names present in the image"
)

schema.verify()
print(schema)

# --- INDEXES ---
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="text_dense",
    index_name="text_dense_index",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

index_params.add_index(
    field_name="sketch_dense",
    index_name="sketch_dense_index",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

# --- LOAD DATA ---
with open("../aic25_batch1_final.pkl", "rb") as f:
    data = pickle.load(f)

# Expect each row (dict) to contain at least:
# {
#   "path": str,
#   "text_dense": list[float] (len 768),
#   "sketch_dense": list[float] (len 512),
#   "ocr": str,
#   "objects": list[str],   # <-- array of strings
#   "colours": list[str],   # <-- array of strings
# }
#
# If your pickle currently lacks "colours", either add it during preprocessing
# or set to [] for each row before insert.

# (Optional) Ensure fields exist / normalize arrays
for row in data:
    row.setdefault("ocr", "")
    row.setdefault("objects", [])
    row.setdefault("colours", [])

chunk_size = 1024
total_chunks = math.ceil(len(data) / chunk_size)

expected = len(data)
inserted_total = 0

for i in range(total_chunks):
    chunk_data = data[i*chunk_size:(i+1)*chunk_size]

    results = client.insert(
        collection_name=COLLECTION_NAME,
        data=chunk_data
    )

    insert_count = results.get("insert_count", len(chunk_data))
    inserted_total += insert_count
    print(f"Chunk {i+1}/{total_chunks} → inserted {insert_count}")

print(f"Expected total: {expected}")
print(f"Inserted total: {inserted_total}")

client.flush(COLLECTION_NAME)

row_count = client.get_collection_stats(COLLECTION_NAME)["row_count"]
print("Row count in Milvus:", row_count)