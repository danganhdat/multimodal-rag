import pickle
import math

# Load your original large pickle file
with open("embeddings/aic_2023_clip.pkl", 'rb') as f:
    clip_embed_data = pickle.load(f)

chunk_size = 10000

total_chunks = math.ceil(len(clip_embed_data) / chunk_size)

for i in range(total_chunks):
    # Correct slicing here!
    chunk_data = clip_embed_data[i*chunk_size:(i+1)*chunk_size]
    
    chunk_filename = f"embeddings/aic_2023_clip_{i}.pkl"
    with open(chunk_filename, 'wb') as f:
        pickle.dump(chunk_data, f)
    
    print(f"Saved chunk {i+1}/{total_chunks} as {chunk_filename}")
