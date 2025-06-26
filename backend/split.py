import pickle
import math

# Load your original large pickle file
with open("aic_2023_clip.pkl", 'rb') as f:
    clip_embed_data = pickle.load(f)

# Define the number of items per chunk (adjust this based on your data size)
chunk_size = 5000  # Adjust as needed

total_chunks = math.ceil(len(clip_embed_data) / chunk_size)

for i in range(total_chunks):
    chunk_data = clip_embed_data[i*chunk_size:(i+1)*chunk_size]
    
    chunk_filename = f"aic_2023_clip_{i+1}.pkl"
    with open(chunk_filename, 'wb') as f:
        pickle.dump(chunk_data, f)
    
    print(f"Saved chunk {i+1}/{total_chunks} as {chunk_filename}")