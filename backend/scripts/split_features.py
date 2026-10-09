"""Split combined SigLIP+TSBIR features pickle into separate files.

Usage:
    python -m backend.scripts.split_features [input.pkl] [output_dir]

Defaults:
    input:  siglip-tsbir-features/aic25_batch1_features.pkl
    output: siglip-tsbir-features/

Produces:
    siglip_features.pkl   — text_dense (768-d) renamed to 'vector', plus metadata
    tsbir_features.pkl    — sketch_dense (512-d) renamed to 'vector', plus metadata
"""

import pickle
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "siglip-tsbir-features" / "aic25_batch1_features.pkl"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent

    print(f"Loading {input_path} ...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"  {len(data)} entries")

    siglip_records = []
    tsbir_records = []

    for item in data:
        meta = {
            "path": item["path"],
            "ocr": item.get("ocr", ""),
            "objects": item.get("objects", []),
            "colours": item.get("colours", []),
        }
        siglip_records.append({**meta, "vector": item["text_dense"]})
        tsbir_records.append({**meta, "vector": item["sketch_dense"]})

    siglip_path = output_dir / "siglip_features.pkl"
    tsbir_path = output_dir / "tsbir_features.pkl"

    print(f"Writing {siglip_path} ({len(siglip_records)} entries, 768-d) ...")
    with open(siglip_path, "wb") as f:
        pickle.dump(siglip_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Writing {tsbir_path} ({len(tsbir_records)} entries, 512-d) ...")
    with open(tsbir_path, "wb") as f:
        pickle.dump(tsbir_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()
