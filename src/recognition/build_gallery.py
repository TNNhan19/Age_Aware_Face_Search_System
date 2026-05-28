import os
import pickle
from pathlib import Path

from tqdm import tqdm

from src.recognition.face_embedder import FaceEmbedder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_person_id(image_path: Path) -> str:
    """
    Extract person ID from filename.

    Example:
        person_001_01.jpg -> person_001
        person_002_query.jpg -> person_002
    """
    parts = image_path.stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return image_path.stem


def build_gallery(
    gallery_dir="data/raw/gallery",
    output_path="models/gallery_embeddings.pkl"
):
    gallery_dir = Path(gallery_dir)
    output_path = Path(output_path)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery folder not found: {gallery_dir}")

    image_paths = [
        p for p in gallery_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in gallery folder: {gallery_dir}")

    embedder = FaceEmbedder()
    gallery_data = []

    print(f"Found {len(image_paths)} images in gallery.")
    print("Building gallery embeddings...")

    for image_path in tqdm(image_paths):
        try:
            embedding, face_info = embedder.extract_embedding(str(image_path))

            item = {
                "person_id": get_person_id(image_path),
                "image_path": str(image_path),
                "embedding": embedding,
                "face_info": face_info
            }

            gallery_data.append(item)

        except Exception as e:
            print(f"Skipped {image_path}: {e}")

    if len(gallery_data) == 0:
        raise ValueError("No valid face embeddings were created.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(gallery_data, f)

    print(f"Saved {len(gallery_data)} embeddings to {output_path}")


if __name__ == "__main__":
    build_gallery()