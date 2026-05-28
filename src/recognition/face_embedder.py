import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", det_size=(640, 640), ctx_id=-1):
        """
        FaceEmbedder loads InsightFace model for face detection and embedding extraction.

        Args:
            model_name: InsightFace model pack name.
            det_size: Detection input size.
            ctx_id: -1 for CPU, 0 for GPU.
        """
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def _select_largest_face(self, faces):
        """
        Select the largest detected face based on bounding box area.
        This is useful when an image contains multiple faces.
        """
        if len(faces) == 0:
            return None

        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        )

        return largest_face

    def extract_embedding(self, image_path):
        """
        Extract normalized face embedding from an image.

        Args:
            image_path: Path to input image.

        Returns:
            embedding: Normalized 512-d vector.
            face_info: Dictionary with bbox and detection info.
        """
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError(f"No face detected in image: {image_path}")

        face = self._select_largest_face(faces)

        embedding = face.embedding.astype("float32")

        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Invalid embedding with zero norm.")

        embedding = embedding / norm

        face_info = {
            "bbox": face.bbox.astype(int).tolist(),
            "num_faces": len(faces),
            "image_path": image_path
        }

        return embedding, face_info


if __name__ == "__main__":
    embedder = FaceEmbedder()

    image_path = "data/raw/query/test.jpg"
    embedding, face_info = embedder.extract_embedding(image_path)

    print("Embedding shape:", embedding.shape)
    print("Embedding norm:", np.linalg.norm(embedding))
    print("Face info:", face_info)
    print("First 10 values:", embedding[:10])