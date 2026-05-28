import cv2
from insightface.app import FaceAnalysis


def main():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 means CPU

    image_path = "data/raw/query/test.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print(f"Cannot read image: {image_path}")
        print("Please put a face image at data/raw/query/test.jpg")
        return

    faces = app.get(img)

    print(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        print("No face detected.")
        return

    face = faces[0]
    embedding = face.embedding

    print("Embedding shape:", embedding.shape)
    print("First 10 embedding values:")
    print(embedding[:10])


if __name__ == "__main__":
    main()