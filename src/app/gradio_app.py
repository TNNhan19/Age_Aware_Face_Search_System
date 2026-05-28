import gradio as gr
from PIL import Image

from src.recognition.search_engine import FaceSearchEngine


engine = FaceSearchEngine()


def search_face(query_image, top_k):
    if query_image is None:
        return [], "Please upload a query image."

    query_path = "data/raw/query/temp_query.jpg"
    query_image.save(query_path)

    try:
        results, query_info, search_time = engine.search(
            query_image_path=query_path,
            top_k=int(top_k)
        )

        gallery_outputs = []
        result_text = f"Detected faces in query: {query_info['num_faces']}\n"
        result_text += f"Search time: {search_time:.4f} seconds\n\n"
        result_text += f"Top-{int(top_k)} results:\n"

        for rank, result in enumerate(results, start=1):
            img = Image.open(result["image_path"]).convert("RGB")
            caption = (
                f"Rank {rank} | "
                f"ID: {result['person_id']} | "
                f"Score: {result['score']:.4f}"
            )
            gallery_outputs.append((img, caption))

            result_text += (
                f"{rank}. {result['person_id']} | "
                f"score={result['score']:.4f} | "
                f"{result['image_path']}\n"
            )

        return gallery_outputs, result_text

    except Exception as e:
        return [], f"Error: {str(e)}"


demo = gr.Interface(
    fn=search_face,
    inputs=[
        gr.Image(type="pil", label="Upload Query Face Image"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K Results")
    ],
    outputs=[
        gr.Gallery(label="Top-K Similar Faces"),
        gr.Textbox(label="Search Details", lines=10)
    ],
    title="Age-Aware Face Search System",
    description=(
        "Upload a face image and retrieve the top-k most similar faces "
        "from the local gallery using InsightFace/ArcFace embeddings "
        "and cosine similarity."
    )
)


if __name__ == "__main__":
    demo.launch()