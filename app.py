# app.py

# Install dependencies (for local testing)
# pip install gradio fastai torch

from fastai.vision.all import *
import gradio as gr
from pathlib import Path
import PIL
import requests

# Download the model if not present

MODEL_URL = "https://huggingface.co/Obraisan/ccaa_flag_model/resolve/main/ccaa_flag_model.pkl"
MODEL_PATH = Path("ccaa_flag_model.pkl")

if not MODEL_PATH.exists():
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model

model = load_learner(MODEL_PATH)

# Function to process uploaded image and return predictions
def classify_flag(image):
    # Convert Gradio image to PILImage expected by Fastai
    img = PILImage.create(image)
    
    # Get prediction
    pred, pred_idx, probs = model.predict(img)
    
    # Create confidence scores dictionary
    confidence_scores = {model.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    
    # Return top prediction and full confidence scores
    return pred, confidence_scores

# Define Gradio interface
demo = gr.Interface(
    fn=classify_flag,
    inputs=gr.Image(label="Upload Flag Image", type="filepath"),
    outputs=[
        gr.Label(label="Predicted Region"),
        gr.Label(label="Confidence Scores")
    ],
    title="Spanish CCAA Flag Classifier",
    description=(
        "Upload an image of any Spanish regional flag, and the AI will identify "
        "which CCAA it belongs to with confidence scores."
    ),
    examples=[
        "examples/pais_vasco_1.jpg",
        "examples/galicia_1.jpg"
    ],
    theme="soft"
)

# Launch the app (for local testing)
if __name__ == "__main__":
    demo.launch()
