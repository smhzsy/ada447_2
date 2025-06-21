from fastai.vision.all import *
import gradio as gr
from pathlib import Path

# Load model
learn = load_learner("export.pkl")

def classify_satellite_image(img):
    """
    Classifies a satellite image
    """
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)

    confidence = float(probs[pred_idx])

    # Confidence threshold check (85%)
    if confidence < 0.85:
        return {"🛰️ Unknown or Uncertain": 1.0}

    # Map class names to display names with emojis
    class_names = {
        "AnnualCrop": "🌾 Annual Crop",
        "Forest": "🌲 Forest",
        "HerbaceousVegetation": "🌿 Herbaceous Vegetation",
        "Highway": "🛣️ Highway",
        "Industrial": "🏭 Industrial Area",
        "Pasture": "🐄 Pasture",
        "PermanentCrop": "🌳 Permanent Crop",
        "Residential": "🏘️ Residential Area",
        "River": "🌊 River",
        "SeaLake": "🌊 Sea/Lake"
    }

    # Return results
    results = {}
    for i, prob in enumerate(probs):
        class_name = learn.dls.vocab[i]
        display_name = class_names.get(class_name, class_name)
        results[display_name] = float(prob)

    return results

# Gradio interface
demo = gr.Interface(
    fn=classify_satellite_image,
    inputs=gr.Image(type="pil", label="Upload a Satellite Image 🛰️"),
    outputs=gr.Label(num_top_classes=3, label="Classification Result 🎯"),
    title="🛰️ Satellite Image Land Classifier",
    description="""
    Upload a satellite image and let the model guess the land type!

    **Supported Classes:**
    - 🌾 Annual Crop
    - 🌲 Forest
    - 🌿 Herbaceous Vegetation
    - 🛣️ Highway
    - 🏭 Industrial Area
    - 🐄 Pasture
    - 🌳 Permanent Crop
    - 🏘️ Residential Area
    - 🌊 River
    - 🌊 Sea/Lake

    ⚠️ If the model is less than 85% confident, it will return 'Unknown or Uncertain'.
    """,
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch() 