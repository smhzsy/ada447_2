# 🛰️ Satellite Image Classification - FastAI + Gradio + Hugging Face

This repository contains a deep learning project developed with **FastAI** to classify satellite images. A model has been trained using the EuroSAT dataset to classify 10 different land types (e.g., forest, agricultural area, industrial area, residential area).

## 🧠 Project Overview

This project was developed as part of the ADA447 course. The goals were:

- To build a land classification model using the EuroSAT satellite image dataset.
- To apply advanced fine-tuning techniques, including:
  - Learning Rate Finder
  - Discriminative Learning Rates
  - Freezing & Unfreezing
- To set a confidence threshold (85%) to return "Unknown or Uncertain" for ambiguous predictions.
- To deploy the final model as an interactive web application accessible via a browser.

> 📌 This project can be used for practical applications such as urban planning, agriculture, environmental monitoring, and land use analysis.

---

## 🚀 Demo

🔗 **Live App**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/semihozsoy/satellite_classifier)

📸 Upload a satellite image. If the model is more than 85% confident, it will classify it; otherwise, it will return "Unknown or Uncertain".

---
## 🗂 Project Structure

```
proje/
├── satellite_classifier.ipynb    # Full model training pipeline (Google Colab)
├── app.py                       # Gradio interface
├── export.pkl                   # The trained model
├── requirements.txt             # Dependencies for Gradio deployment
└── README.md                    # You're here!
```


---

## 📊 Model Performance

- **Validation Accuracy:** ~98.7%
- **Confidence Threshold:** 85%
- **Number of Classes:** 10 (Land Types)
- **Behavior for Out-of-Domain Inputs:** Returns "Unknown or Uncertain" for inputs that do not resemble satellite images.

---

## ✍️ Author

**Semih Özsoy**
Cybersecurity enthusiast, AI researcher for fun

📬 For questions, feel free to reach out or open an issue. 
