import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np

REPO_ID = "PraneshJs/Emotion-detection-Text"

# Download ONNX model + tokenizer
model_path = hf_hub_download(repo_id=REPO_ID, filename="roberta_sentiment_quant.onnx")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")

# Load ONNX runtime session
ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def predict(text: str):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    outputs = ort_session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )
    logits = outputs[0]
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    pred = np.argmax(probs, axis=1).item()

    labels = ["Negative 😡", "Neutral 😐", "Positive 😃"]
    return labels[pred], float(probs[0][pred])
