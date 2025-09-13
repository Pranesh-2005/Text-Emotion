# 📦 txtemo

**txtemo** is a lightweight, production-ready Python package for sentiment/emotion detection using a quantized RoBERTa model exported to ONNX.  
It runs fast on CPU, making it suitable for local apps, servers, and even lightweight devices.

---

## 🚀 Features

- ✅ Quantized RoBERTa ONNX model for speed & efficiency
- ✅ Runs fully on CPU (no GPU required)
- ✅ Easy-to-use Python API
- ✅ Hugging Face Hub integration (auto-downloads model + tokenizer)
- ✅ Returns both labels (Positive/Negative/Neutral) and confidence score

---

## 📥 Installation

```sh
pip install txtemo
```

---

## 📝 Usage

```python
from txtemo import predict

print(predict("I love this AI model!"))  
# ('Positive 😃', 0.92)

print(predict("This is the worst thing ever."))  
# ('Negative 😡', 0.89)

print(predict("Pranesh"))  
# ('Neutral 😐', 0.75)
```

---

## 🖥️ Command Line Interface (CLI)

You can also use txtemo directly from the command line:

```sh
txtemo "This library is amazing!"
# Output: Positive 😃 (0.92)
```

---

## 📊 Labels

- Negative 😡
- Neutral 😐
- Positive 😃

---

## ⚡ Performance

- **Model:** RoBERTa-base (quantized, ONNX)
- **Average inference speed:** ~3x faster than PyTorch version
- **Memory footprint:** Reduced by 50%+

---

## 🌍 Use Cases

- Chatbots 🤖
- Customer feedback analysis 📢
- Social media monitoring 📱
- Product reviews sentiment 🛒

---

## 🔗 Model Source

Hosted on Hugging Face Hub:  
[PraneshJs/Emotion-detection-Text](https://huggingface.co/PraneshJs/Emotion-detection-Text)

---

## 📌 Author

**Pranesh S**  
📧 Contact: [praneshmadhan646@gmail.com]