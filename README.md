# SimpleRNN - IMDB Movie Review Sentiment Analysis

A deep learning project using SimpleRNN to classify IMDB movie reviews as positive or negative.

## 📁 Project Files

| File | Description |
|------|-------------|
| `simplernn.py` | Trains the SimpleRNN model on IMDB dataset and saves as `simpleRnn_imdb.h5` |
| `streamlit_app.py` | Interactive web app for real-time sentiment predictions |
| `prediction.py` | Command-line script for testing predictions |
| `embedding.py` | Educational script demonstrating word embeddings |
| `simpleRnn_imdb.h5` | Pre-trained model file |

## 🚀 Quick Start

### Installation
```bash
pip install tensorflow keras streamlit numpy
```

### Usage

**Train model:**
```bash
python simplernn.py
```

**Web interface:**
```bash
streamlit run streamlit_app.py
```

**Command-line prediction:**
```bash
python prediction.py
```

## 📊 Model Architecture

```
Input (500 tokens) → Embedding(128) → SimpleRNN(128) → Dense(1, sigmoid) → Output
```

### Model Specs
- **Vocabulary Size**: 10,000 words
- **Sequence Length**: 500 tokens
- **Embedding Dimension**: 128
- **RNN Units**: 128 (ReLU activation)
- **Output**: Binary classification (sigmoid)
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Early Stopping**: Yes (patience=5)

## 💡 How It Works

1. **Input**: Movie review text
2. **Preprocessing**: Convert to lowercase, tokenize, encode using IMDB word index, pad to 500 tokens
3. **Model**: Embedding layer learns word representations, SimpleRNN processes sequences
4. **Output**: Score (0-1) - > 0.5 = Positive, ≤ 0.5 = Negative

## 📋 Example Usage

```python
from prediction import predict_sentiment

sentiment, score = predict_sentiment("This movie was fantastic!")
print(f"Sentiment: {sentiment}, Score: {score}")
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NumPy

---


