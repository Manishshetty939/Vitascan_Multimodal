
---

# 🧬 **VitaScan**

### Multimodal Vitamin Deficiency Detection using Deep Learning



**VitaScan** is an AI-powered, **non-invasive health screening system** that detects vitamin deficiencies by jointly analyzing **medical images** (skin, nails, eyes) and **patient-reported symptoms**.
The system emulates clinical diagnostic reasoning by fusing **visual biomarkers** with **semantic symptom representations**, enabling accurate and explainable predictions.

---

## 🚀 Project Overview

Vitamin deficiencies often manifest through subtle visual cues and subjective symptoms. VitaScan leverages **multimodal deep learning** to combine these complementary sources of information, delivering high-accuracy predictions through a lightweight web interface.

**Input Modalities**

* 📷 Medical images (skin / nail / eye)
* 📝 Free-text symptom descriptions

**Output**

* 🎯 Predicted vitamin deficiency
* 📊 Confidence score
* 🔍 Visual explanation (Grad-CAM)

---

## 🎯 Supported Vitamin Deficiencies

| Vitamin        | Common Indicators                   |
| -------------- | ----------------------------------- |
| 🟠 Vitamin A   | Night blindness, dry eyes           |
| 🔵 Vitamin B12 | Fatigue, numbness, memory issues    |
| 🟢 Vitamin C   | Bleeding gums, poor wound healing   |
| 🟡 Vitamin D   | Bone pain, muscle weakness          |
| 🔴 Vitamin E   | Coordination issues, blurred vision |

---

## 🧠 Key Features

* ✅ Multimodal learning (Images + Text)
* ✅ Dual CNN backbone (ResNet50 + EfficientNet-B0)
* ✅ Transformer-based symptom embeddings (MiniLM)
* ✅ Strict leakage-free training & evaluation
* ✅ Explainable AI with Grad-CAM
* ✅ Real-time Flask web application
* ✅ High accuracy under controlled conditions

---

## 🏗️ System Architecture

```
User Input
 ├── Medical Image (Skin / Nail / Eye)
 ├── Symptoms (Text)
        ↓
Image Feature Extraction
 ├── ResNet50 (Global features)
 ├── EfficientNet-B0 (Fine-grained features)
        ↓
Symptom Encoding
 └── Transformer (MiniLM – 384D)
        ↓
Multimodal Feature Fusion
        ↓
Neural Classifier
        ↓
Vitamin Deficiency Prediction
```

---

## 🧪 Methodology

### 🔹 Image Processing

* Images resized to **224 × 224**
* Normalized and passed through pretrained CNNs
* Feature extraction from **Global Average Pooling** layers

### 🔹 Symptom Processing

* Free-text symptoms encoded using **Sentence Transformers**
* Captures semantic similarity between symptom descriptions
* Generates **384-dimensional embeddings**

### 🔹 Multimodal Fusion

Feature concatenation of:

* ResNet50 → **2048**
* EfficientNet-B0 → **1280**
* Symptom embeddings → **384**

**Total fused feature vector:** **3712 dimensions**

---

## 🔍 Explainability (Grad-CAM)

To enhance transparency:

* Grad-CAM heatmaps highlight critical image regions
* Helps interpret model decisions
* Improves clinical trust and usability

---

## 📊 Results

| Model Configuration       | Test Accuracy |
| ------------------------- | ------------- |
| Image-only fusion         | ~67%          |
| Multimodal (with leakage) | 100% ❌        |
| Multimodal (leak-free)    | ≈99% ✅        |

⚠️ *Results are obtained under controlled experimental settings.
Real-world clinical performance may vary.*

---

## 🖥️ Web Application (Flask)

### Features

* 📤 Upload medical images
* ✍️ Enter symptoms in natural language
* 📈 Get vitamin deficiency prediction
* 🎯 Confidence score display

### Run Locally

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
VITASCAN/
├── app.py
├── Data/
│   ├── Vitamin_A/
│   ├── Vitamin_B12/
│   ├── Vitamin_C/
│   ├── Vitamin_D/
│   └── Vitamin_E/
├── data/
│   ├── split.csv
│   └── symptoms.csv
├── models/
├── features/
├── source/
├── symptoms/
└── README.md
```

---

## ⚠️ Repository Note

Due to GitHub size limits, the following are **not included**:

* Trained model files (`.h5`, `.keras`)
* Extracted feature files (`.npy`)
* Image datasets and embeddings

These are generated **locally** during training.

### Ignored via `.gitignore`

```
models/
features/
Data/
*.h5
*.npy
```

---

## ▶️ Reproducibility

### Dataset Structure

```
Data/
├── Vitamin_A/
├── Vitamin_B12/
├── Vitamin_C/
├── Vitamin_D/
└── Vitamin_E/
```

### Pipeline Execution

```bash
python source/create_train_test_split.py
python symptoms/create_symptoms_csv.py
python symptoms/symptom_embedding.py
python symptoms/split_symptom_embeddings.py
python source/feature_extractor_split.py
python source/multimodal_fusion_classifier_split.py
```

---

## 🛠️ Tech Stack

* Python 3
* TensorFlow / Keras
* ResNet50
* EfficientNet-B0
* Sentence Transformers (MiniLM)
* NumPy, Pandas, Scikit-learn
* Flask
* HTML / CSS

---

## 🧩 Learning Outcomes

* Multimodal deep learning system design
* Preventing data leakage
* Feature-level fusion strategies
* Explainable AI in healthcare
* End-to-end ML deployment

---

## ⚠️ Limitations

* Uses curated academic datasets
* Symptoms are synthetically generated
* Not a substitute for professional medical diagnosis

---

## 🔮 Future Enhancements

* Real patient symptom data
* Clinical validation
* Mobile application
* Cloud deployment
* Severity estimation of deficiencies

---

## 👨‍💻 Author

**Manish Shetty**
AI / ML Engineer
📍 India

---

## ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork for experimentation
* 🧠 Share feedback

---


