🧬 VitaScan
Multimodal Vitamin Deficiency Detection using Deep Learning
<div align="center">

✨ AI-powered, Non-Invasive Health Screening System ✨
Combining medical images and patient-reported symptoms for accurate vitamin deficiency detection

</div>
🚀 Project Overview

VitaScan is an advanced multimodal deep learning system designed to detect vitamin deficiencies by jointly analyzing:

📷 Medical images (skin / nails / eyes)

📝 Patient-reported symptoms (natural language text)

The system mimics clinical diagnostic reasoning by fusing visual biomarkers with semantic symptom embeddings, delivering accurate, explainable, and real-time predictions through a web-based interface.

🎯 Supported Vitamin Deficiencies
Vitamin	Common Indicators
🟠 Vitamin A	Night blindness, dry eyes
🔵 Vitamin B12	Fatigue, numbness, memory issues
🟢 Vitamin C	Bleeding gums, poor wound healing
🟡 Vitamin D	Bone pain, muscle weakness
🔴 Vitamin E	Coordination issues, blurred vision
🧠 Key Features

✅ Multimodal learning (Images + Text)
✅ Dual CNN architecture (ResNet50 + EfficientNet-B0)
✅ Transformer-based symptom embeddings (MiniLM)
✅ Leakage-free training and evaluation
✅ Explainable AI with Grad-CAM
✅ Real-time Flask web application
✅ High accuracy under controlled conditions

🏗️ System Architecture
User Input
 ├── Image (Skin / Nail / Eye)
 ├── Symptoms (Text)
        ↓
Image Feature Extraction
 ├── ResNet50 (Global features)
 ├── EfficientNet-B0 (Fine-grained features)
        ↓
Symptom Encoding
 └── Transformer (MiniLM – 384-D embeddings)
        ↓
Multimodal Feature Fusion
        ↓
Neural Classifier
        ↓
Vitamin Deficiency Prediction

🧪 Methodology
🔹 Image Processing

Images resized to 224×224

Normalized and processed using two pretrained CNNs

Feature extraction from Global Average Pooling layers

🔹 Symptom Processing

Free-text symptoms encoded using Sentence Transformers

Captures semantic relationships between symptoms

Generates 384-dimensional embeddings

🔹 Multimodal Fusion

Concatenation of:

ResNet50 features (2048)

EfficientNet-B0 features (1280)

Symptom embeddings (384)

Final fused vector: 3712 dimensions

🔍 Explainability (Grad-CAM)

To improve transparency and trust:

Grad-CAM heatmaps highlight influential image regions

Helps understand why the model made a prediction

Useful for both users and clinicians

📊 Results
Model	Test Accuracy
Image-only fusion	~67%
Multimodal (with leakage)	100% ❌
Multimodal (Leak-free)	≈99% ✅

⚠️ High performance is achieved under controlled experimental conditions.
Real-world clinical performance may vary due to noise in symptoms and imaging quality.

🖥️ Web Application (Flask UI)
Features

📤 Upload medical image

✍️ Enter symptoms in natural language

📈 Get predicted vitamin deficiency

🎯 Confidence score output

Run locally
python app.py


Open browser:

http://127.0.0.1:5000

📁 Project Structure
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
│   ├── resnet50_finetuned_model_final.h5
│   ├── efficientnet_b0_finetuned_best.h5
│   └── multimodal_fusion_classifier_split.h5
├── features/
│   ├── resnet_train.npy
│   ├── resnet_test.npy
│   ├── efficientnet_train.npy
│   ├── efficientnet_test.npy
│   ├── symptoms_train.npy
│   └── symptoms_test.npy
├── results_saved/
│   └── multimodal_final/
├── source/
│   ├── create_train_test_split.py
│   ├── feature_extractor_split.py
│   └── multimodal_fusion_classifier_split.py
├── symptoms/
│   ├── create_symptoms_csv.py
│   ├── symptom_embedding.py
│   └── split_symptom_embeddings.py
└── README.md

⚠️ Repository Note (Important)

Due to GitHub file size limits and best practices, the following are NOT included in this repository:

🚫 Not Pushed to GitHub

Trained model files (.h5, .keras)

Extracted feature files (.npy)

Image & symptom embeddings

Original medical image datasets

These files are generated locally during training and inference.

📦 Files Ignored via .gitignore
models/
features/
Data/
*.h5
*.npy

▶️ How to Reproduce Results

Prepare dataset:

Data/
├── Vitamin_A/
├── Vitamin_B12/
├── Vitamin_C/
├── Vitamin_D/
└── Vitamin_E/


Run pipeline:

python source/create_train_test_split.py
python symptoms/create_symptoms_csv.py
python symptoms/symptom_embedding.py
python symptoms/split_symptom_embeddings.py
python source/feature_extractor_split.py
python source/multimodal_fusion_classifier_split.py


Models and features will be generated locally.

🛠️ Tech Stack

Python 3

TensorFlow / Keras

ResNet50

EfficientNet-B0

Sentence Transformers (MiniLM)

NumPy, Pandas, Scikit-learn

Flask

HTML / CSS

🧩 Learning Outcomes

Multimodal deep learning design

Preventing data leakage

Feature-level fusion strategies

Explainable AI in healthcare

End-to-end ML system deployment

⚠️ Limitations

Uses curated academic datasets

Symptoms are synthetically generated

Not a substitute for medical diagnosis

🔮 Future Enhancements

Real patient symptom data

Clinical validation

Mobile application

Cloud deployment

Vitamin deficiency severity estimation

👨‍💻 Author

Manish Shetty
AI / ML Engineer
📍 India

⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork for experimentation

🧠 Share feedback
