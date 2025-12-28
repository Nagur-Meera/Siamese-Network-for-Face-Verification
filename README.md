# Siamese Network for Face Verification

## 📌 Task Overview

**Title:** Siamese Network for Face Verification  

The objective of this task is to build and train a **Siamese Neural Network** using **Triplet Loss** to verify whether two face images belong to the same person. The model learns discriminative facial embeddings such that images of the same identity are close in embedding space, while images of different identities are far apart.

---

## 🛠 Technologies Used

- **Python 3**
- **TensorFlow 2.x**
- **OpenCV**
- **scikit-learn**
- **Kaggle GPU (Tesla P100)**

---

## 📂 Dataset

- **Labelled Faces in the Wild (LFW – Deep Funneled)**
- Over **13,000 face images** of **5,700+ identities**
- Pre-aligned and cropped faces
- Only identities with **at least two images** are used

---

## 🧠 Approach

### 1. Data Preprocessing
- Images are loaded using **OpenCV**
- Resized to **112 × 112**
- Pixel values normalized to **[0, 1]**
- Dataset organized by person identity

---

### 2. Online Triplet Mining
Triplets are generated dynamically during training:
- **Anchor (A):** A face image
- **Positive (P):** Another image of the same person
- **Negative (N):** An image of a different person

This ensures diverse and continuous triplet generation without storing all triplets offline.

---

### 3. Siamese Network Architecture
- Backbone: **EfficientNetB0 (ImageNet pretrained)**
- Shared weights across Anchor, Positive, and Negative branches
- Output: **128-dimensional L2-normalized embedding**
<pre>
Input Image
    ↓
EfficientNetB0
    ↓
Global Average Pooling
    ↓
Dense Layers
    ↓
128-D L2 Normalized Embedding
</pre>
---

### 4. Triplet Loss Function
The model is trained using **Triplet Loss**:

L = max(0, ||A − P||² − ||A − N||² + margin)

- Margin = **0.5**
- Distance metric: **Euclidean distance**

---

### 5. Model Training
- Optimizer: **Adam**
- Batch size: **32**
- Epochs: **30**
- Training performed on **GPU**
- Validation triplets used to monitor convergence

---

### 6. Evaluation
- Face pairs (same/different identity) are generated
- Embeddings extracted using the trained model
- Euclidean distance used for verification
- Threshold-based classification

**Metrics:**
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- **ROC Curve and AUC**

---

## 📊 Results
- Clear separation between same-person and different-person embeddings
- Strong verification performance on unseen face pairs
- ROC curve demonstrates effective discrimination capability

---

## 📦 Deliverables

✔ Training Notebook (Kaggle)  
✔ Saved Model Files (`.keras`)  
✔ Evaluation Scripts  
✔ ROC Curve & Metrics Plot  

---

## 🏁 Conclusion

This project successfully demonstrates face verification using a Siamese Network trained with triplet loss. By leveraging pretrained EfficientNet features and online triplet mining, the model learns robust facial embeddings capable of distinguishing identities under unconstrained conditions.

---

## 👤 Author
**Shaik Nagur Meeravali**

