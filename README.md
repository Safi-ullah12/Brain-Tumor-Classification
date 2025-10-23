# 🧠 Brain Tumor Classification using Deep Learning

**Skills & Tools:** Python | TensorFlow | Keras | Transfer Learning | Computer Vision (OpenCV) | Data Augmentation | NumPy | Pandas | Scikit-learn

---

## 📌 Overview
This project focuses on **automated brain tumor classification** using **deep learning** and **transfer learning**.

MRI images are categorized into four tumor types — **glioma**, **meningioma**, **pituitary**, and **no tumor**.

Three pre-trained CNN architectures — **VGG16**, **VGG19**, and **InceptionV3** — were trained and compared to determine the best-performing model for medical image diagnosis.

### Workflow Overview
- 🧩 **Data Engineering:** Cleaning, augmentation, and stratified dataset splitting  
- 📊 **Exploratory Data Analysis:** Visualizing dataset balance and class distributions  
- 🧠 **Model Training:** Transfer learning using TensorFlow–Keras  
- 🧾 **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and ROC curve analysis  

---

## 🩺 Dataset

**Source:** [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

**Classes:**
- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

### Preprocessing Steps
- Resized images: `224×224` (VGG16/VGG19) and `299×299` (InceptionV3)  
- Data Augmentation: resize, rotation, vertical flip  
- Split: 90% training / 10% validation (stratified) + separate test set  
- Total: **3,264 original images → augmented to ~9,792 images**

---

## ⚙️ Experimental Setup

All experiments were conducted in **Google Colab** using **TensorFlow–Keras**.

| Parameter | Setting |
|------------|----------|
| Optimizer | Adam (lr = 0.001) |
| Batch Size | 16 |
| Epochs | 100 |
| Transfer Learning | Pre-trained on ImageNet |
| Callbacks | EarlyStopping, ModelCheckpoint |

---

## 🧩 Models & Results

### **1️⃣ VGG16**
- Training Accuracy: **96%**  
- Validation Accuracy: **95%**  
- Test Accuracy: **77%** *(Best Performing Model)*  

| Class | Precision | Recall | F1-Score |
|--------|------------|--------|----------|
| Glioma | 0.97 | 0.23 | 0.37 |
| Meningioma | 0.68 | 0.97 | 0.80 |
| No Tumor | 0.73 | 0.99 | 0.84 |
| Pituitary | 0.98 | 0.85 | 0.91 |

> ✅ Consistent performance across most classes  
> ⚠️ Glioma detection remained challenging

---

### **2️⃣ VGG19**
- Training Accuracy: **97%**  
- Validation Accuracy: **92%**  
- Test Accuracy: **76%**

| Class | Precision | Recall | F1-Score |
|--------|------------|--------|----------|
| Glioma | 1.00 | 0.23 | 0.38 |
| Meningioma | 0.69 | 0.98 | 0.81 |
| No Tumor | 0.70 | 0.99 | 0.82 |
| Pituitary | 0.96 | 0.80 | 0.87 |

> ⚠️ Similar glioma misclassification issue observed.

---

### **3️⃣ InceptionV3**
- Training Accuracy: **98%**  
- Validation Accuracy: **91%**  
- Test Accuracy: **73%**

| Class | Precision | Recall | F1-Score |
|--------|------------|--------|----------|
| Glioma | 0.88 | 0.22 | 0.35 |
| Meningioma | 0.74 | 0.95 | 0.83 |
| No Tumor | 0.66 | 0.98 | 0.79 |
| Pituitary | 0.84 | 0.73 | 0.78 |

> ⚠️ High training accuracy but lower validation — slight overfitting observed.

---

### **Model Comparison Summary**

| Model | Training Acc | Validation Acc | Test Acc | AUC (Avg) |
|--------|---------------|----------------|-----------|------------|
| VGG16 | 96% | 95% | **77%** | 0.88 |
| VGG19 | 97% | 92% | **76%** | 0.88 |
| InceptionV3 | 98% | 91% | **73%** | 0.84 |

> 🟩 **VGG16** outperformed other models with the best generalization capability on unseen data.

---

## 📉 Confusion Matrix & ROC Analysis
- Confusion matrices confirm high accuracy for *meningioma* and *no tumor*, but lower recall for *glioma*.  
- ROC-AUC results indicate:  
  - **VGG16 = 0.88** (strong discriminative ability)  
  - **InceptionV3** performed best on *meningioma* (AUC 0.95) and *pituitary* (AUC 0.92).  

---

## 🧠 Key Insights
- Glioma remains the hardest class to identify due to **visual similarity** and **limited data variability**.  
- Overfitting was observed in deeper architectures (InceptionV3) — mitigated using **dropout, early stopping, and data augmentation**.  
- Future focus: **data rebalancing** and **feature extraction improvement**.

---

## 🔮 Future Improvements
1. **Dataset Expansion:** Increase dataset size and diversity for better generalization.  
2. **Hyperparameter Optimization:** Fine-tuning learning rates and batch sizes.  
3. **Explainable AI (XAI):** Use Grad-CAM or LIME to visualize model decisions for clinical transparency.  
4. **Model Deployment:** Develop a Streamlit or Flask app for real-time MRI predictions.  
5. **Hybrid Models:** Combine CNNs with attention or transformers for better feature representation.  

---

## 🚀 How to Run

```bash
# 1️⃣ Clone Repository
git clone https://github.com/Safi-ullah12/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

# 2️⃣ Install Dependencies
pip install -r requirements.txt
## 🚀 How to Run  

```bash
# 1️⃣ Clone Repository
git clone https://github.com/Safi-ullah12/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# 3️⃣ Run the Pipeline
python main.py
```
## 👤 Authors
**Aizaz Hussain**
 **Tiamoor Yousaf**,
 **Safi Ullah**
  🎓 *BSc Computer Science*
   💡 *Passionate about AI for Healthcare Innovation* 
   📧 **Email:** [safi60183@email.com](mailto:safi60183@email.com) 🔗 **Links:** - [LinkedIn Profile](https://www.linkedin.com/in/safi-ullah-10bbb927a)

