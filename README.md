Skills & Tools: Python | TensorFlow | Keras | Transfer Learning | Computer Vision (OpenCV) | Data Augmentation | NumPy | Pandas | Scikit-learn

🧠 Brain Tumor Classification using Deep Learning
📌 Overview
This project focuses on automated brain tumor classification using deep learning and transfer learning.
MRI images are categorized into four tumor types — glioma, meningioma, pituitary, and no tumor.

Three pre-trained CNN architectures — VGG16, VGG19, and InceptionV3 — were trained and compared to determine the best-performing model.


🧩 Data Engineering: Cleaning, augmentation, and stratified dataset splitting

📊 Exploratory Data Analysis: Visualizing dataset balance and class distributions

🧠 Model Training: Transfer learning with TensorFlow-Keras

🧾 Evaluation: Accuracy, precision, recall, F1-score, confusion matrix, and ROC curve analysis

🩺 Dataset

*
Source: [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

Classes:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

Preprocessing Steps:

Resized images: 224x224 (VGG16/VGG19) and 299x299 (InceptionV3)

Data Augmentation:Resize,rotation, flip vartically, 

Split: 90% training / 10% validation (stratified), separate test set for evaluation

Total: 3,264 original images → augmented to 9,792 images

⚙️ Experimental Setup

All experiments were conducted in Google Colab using TensorFlow-Keras.

Parameter	Setting
Optimizer	Adam (lr = 0.001)
Batch Size	16
Epochs	100
Transfer Learning	Pre-trained on ImageNet
Callbacks	Early Stopping, Model Checkpoint
🧩 Models & Results
1️⃣ VGG16

Training Accuracy: 96%

Validation Accuracy: 95%

Test Accuracy: 77% (Best Performing Model)

Class	Precision	Recall	F1-Score
Glioma	0.97	0.23	0.37
Meningioma	0.68	0.97	0.80
No Tumor	0.73	0.99	0.84
Pituitary	0.98	0.85	0.91
Overall Accuracy	0.77		

✅ Consistent and robust performance across most classes.
⚠️ Glioma detection remained challenging

2️⃣ VGG19

Training Accuracy: 97%

Validation Accuracy: 92%

Test Accuracy: 76%

Class	Precision	Recall	F1-Score
Glioma	1.00	0.23	0.38
Meningioma	0.69	0.98	0.81
No Tumor	0.70	0.99	0.82
Pituitary	0.96	0.80	0.87
Overall Accuracy	0.76		
⚠️ Similar glioma misclassification issue observed.

3️⃣ InceptionV3

Training Accuracy: 98%

Validation Accuracy: 91%

Test Accuracy: 73%

Class	Precision	Recall	F1-Score
Glioma	0.88	0.22	0.35
Meningioma	0.74	0.95	0.83
No Tumor	0.66	0.98	0.79
Pituitary	0.84	0.73	0.78
Overall Accuracy	0.73		

⚠️ High training accuracy but lower validation — slight overfitting observed.

Model Comparison Summary
Model	Training Acc	Validation Acc	Test Acc	AUC (Avg)
VGG16	96%	95%	77%	0.88
VGG19	97%	92%	76%	0.88
InceptionV3	98%	91%	73%	0.84

🟩 VGG16 outperformed other models on unseen data with the best generalization capability.

📉 Confusion Matrix & ROC Analysis

Confusion matrices confirm high accuracy for meningioma and no tumor, but lower for glioma.

ROC-AUC results indicate VGG16 = 0.88, showing good discriminative ability.

InceptionV3 achieved strong performance on meningioma (AUC 0.95) and pituitary (AUC 0.92).

🧠 Key Insight

Glioma class remains the hardest to identify — future work will improve this using class rebalancing and advanced architectures.

🔮 Future Improvements

1.	Dataset Expansion: Increasing the dataset size and diversity can enhance model generalization and reduce overfitting.
2.	Hyper parameter Optimization: Implementing fine tuning techniques and automated optimization could further improve model performance.
3.	Explainable AI (XAI): Incorporating interpretability methods such as Grad CAM or LIME can help visualize model decisions, improving clinical trust and transparency.
4.	Model Deployment: Future work may include developing a real time web or mobile application for assisting radiologists in early tumor detection.
5.	Hybrid Models: Combining CNNs with other deep learning techniques may further improve diagnostic accuracy.


🚀 How to Run
# 1️⃣ Clone Repository
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# 3️⃣ Run the Pipeline
python main.py


This will:
✅ Preprocess dataset
✅ Train VGG16, VGG19, InceptionV3
✅ Save model checkpoints, results, and graphs

📂 Project Structure
📂 Brain Tumor Classification
 ┣ 📂 notebooks/          # Jupyter notebooks for experiments
 ┣ 📂 logs/               # TensorBoard logs
 ┣ 📂 results/            # Confusion matrix, ROC curves, reports
 ┣ 📂 src/                # Source code (pipeline, models, evaluation,main)
 ┣ 📜 requirements.txt    # Python dependencies
 ┣ 📜 README.md           # Project documentation

👤 Authors

Safi Ullah
Aizaz Husain
Taimor yousaf
🎓 BSc Computer Science | 
💡 Passionate about AI for Healthcare Innovation
📧 safi60183@email.com
🔗 LinkedIn
 | GitHub
 Note:
This report was written with help of ChatGPT for clearity 