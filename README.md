# 💳 Credit Card Fraud Detection using AutoEncoders

This project applies deep learning techniques—specifically **AutoEncoders**—to detect fraudulent transactions in credit card data. Fraudulent transactions are rare (highly imbalanced), making unsupervised learning a powerful approach to learn "normal" patterns and flag anomalies.

---

## 📌 Project Summary

- **Goal**: Detect fraudulent credit card transactions using unsupervised deep learning
- **Approach**: AutoEncoder neural network trained to reconstruct non-fraudulent (normal) transactions
- **Dataset**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Tech**: Python, Keras, TensorFlow, Scikit-learn, Matplotlib, Pandas

---

## 📁 Files

- `CCFraudDetection_AutoEncoders.ipynb` – main Jupyter Notebook with full analysis
- `requirements.txt` – dependencies for running the notebook

---

## ⚙️ Technologies Used

- 📦 Libraries:
  - `TensorFlow`, `Keras` – model building
  - `Pandas`, `NumPy` – data manipulation
  - `Scikit-learn` – preprocessing, metrics
  - `Matplotlib`, `Seaborn` – visualization

---

## 🧠 Model Architecture

- **Encoder**: Dense layers to compress input features
- **Decoder**: Symmetric dense layers to reconstruct original input
- **Loss**: Reconstruction error (MSE) used to detect anomalies
- **Threshold**: Dynamic or static threshold to flag frauds based on reconstruction error

---

## 📊 Evaluation Metrics

- **Precision / Recall**
- **F1 Score**
- **ROC-AUC**
- **Confusion Matrix**
- Visual inspection of reconstruction error distributions

---

## 📈 Results
- Achieved high recall for fraud detection using a low reconstruction error threshold.
- Demonstrated clear separation of normal and fraudulent transactions based on reconstruction loss.
- Visualized the impact of threshold tuning on model performance.
