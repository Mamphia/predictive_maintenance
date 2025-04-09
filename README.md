# 🚀 CMAPSS RUL Prediction Pipeline  
**End-to-End Predictive Maintenance for Aircraft Engines**  
*LSTM, Random Forest, and XGBoost for Remaining Useful Life (RUL) Estimation*  

![CMAPSS Pipeline Demo](https://via.placeholder.com/800x400?text=CMAPSS+RUL+Prediction+Visualization)  
*(Replace with actual demo GIF/screenshot)*  

---

## **📌 Table of Contents**  
- [Key Features](#-key-features)  
- [Tech Stack](#-tech-stack)  
- [Results](#-results)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Deployment](#-deployment)  
- [Contributing](#-contributing)  
- [License](#-license)  

---

## **✨ Key Features**  
✅ **End-to-End Pipeline** – From raw sensor data to RUL predictions  
✅ **Hybrid Modeling** – LSTM (for temporal patterns) + Random Forest/XGBoost (for tabular features)  
✅ **Feature Engineering** – Rolling stats, lag features, and MinMax scaling  
✅ **Production-Ready** – Model serialization (`joblib`, `.keras`) + FastAPI inference example  
✅ **Visual Analytics** – Error breakdowns, radar plots, and 3D performance surfaces  

---

## **🛠️ Tech Stack**  
- **Python 3.9+**  
- **ML Frameworks**: TensorFlow/Keras, Scikit-learn, XGBoost  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: FastAPI, Docker  

---

## **📊 Results**  
| Model          | MAE  | MSE   | Inference Speed (ms/sample) |  
|----------------|------|-------|-----------------------------|  
| **LSTM**       | 30.4 | 1635  | 15.2                        |  
| **XGBoost**    | 55.3 | 5143  | 2.1                         |  
| **Random Forest** | 56.1 | 5204  | 3.8                         |  

**LSTM outperforms traditional ML by ~45% in MAE!**  

![Prediction Comparison](https://via.placeholder.com/600x300?text=LSTM+vs+XGBoost+vs+RF+Performance)  
