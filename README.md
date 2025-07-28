#  HealthWise: Heart Disease Predictor

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b)
![Python](https://img.shields.io/badge/Made%20With-Python-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A smart web application that predicts the likelihood of heart disease based on user-input health metrics. It uses a trained **Random Forest Classifier** and provides real-time, interactive predictions through a clean and simple **Streamlit** interface.

---

##  Demo

🔗 Live App (Streamlit Cloud): [Click here to try the app](https://your-streamlit-link.streamlit.app)

---

##  Features

- ✅ User-friendly web interface for heart disease prediction
- ✅ Cleaned and preprocessed medical dataset
- ✅ Real-time predictions using a Random Forest model
- ✅ ROC Curve Visualization for model performance
- ✅ Saves predictions with timestamps
- ✅ Deployment-ready with `requirements.txt`

---

##  Technologies Used

- Python 
- Pandas, NumPy
- scikit-learn
- Streamlit
- Matplotlib / Seaborn
- Joblib

---

## 🗂️ Project Structure
HealthWise-Project/
│
├── app.py                  # Streamlit frontend for user interaction
├── train_model.py          # Script to train and evaluate the Random Forest model
├── preprocessing.py        # Contains data cleaning and preprocessing functions
├── make_predictions.py     # Makes batch predictions on new or existing data
├── model_building.py       # Model evaluation script (e.g., ROC curve, AUC score)
│
├── models/                 # Stores trained models and related assets
│   ├── heart_disease_model.pkl     # Trained Random Forest model
│   └── feature_columns.pkl         # Feature column names used during training
│
├── data/                   # Contains input and output data files
│   ├── cleaned_heart.csv           # Cleaned and preprocessed heart dataset
│   └── predicted_heart.csv         # Prediction results (generated after inference)
│
├── requirements.txt        # List of required Python libraries for setup
└── README.md               # Project documentation and usage guide
