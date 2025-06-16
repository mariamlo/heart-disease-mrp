# Heart Disease MRP
Early Detection of Heart Disease: A Machine Learning Approach Using CDC Health Indicators

This project aims to develop a machine learning-based risk assessment model for predicting heart disease using the 2022 CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset. The study explores relationships between various personal health indicators and cardiovascular risk, identifies key predictors, and evaluates the performance of several machine learning models (Logistic Regression, Random Forest, SVM, and XGBoost).

Repository Structure
├── data/
│   └── heart_2022_no_nans.csv        # Cleaned dataset (or link if not uploaded for privacy)
├── src/
│   ├── eda.py                        # Exploratory Data Analysis code
│   ├── preprocessing.py              # Data cleaning, encoding, scaling, SMOTE
│   ├── modeling.py                   # Model training, hyperparameter tuning
│   ├── utils.py                      # Helper functions
├── results/
│   └── figures/                       # EDA plots (.png files)
│   └── model_outputs/                 # Evaluation metrics, confusion matrices, etc.
├── MRP_Literature_Review_and_EDA.pdf  # Project report (literature review + EDA)
└── README.md                          # Project documentation

References

Pytlak, K. (2023, October 12). Indicators of heart disease (2022 update). Kaggle. https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data 
