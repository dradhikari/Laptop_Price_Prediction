# 💻 Laptop Price Prediction Using Machine Learning

##  Project Overview
This project focuses on predicting laptop prices based on hardware specifications and brand-related features using machine learning regression models.  
The goal is to build, compare, and evaluate multiple models to identify the best-performing approach for accurate price prediction.

---

## Dataset
The dataset is from open source Kaggle contains various laptop attributes such as:
- Brand and model information
- Processor, RAM, storage details
- GPU and operating system
- Display characteristics
- Price (target variable)

> Target Variable: **Laptop Price**

---

##  Workflow
1. Data Cleaning and Preprocessing  
2. Feature Engineering  
3. Model Training  
4. Model Evaluation  
5. Cross-Validation  
6. Performance Comparison  

---

##  Models Implemented
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Regression (SVR)  
- Decision Tree Regressor  
- Random Forest Regressor  
- AdaBoost Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

---

##  Model Performance 

| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Linear Regression | 0.210 | 0.271 | 0.807 |
| Ridge Regression | 0.209 | 0.268 | 0.813 |
| Lasso Regression | 0.211 | 0.272 | 0.807 |
| KNN | 0.193 | 0.275 | 0.803 |
| SVR | 0.202 | 0.271 | 0.808 |
| Decision Tree | 0.186 | 0.254 | 0.832 |
| Random Forest | 0.159 | 0.208 | 0.887 |
| AdaBoost | 0.226 | 0.282 | 0.792 |
| Gradient Boosting | 0.159 | 0.212 | 0.883 |
| XGBoost | **0.152** | **0.206** | **0.889** |

---

##  Cross-Validation Results (CV = 5)

| Model | R² (Mean) | Std Dev |
|------|-----------|---------|
| Linear Regression | 0.8179 | 0.0356 |
| Ridge Regression | 0.8076 | 0.0323 |
| Lasso Regression | 0.8092 | 0.0345 |
| KNN | 0.7641 | 0.0354 |
| SVR | 0.8025 | 0.0428 |
| Random Forest | 0.8718 | 0.0155 |
| AdaBoost | 0.7820 | 0.0151 |
| Gradient Boosting | 0.8785 | 0.0258 |
| XGBoost | **0.8787** | **0.0182** |

---

##  Best Model
**XGBoost Regressor**
- Highest R² score
- Lowest MAE and RMSE
- Strong generalization with low variance across folds

---

##  Key Insights
- Ensemble models significantly outperform linear models
- Tree-based boosting methods provide better robustness
- Cross-validation confirms model stability and reliability

---

##  Tools and Technologies 
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---

## Project Structure

```text
Laptop_Price_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Data/
│   └── laptop_data.csv
├── Notebooks/
│   └── Program.ipynb
├── Report/
    └── Laptop_Price_Report.pdf

```



##  Future Improvements
- Deploy as a Streamlit web application
- Hyperparameter tuning using Bayesian Optimization

---

## Author 
Dilip Adhikari

---

