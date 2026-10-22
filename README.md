# Customer Churn & Revenue Intelligence Platform

## Overview

The Customer Churn & Revenue Intelligence Platform is a machine learning project designed to estimate customer churn risk and support data-driven retention decisions.

The project combines exploratory data analysis, machine learning, customer risk segmentation, retention recommendations, and an interactive Streamlit application.

## Business Problem

Customer churn can affect recurring revenue and long-term customer relationships. Businesses need a way to identify customers who may leave and prioritize retention outreach.

This project uses customer account and service information to estimate churn probability and classify customers into risk categories.

## Project Features

- Customer churn prediction using a trained Random Forest pipeline
- Data preprocessing within the machine learning pipeline
- Classification threshold selection
- Single-customer prediction through Streamlit
- Bulk customer prediction using CSV uploads
- Low, Medium, and High risk categories
- Rule-based retention recommendations
- Executive dashboard with KPI cards and charts
- Downloadable customer prediction results

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Power BI

## Machine Learning

The project uses a Random Forest classification pipeline with preprocessing steps.

The selected initial classification threshold is 0.45. A customer is flagged as likely to churn when the predicted probability is at least 45%.

The risk categories are separate from the classification threshold:

- Low risk: probability below 40%
- Medium risk: probability from 40% to below 70%
- High risk: probability of at least 70%

## Model Evaluation

Exploratory test-set results at the 0.45 threshold:

| Metric | Result |
|---|---:|
| Precision | 52.96% |
| Recall | 79.68% |
| F1 Score | 63.63% |
| ROC-AUC | 84.07% |

The test set was examined during earlier model comparisons and threshold experiments. These results should therefore be treated as exploratory rather than a fully untouched final performance estimate.

## Run Locally

1. Install a compatible Python version.
2. Clone or download this repository.
3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Place `random_forest_churn_model.pkl` beside `app.py`.
5. Launch the application:

   ```bash
   python -m streamlit run app.py
   ```

6. Open the local URL displayed in the terminal.

## Project Limitations

- Predictions estimate churn probability; they do not guarantee future behavior.
- Retention recommendations are rule-based suggestions and have not been proven to reduce churn.
- Monthly charges associated with high-risk customers are not guaranteed lost revenue.
- Performance on synthetic test records does not establish real-world business effectiveness.

## Future Improvements

- Improve dashboard layout and filtering
- Add explainability for individual predictions
- Evaluate the model on a genuinely untouched test set
- Validate retention strategies using actual campaign outcomes
- Deploy the application for external access

## Author

Harsh Bandekar

Data Science | Machine Learning | Data Analytics
