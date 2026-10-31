# Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot

## Live Dashboard
https://hedge-amenity-upscale.ngrok-free.dev/

## Project Overview
This project analyzes Netflix customer data to understand customer behavior, identify churn patterns, evaluate subscription performance, and generate business insights.

## Objectives
- Analyze customer demographics and subscription behavior.
- Identify factors associated with customer churn.
- Build a machine learning model to predict churn risk.
- Segment customers based on behavioral and subscription features.
- Estimate historical customer value.
- Present insights through a Streamlit dashboard and AI Analytics Copilot.

## Technologies Used
- Python
- SQLite and SQL
- Pandas and NumPy
- Matplotlib, Seaborn and Plotly
- Scikit-learn
- Streamlit
- Groq API
- Google Colab

## Project Workflow
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis
3. SQL Business Analysis
4. Feature Engineering
5. Churn Prediction
6. Customer Segmentation
7. Customer Lifetime Value Estimation
8. Model Evaluation and Explainability
9. Streamlit Dashboard and AI Analytics Copilot

## Key Findings
- Total customers: 8,000
- Overall churn rate: 24.89%
- Basic plan churn rate: 26.31%
- Churn rate among customers with failed payments: 28.39%
- Churn rate among customers with negative feedback: 30.05%
- Most common churn reason: Too expensive (365 customers)
- Average total watch time per customer: 411.47 minutes

## Machine Learning
Logistic Regression:
- Accuracy: 72.31%
- Precision: 46.33%
- Recall: 71.36%
- F1-score: 56.18%
- ROC-AUC: 0.7891

## Dashboard Features
1. Overview
2. Churn Analysis
3. Customer Segmentation
4. Customer Lifetime Value
5. Churn Prediction
6. AI Analytics Copilot

## How to Run
Install the required dependencies, make sure all supporting files are available, configure your Groq API key securely, and run:

    streamlit run app.py

## Limitations
- SQLite is used instead of PostgreSQL.
- The churn model requires further validation before production use.
- Customer segmentation has a relatively low silhouette score.
- CLV is a historical estimate, not a predictive model.
- AI-generated SQL is limited to supported questions.

## Conclusion
This project demonstrates an end-to-end customer analytics workflow using SQL, Python, machine learning, and interactive business intelligence.
