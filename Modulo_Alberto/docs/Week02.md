# Team-G
#### Demand Forecasting
--- 
## WEEK 1/2
---
## Problem Statement 
---
### Research Question

Can a modern, deep learning, supervised regression model accurately forecast multi-horizon concrete demand (hourly/daily volume) using historical demand data, temporal known/unknown features and exogenous variables while achieving a forecasting performance with a MAPE (Mean Absolute Percentage Error) below 15%? 


### Domain Context and Motivation

Regression is one of the most valuable machine learning approaches in supervised model design due to its capability to learn based on constant changes to the input to find repercussions in the output and with this, minimize the error accordingly. The purpose of our project is to predict based on a large amount of repetitive data that has been collected for each day and many years in the past. This makes a temporal auto-regressive model as a good take to generate predictions for the company future demand. 


### Machine Learning Task Definition

The problem is formulated as a supervised regression task, as the objective is to predict a continuous target variable corresponding to concrete demand volume. This formulation is appropriate given the availability of historical labeled data and the quantitative nature of the outputs. Additionally, the temporal structure of the data introduces time-dependent patterns such as seasonality and trends, which align with regression-based forecasting approaches. Consequently, the task can be interpreted as a regression problem with time series characteristics.


### Evaluation Metrics

Model performance will be evaluated using: 
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²) 

MAPE is selected as the primary metric due to its interpretability in relative terms, making it suitable for operational decision-making. RMSE is included to penalize large deviations, while R² provides a measure of explained variance. 


### Key References

- Future Business Journal (2025) – Presents a real-world demand forecasting system for a construction supplies distributor using time series and ML models, highlighting practical implementation and integration into business workflows.
- Barker et al. (2021) – Uses machine learning models to predict construction demand based on economic and demographic factors, showing how forecasting can guide strategic decisions in building supply industries.
- Sammour et al. (2023) – Applies machine learning models to forecast demand in the construction sector, demonstrating how predictive techniques can support planning and operational decision-making in environments similar to concrete supply chains.

