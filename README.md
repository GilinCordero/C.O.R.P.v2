# Equipo-G
Predicción de la Demanda

Problem Statement 

### Research Question

This study aims to determine whether a supervised machine learning regression model can accurately forecast short-term concrete demand (hourly/daily volume) using historical demand data, temporal features regarding logistics for each delivery, and operational variables. The objective is to achieve a forecasting performance with a Mean Absolute Percentage Error (MAPE) below 15%, ensuring practical applicability in real-world industrial settings.


### Domain Context and Motivation

Precision in concrete forecasting is vital because the material’s perishability makes misallocation costly for the supplier. Accurate predictions prevent the waste of overestimation and the project delays of underestimation. By getting the numbers right, the company optimizes logistics and production, ensuring strategic growth.


### Machine Learning Task Definition

The problem is formulated as a supervised regression task, as the objective is to predict a continuous target variable corresponding to concrete demand volume. This formulation is appropriate given the availability of historical labeled data and the quantitative nature of the outputs. Additionally, the temporal structure of the data introduces time-dependent patterns such as seasonality and trends, which align with regression-based forecasting approaches. Consequently, the task can be interpreted as a regression problem with time series characteristics.


### Evaluation Metrics

Model performance will be evaluated using Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²). MAPE is selected as the primary metric due to its interpretability in relative terms, making it suitable for operational decision-making. RMSE is included to penalize large deviations, while R² provides a measure of explained variance. 


### Key References

- Future Business Journal (2025) – Presents a real-world demand forecasting system for a construction supplies distributor using time series and ML models, highlighting practical implementation and integration into business workflows.
- Barker et al. (2021) – Uses machine learning models to predict construction demand based on economic and demographic factors, showing how forecasting can guide strategic decisions in building supply industries.
- Sammour et al. (2023) – Applies machine learning models to forecast demand in the construction sector, demonstrating how predictive techniques can support planning and operational decision-making in environments similar to concrete supply chains.

