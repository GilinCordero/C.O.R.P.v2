# TFT (Temporal Fusion Transformers)

During the last iteration of this project, we focused our predictions in time series data preparation, which involved recording our data sequentially with regular intervals of days, weeks and even months. 

The business now requires a more profound approach, with an hourly prediction of demand throughout all of their plants. 

### What we did previously

SARIMAX, gave us a series of facilities to make a decent prediction of the daily demand as well as the remissions that would most likely appear during each day.
However, it is an old statistical and linear approach to solve this type of situations.
We still chose it due to the seasonality and stationarity strengths, because the data is not to variable in terms of daily changes and differentials, it was more than enough for the results we were looking for.

### Why we now choose TFT

The answer is really simple: *we need to change the goal into predicting within an hourly level of detail*. 

Here is a documented list of all the additional theoretical advantages of using this model instead of SARIMAX:

1. The number of interacting patterns and `exogenous` factors may multiply in a dramatic way, thus we are **no longer needing a linear model**.
2. It enables `multi-horizon` time-series forecasting, meaning that it predicts a variable of interest at multiple future time steps. This gives us an advantage given the requirement: *"We need to visualize predictions 45-60 days in advanced"*.
3. Due to its` transformer arquitecture`, a mechanism of score vectors is used to strengthen correlations betweeen the objetive variable and all the others, meaning that we can analyze with ease which variables positively affect the model results earlier.

## Analysis of our data for model justification

#### First Observations of our database:

* 356332 rows (remissions) in total
* 6 active plants
* Plant 514 active since 2022, the rest since 2020
* 15 default columns

#### Our features

##### - Current Features

| _variable name_ | _meaning/description_ | _early-importance-rating (0-5)_ | _importance-rating-justification_ | _addition-verdict-justification_ | _early-training-set-addition-verdict_ | repeatable? | unique_count |
|---|---|---|---|---|---|---|---|
| **tkt_code** | The id (number) assigned to each remission, meaning every time an appointment is created and completed successfully. | 0 | NO addition to training | It is just a number to identify each order; it doesn’t mean anything to us. | NO addition | Yes (confirmed during EDA) | Not relevant |
|              |                                                                                                                      |   |                         |                                                                             |             |                            |              |



##### - Potential-future features


#### Observations (EDA/Cleaning_phase) for each variable

* [**tkt_code**]{.underline}:
    + 
* 


#### Types supported:

* **Time-varying** _known

* **Time-varying** _unknown_:

* **Time-invariant** _real_

* **Time-invariant** _categorical_


### In my words, what is a TFT?

Transformer-based deep learning model that uses self-attention mechanism to capture complex temporal dynamics of **multiple** time sequences.





