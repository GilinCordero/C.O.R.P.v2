# TFT (Temporal Fusion Transformers)

During the last iteration of this project, we focused our predictions in time series data preparation, which involved recording our data sequentially with regular intervals of days, weeks and even months. 

The business now requires a more profound approach, with an hourly prediction of demand throughout all of their plants. 

### What we did previously

SARIMAX, gave us a series of facilities to make a decent prediction of the daily demand as well as the remissions that would most likely appear during each day.
However, it is an old statistical and linear approach to solve this type of situations.
We still chose it due to the seasonality and [...] strengths, because the data is not to variable in terms of daily changes and differentials, it was more than enough for the results we were looking for.

### Why we now choose TFT

The answer is really simple: we need to change the goal into predicting within an hourly level of detail. The number of interacting patterns and exogenous factors may multiply in a dramatic way, thus we are no longer needing a linear model. 