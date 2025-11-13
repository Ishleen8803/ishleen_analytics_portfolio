# Inflation Forecasting with Money Growth and Survey Expectations

[View full report](inflation_forecasting_money_expectations_var.pdf)

**Tools:** R, FRED macroeconomic data  
**Techniques:** Vector autoregression, F tests, forecast evaluation, replication study, survey expectations

## Overview

This project investigates whether broad money growth still helps to forecast United States inflation once survey based inflation expectations are available. It first replicates a published money and inflation vector autoregression study for a United Kingdom setting on United States data and then extends the model by adding household inflation expectations from the Michigan survey. The work combines economic theory with empirical time series modelling.

## Approach

* Collected quarterly data for consumer prices, broad money and inflation expectations from FRED for the period nineteen seventy nine to twenty nineteen.  
* Constructed year on year percentage changes for inflation and money growth and specified a vector autoregression with four lags for inflation and money.   
* Estimated the baseline system in R and tested whether lags of money growth add forecast power for inflation using a joint F test on the money block.  
* Extended the model by adding one year ahead Michigan survey expectations as a third endogenous variable and re estimated the vector autoregression with inflation, money growth and expectations.   
* Compared fit and forecast performance across specifications using changes in explained variance, residual variance and formal F tests on the money and expectations blocks.

## Results

* In the baseline system with inflation and money, lagged money growth has the expected positive signs but is not statistically significant and a joint F test finds that money lags add no forecast power once inflation persistence is controlled for.   
* Adding inflation expectations increases the inflation equation fit from roughly fifty eight percent to above ninety percent explained variance and reduces forecast errors.   
* Conditional on expectations, the money block becomes statistically redundant, while the expectations block is strongly significant, showing that survey expectations dominate broad money as a short horizon predictor of inflation.   

## Impact

The study shows that in a modern policy regime with forward looking behaviour, money growth is a weak guide to near term inflation once expectations are observed. The project demonstrates the ability to:

* Translate economic theory into testable econometric models  
* Work with real macroeconomic data from public sources  
* Use vector autoregressions and formal hypothesis tests to answer a clear forecasting question  
* Write up results in a structured, policy relevant way
