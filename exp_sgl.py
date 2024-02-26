
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# Import the statistical data file of passenger flow over the years
psgr = pd.read_csv("p_flow.csv")

# Set the date as the index
psgr.set_index('Month', inplace=True)
psgr.index = pd.to_datetime(psgr.index)

# Define the model using Exponential Smoothing
model = ExponentialSmoothing(psgr['#Passengers'], trend='add', seasonal='add', seasonal_periods=12)

# Fit the model to the data
fit_model = model.fit()

# Add the fitted values to the dataframe
psgr['Holt_Winters'] = fit_model.fittedvalues

# Get the last date and add one month to it
last_date = psgr.iloc[[-1]].index
last_date = last_date + timedelta(days=31)

# Append a new row with the last date to extend the forecast
psgr = psgr.append(pd.DataFrame(index=last_date))

# Plot the data and the forecast
plt.figure(figsize=(15, 8))
plt.plot(psgr['#Passengers'], label='Passenger Flow Data', color='steelblue')
plt.plot(psgr['Holt_Winters'], label='Holt-Winters Method', color='purple')
plt.legend(title='', loc='upper left', fontsize=14)
plt.ylabel('Passenger Flow', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.title('Passenger Flow in 2021. Forecast for January 1, 2022 (Holt-Winters Method)', fontsize=16)
plt.show()


