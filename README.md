### DEVELOPED BY: VINOD KUMAR S
### REGISTER NO: 212222240116
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

# PROGRAM:
```python
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load your coffee sales data
coffee_sales_data = pd.read_csv('coffeesales.csv')

# Convert 'date' column to datetime and set it as the index
coffee_sales_data['date'] = pd.to_datetime(coffee_sales_data['date'])
coffee_sales_data.set_index('date', inplace=True)

# Resample the 'money' column by day to get daily total sales
daily_sales = coffee_sales_data['money'].resample('D').sum()

# Simulating an ARMA(1,1) Process
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
plt.show()

# Simulating an ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
plt.show()


```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2024-09-10 190354](https://github.com/user-attachments/assets/f407779b-15b6-44bf-ae48-aed551683129)


## Partial Autocorrelation


![Screenshot 2024-09-10 190412](https://github.com/user-attachments/assets/aa55c811-91e8-4748-b897-85d61ae5c5df)

## Autocorrelation


![Screenshot 2024-09-10 190405](https://github.com/user-attachments/assets/4bca0df0-cb85-42ed-b4a4-aa4e9f1293c3)


## SIMULATED ARMA(2,2) PROCESS:


![Screenshot 2024-09-10 190658](https://github.com/user-attachments/assets/fdf37e95-1a71-43cf-b096-51d77c143026)

## Partial Autocorrelation


![Screenshot 2024-09-10 190728](https://github.com/user-attachments/assets/2d59fa54-bbdf-4f88-b8ed-a7b166f4dddf)


## Autocorrelation
![Screenshot 2024-09-10 190712](https://github.com/user-attachments/assets/ecc6f156-46c0-4466-bced-bd72af4f8897)




# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
