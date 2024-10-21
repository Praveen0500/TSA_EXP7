## DEVELOPED BY: PRAVEEN S
## REGISTER NO: 212222240078
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/OnionTimeSeries - Sheet1.csv', parse_dates=['Date'], index_col='Date')

print(data.head())

result = adfuller(data['Min'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = AutoReg(train['Min'], lags=13)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test['Min'], predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['Min'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['Min'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

print("PREDICTION:")
print(predictions)

plt.figure(figsize=(10,6))
plt.plot(test.index, test['Min'], label='Actual Price')
plt.plot(test.index, predictions, color='red', label='Predicted Price')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Minimum Price')
plt.legend()
plt.show()
```
## OUTPUT:

### GIVEN DATA
![image](https://github.com/user-attachments/assets/0d6ad502-acee-4ca3-bee8-39752d95cb82)

### ADF-STATISTIC AND P-VALUE
![image](https://github.com/user-attachments/assets/34e0d6b2-4edd-4449-a8d8-32579924a9fc)


### PACF - ACF
![Untitled](https://github.com/user-attachments/assets/8f1a9e93-d5bf-4afb-bd44-cb3025f9afa5)

### MSE VALUE
![image](https://github.com/user-attachments/assets/f33a7fd5-94ad-44c9-9161-fb70e4130a5f)


### PREDICTION
![image](https://github.com/user-attachments/assets/238165a2-42b6-4b36-a104-e215f3b0adcd)

### FINAL PREDICTION
![Untitled](https://github.com/user-attachments/assets/097027e3-63bb-4cff-aa58-bdc32c574d7d)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
