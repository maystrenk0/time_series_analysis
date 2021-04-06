import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
import ml_metrics as metrics


import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
data = pd.read_csv('daily-total-female-births.csv', index_col=['Date'], parse_dates=['Date'])
births = data.Births
sns.set(rc={'figure.figsize': (15, 7)})
sns.lineplot(data=births)
plt.title('Female births(daily data)')
plt.show()

data0 = data.resample('W').mean()
births = births.resample('W').mean()
sns.lineplot(data=births)
plt.title('Female births(weekly data)')
plt.show()

print(births.describe())
print('\n')
sns.set(rc={'figure.figsize': (6, 4)})
sns.histplot(births)
plt.show()


def plot_moving_average(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


plot_moving_average(data0, 4)
plot_moving_average(data0, 12)
plot_moving_average(data0, 24)
plot_moving_average(data0, 12, plot_intervals=True, plot_anomalies=True)

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(data, model='additive')
fig = decomposition.plot()
plt.show()


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plot_double_exponential_smoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas

        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """

    plt.figure(figsize=(20, 8))
    for alpha in alphas:
        for beta in betas:
            dataexpsm = double_exponential_smoothing(series, alpha, beta)
            plt.plot(dataexpsm, label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series.values, label="Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)
    plt.show()


plot_double_exponential_smoothing(births, alphas=[0.75], betas=[0.9, 0.1])

row = [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(births)
a = np.vstack([jb_test])
summary = SimpleTable(a, row)
print(summary)
print('\n')

test = sm.tsa.adfuller(births)
print('adf: ', test[0])
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']:
    print('not stationary')
else:
    print('stationary')
print('\n')

births1diff = births.diff(periods=1).dropna()
test = sm.tsa.adfuller(births1diff)
print('adf: ', test[0])
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']:
    print('not stationary')
else:
    print('stationary')
print('\n')

m = births1diff.index[int(len(births1diff.index)/2+1)]
r1 = sm.stats.DescrStatsW(births1diff[m:])
r2 = sm.stats.DescrStatsW(births1diff[:m])
print('p-value: ', sm.stats.CompareMeans(r1, r2).ttest_ind()[1])
print('\n')

sns.set(rc={'figure.figsize': (15, 7)})
sns.lineplot(data=births1diff)
plt.title("First difference")
plt.show()

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(births1diff.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(births1diff, lags=25, ax=ax2)
plt.show()
src_data_model = births[:"1959-09-01"]
model = sm.tsa.ARIMA(src_data_model, order=(2, 1, 1), freq='W').fit(full_output=False, disp=0)
print(model.summary())
print('\n')

q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
print(pd.DataFrame({'Q-stat': q_test[1], 'p-value': q_test[2]}))

print('\n')

pred = model.predict("1959-09-01", "1959-12-31", typ='levels')
trn = births["1959-09-01":]
r2 = r2_score(trn, pred[0:32])
print('R^2: %1.2f' % r2)
print('RMSE: %1.2f' % metrics.rmse(trn, pred[0:32]))
print('MAE: %1.2f' % metrics.mae(trn, pred[0:32]))
births.plot(figsize=(12, 6))
pred.plot(style='r--')
plt.title("ARIMA")
plt.show()
