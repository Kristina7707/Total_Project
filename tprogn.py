
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Импортирование файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")

# Преобразование колонки с датами в формат даты
psgr['Month'] = pd.to_datetime(psgr['Month'])

# Использование столбца с датами в качестве индекса
psgr.set_index('Month', inplace=True)

# Разделение на обучающую и тестовую выборки
train = psgr[:'2021-12']
test = psgr['2022-01':]

import warnings
warnings.simplefilter(action = 'ignore', category = Warning)
warnings.simplefilter('ignore', ValueWarning)

from statsmodels.tsa.statespace.sarimax import SARIMAX
 # объект модели SARIMAX(1, 0, 0)x(1, 1, 1, 12)
arma = SARIMAX(train, 
                order = (1, 0, 0), 
                seasonal_order = (1, 1, 1, 12))
 
result = arma.fit()
start = len(train)
end = len(train) + len(test) - 1
predictions = result.predict(start, end)

plt.plot(train, color = 'black')
plt.plot(test, color = 'red')
plt.plot(predictions, color = 'green')
plt.title('Обучающая выборка, тестовая выборка и тестовый прогноз')
plt.ylabel('Поток пассажиров')
plt.xlabel('Год')
plt.grid()
plt.show()


from arma import predictions, result
from sklearn.metrics import mean_squared_error

start = len(psgr)
end = (len(psgr) - 1) + 12 * 3

forecast = result.predict(start, end)

plt.figure(figsize = (25,10))
plt.plot(psgr, color = 'black')
plt.plot(forecast, color = 'blue')
plt.title('Фактические данные и прогноз на будущее')
plt.ylabel('Поток пассажиров')
plt.xlabel('Год')
plt.grid()
plt.show()
