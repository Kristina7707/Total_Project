import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# импорт файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")
# в качестве индекса принимается дата
psgr.set_index('Month', inplace = True)
psgr.index = pd.to_datetime(psgr.index)


# обучающая выборка 
train = psgr[:'2021-12']
# тестовая выборка 
test = psgr['2022-01':]
from statsmodels.tools.sm_exceptions import  ValueWarning
  
# отключение предупрждения
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
print(predictions)

#построение графика
plt.plot(train, color = 'black')
plt.plot(test, color = 'red')
plt.plot(predictions, color = 'green')
plt.title('Обучающая выборка, тестовая выборка и тестовый прогноз')
plt.ylabel('Поток пассажиров')
plt.xlabel('Год')
plt.grid()
plt.show()


