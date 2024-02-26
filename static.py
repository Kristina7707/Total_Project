import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# импорт файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")
# в качестве индекса принимается дата
psgr.set_index('Month', inplace = True)
psgr.index = pd.to_datetime(psgr.index)

# вычисление и вывод вероятности
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(psgr['#Passengers'])
print('p-value = ' + str(adf_test[1]))
