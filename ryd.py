import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# импорт файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")
# в качестве индекса принимается дата
psgr.set_index('Month', inplace = True)
psgr.index = pd.to_datetime(psgr.index)

#импорт и построение графика
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decompose = seasonal_decompose(psgr)
decompose.plot()
plt.show()
