# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# импорт файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")
# в качестве индекса принимается дата
psgr.set_index('Month', inplace = True)
psgr.index = pd.to_datetime(psgr.index)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(psgr)
plt.show()  

import statsmodels.api as sm

psgr_log = np.log(psgr + 1)

# Вычисление коэффициентов автокорреляции
result = sm.graphics.tsa.acf(psgr_log, q=1)

print("Коэффициент автокорреляции:")
print(result[])
