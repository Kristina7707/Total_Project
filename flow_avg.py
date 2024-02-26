import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# импорт файла статистических данных о потоке пассажиров по годам
psgr = pd.read_csv("p_flow.csv")
# в качестве индекса принимается дата
psgr.set_index('Month', inplace = True)
psgr.index = pd.to_datetime(psgr.index)

# построение графика
plt.figure(figsize = (15,8))
plt.plot(psgr, label = 'Поток пассажиров по годам', color = 'steelblue')
plt.plot(psgr.rolling(window = 12).mean(), label = 'Скользящее среднее за 12 лет', color = 'orange')
plt.legend(title = '', loc = 'upper left', fontsize = 14)
plt.xlabel('Год', fontsize = 14)
plt.ylabel('Поток пассажиров', fontsize = 14)
plt.title('Поток пассажиров с 2010 по 2022 год', fontsize = 16)
plt.show()



plt.figure(figsize = (15,8))
plt.plot(psgr, label = 'Поток пассажиров по годам', color = 'steelblue')
plt.plot(psgr.rolling(window = 12).std(), label = 'Стандартное отклонение', color = 'orange')
plt.legend(title = '', loc = 'upper left', fontsize = 14)
plt.xlabel('Год', fontsize = 14)
plt.ylabel('Поток пассажиров', fontsize = 14)
plt.title('Поток пассажиров с 2010 по 2022 год', fontsize = 16)
plt.show()