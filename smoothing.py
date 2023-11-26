from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import os

# 读取数据，插入标签
data = pd.read_csv('data/all_data.csv')
coal_and_gangue = pd.DataFrame(signal.savgol_filter(data.iloc[:, :-1], 9, 2))
coal_and_gangue['label'] = data['label']
print(coal_and_gangue)

fig, ax = plt.subplots()
x = data.columns[:-1].values.astype(float)
ax.set_xlabel('Wavelength(nm)')
ax.set_ylabel('Absorbance')
p = 0
for i in range(0, coal_and_gangue.shape[0], 20):
    ax.plot(x, coal_and_gangue.iloc[i, 0:-1].values)
    p = p + 1
plt.show()
print(p)
coal_and_gangue.columns = data.columns
coal_and_gangue.to_csv('data/all_data_smoothed.csv', index=False)



