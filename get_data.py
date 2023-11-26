import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# hebei_coal = pd.read_excel("data/hebei_coal.xlsx", header=0)
# hebei_gangue = pd.read_excel("data/hebei_gangue.xlsx", header=0)

# 每5行取一次平均
# hebei_coal = hebei_coal.groupby(np.arange(len(hebei_coal))//5).agg(['mean'])
# hebei_gangue = hebei_gangue.groupby(np.arange(len(hebei_gangue))//5).agg(['mean'])
# hebei_coal.to_csv("data/hebei_coal.csv")
# hebei_gangue.to_csv("data/hebei_gangue.csv")

# henan = pd.read_excel("data/henan.xlsx", header=0)
# 正则表达式筛选
# henan = henan[henan['sample'].str.contains("20-[0-5]*.sam")].to_csv('data/henan.csv', index=None)
# henan = pd.read_csv("data/henan.csv", header=0)
# henan_gangue = henan[henan['sample'].str.contains("ganshi")]
# henan_gangue = henan_gangue.groupby(np.arange(len(henan_gangue))//5).agg(['mean'])
# henan_gangue.to_csv("data/henan_gangue.csv")
#
# henan_coal = henan[~henan['sample'].str.contains("ganshi")]
# henan_coal = henan_coal.groupby(np.arange(len(henan_coal))//5).agg(['mean'])
# henan_coal.to_csv("data/henan_coal.csv")

# shandong_coal = pd.read_excel("data/shandong_coal.xlsx", header=0)
# shandong_coal = shandong_coal.groupby(np.arange(len(shandong_coal))//5).agg(['mean'])
# shandong_coal.to_csv("data/shandong_coal.csv")
# shandong_gangue = pd.read_excel("data/shandong_gangue.xlsx", header=0)
# shandong_gangue = shandong_gangue.groupby(np.arange(len(shandong_gangue)) // 5).agg(['mean'])
# shandong_gangue.to_csv("data/shandong_gangue.csv")

'''henan_coal = pd.read_csv("data_2/henan_coal.csv").iloc[:, 1:]
henan_gangue = pd.read_csv("data_2/henan_gangue.csv").iloc[:, 1:]
hebei_coal = pd.read_csv("data_2/hebei_coal.csv").iloc[:, 1:]
hebei_gangue = pd.read_csv("data_2/hebei_gangue.csv").iloc[:, 1:]
shandong_coal = pd.read_csv("data_2/shandong_coal.csv").iloc[:, 1:]
shandong_gangue = pd.read_csv("data_2/shandong_gangue.csv").iloc[:, 1:]
all_data = pd.concat([henan_coal, henan_gangue, hebei_coal, hebei_gangue, shandong_coal, shandong_gangue])
all_data.to_csv("data_2/all_data.csv", index=False)'''


# 读取数据，插入标签
'''data = pd.read_csv('data/all_data.csv')
plt.rc("font", family='Microsoft YaHei')
fig, ax = plt.subplots()
x = data.columns[:-1].values.astype(float)
ax.set_xlabel('波长(nm)')
ax.set_ylabel('吸光度')
p = 0
for i in range(0, data.shape[0]):
    ax.plot(x, data.iloc[i, 0:-1].values)
    p = p + 1
plt.show()
print(p)'''
