import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

henan_coal = pd.read_csv("data/henan_gangue.csv")
henan_coal = henan_coal.iloc[:, 1:-1]
henan_coal.index = range(1, len(henan_coal)+1)
henan_coal['index'] = henan_coal.index.values.astype(int)
plt.rcParams.update({'font.size': 35})
outliers = 1
while outliers:
    outliers1 = 0
    coal = henan_coal.values
    coal_mean = henan_coal.mean(axis=0)[:-1]
    ED_list = []

    for i in range(coal.shape[0]):
        ED_list.append(np.linalg.norm(coal_mean - coal[i][:-1]))
    r1 = np.array(ED_list).mean()
    print(3 * r1)
    fig, ax = plt.subplots()

    ax.bar(range(1, len(ED_list) + 1), ED_list, width=0.4)
    ax.plot(3 * r1)
    ax.hlines(3 * r1, 0, henan_coal.shape[0]+1, colors='orange', linestyles='--')
    # ax.set_xlim([0, 121])
    # ax.set_ylim([0, 1.2])
    # plt.grid()

    x = range(0, 60, 10)
    plt.xticks(x)

    plt.show()
    for j in range(coal.shape[0]):
        if ED_list[j] > 3 * r1:
            print(int(coal[j][-1]))
            henan_coal.drop(index=int(coal[j][-1]), inplace=True)
            outliers1 += 1
    print('outliers1 = ', outliers1)
    outliers = outliers1
