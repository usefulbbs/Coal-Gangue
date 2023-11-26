from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn import metrics

data = pd.read_csv('data/all_data.csv')

data_x = data.iloc[:, :-1]
data_y = data.iloc[:, -1]

kf = KFold(n_splits=5, shuffle=True, random_state=1)

rf = RandomForestClassifier(n_estimators=2, criterion='gini', random_state=1)
acc_train = []
acc_test = []
for k, (train_index, test_index) in enumerate(kf.split(data_x)):
    kf_train_x, kf_test_x = data_x.values[train_index], data_x.values[test_index]
    kf_train_y, kf_test_y = data_y.values[train_index], data_y.values[test_index]

    # myTrain_X = msc(kf_train_x.mean(axis=0), kf_train_x, 1)
    # myTest_X = msc(kf_train_x.mean(axis=0), kf_test_x, 1)

    rf.fit(kf_train_x, kf_train_y)
    kf_train_yhat = rf.predict(kf_train_x)
    kf_test_yhat = rf.predict(kf_test_x)
    kf_acc_train = metrics.accuracy_score(kf_train_y, kf_train_yhat)
    kf_acc_test = metrics.accuracy_score(kf_test_y, kf_test_yhat)
    print('第', k+1, '折')
    print('train = ' + str(kf_acc_train), 'test = ' + str(kf_acc_test))

    acc_train.append(kf_acc_train)
    acc_test.append(kf_acc_test)
print('-'*35)
print('train-acc = ', np.mean(acc_train))
print('test-acc = ', np.mean(acc_test))
