from sklearn import svm  # svm函数需要的
import pandas as pd
import numpy as np  # numpy科学计算库
from sklearn import model_selection
import matplotlib.pyplot as plt  # 画图的库
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing

# 导入数据
df_train = pd.read_excel(r'E:/yrt/classify/fearch_train1.xls', index_col=None)
df_test = pd.read_excel(r'E:/yrt/classify/fearch_test.xls', index_col=None)

# 设置y值
x_train = df_train.drop(["文件名", "类别"], axis=1)
y_train = df_train["类别"]
x_test = df_test.drop(["文件名", "类别"], axis=1)
y_test = df_test["类别"]

# 搭建模型
clf = svm.SVC(kernel='rbf', gamma=0.1,  # 核函数
              decision_function_shape='ovo',  # one vs one 分类问题
              C=10,
              probability=True)

clf.fit(x_train, y_train)  # 训练

# 预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# 准确率
train_acc = accuracy_score(y_train, train_predict)
test_acc = accuracy_score(y_test, test_predict)

print("SVM训练集准确率: {0:.3f}, SVM测试集准确率: {1:.3f}".format(train_acc, test_acc))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)
confusion_matrix = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(7, 5), dpi=80)
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='3d', annot_kws={"fontsize":14})
plt.xlabel('Predicted labels', fontproperties='Times New Roman', fontsize=18)
plt.ylabel('True labels', fontproperties='Times New Roman', fontsize=18)
plt.xticks(fontproperties='Times New Roman', fontsize=14)
plt.yticks(fontproperties='Times New Roman', fontsize=14)
# print(confusion_matrix)
print(confusion_matrix_result)
plt.savefig('E:/yrt/confusion_matrix.jpg')
plt.show()

print(confusion_matrix_result)
print(confusion_matrix)
tn, fp, fn, tp = confusion_matrix_result.ravel()
print(tn, fp, fn, tp)

# 基于svm 实现分类  # 基于网格搜索获取最优模型
from sklearn.model_selection import GridSearchCV
model = svm.SVC(probability=True)
params = [
{'kernel': ['linear'], 'C':[1, 10, 100, 1000]},
{'kernel': ['poly'], 'C':[1, 10], 'degree':[2, 3]},
{'kernel': ['rbf'], 'C':[1, 10, 100, 1000],
 'gamma':[1, 0.1, 0.01, 0.001]}]
model = GridSearchCV(estimator=model, param_grid=params, cv=5)
model.fit(x_train, y_train)

# 网格搜索训练后的副产品
print("模型的最优参数：", model.best_params_)
print("最优模型分数：", model.best_score_)
print("最优模型对象：", model.best_estimator_)

## 预测
# 导入数据
df2 = pd.read_excel(r'E:/yrt/D2/locate_result/0926051056.xls', index_col=None)

# 设置y值
x = df2.drop(["关键时序片段序号", "4秒分割序号"], axis=1)

archxlist = []
archylist = []
for i in range(1, df2.shape[0]):
    # print(index, row)
    res = clf.predict(pd.DataFrame([x.loc[i, :]]))
    if res[0] == 1:  # 输入训练好的svm的输出结果是1（即是拱地行为）
        archpart = int(df2.loc[[i]].values[0][1])  # 获取该片段开始的秒数
        archxlist.append([archpart, archpart + 3])
        print([archpart, archpart + 3], clf.predict_proba(pd.DataFrame([x.loc[i, :]])))
print(archxlist)

def plot_sequence_diagram(lines_list, lines_names, title, x_lable, y_lable, save_path):
    if len(lines_list) == 0 or len(lines_list) != len(lines_names):
        return
    colorlist = ['black', 'blue']
    fig = plt.figure(figsize=(8, 7), dpi=150)
    for n in range(len(lines_list)):
        line_list = lines_list[n]
        x_list = []
        y_list = []
        for m in range(len(line_list)):
            x_list.append(line_list[m])
            y_list.append([n+0.5, n+0.5])
        for i in range(len(x_list)):
            plt.scatter(x_list[i], y_list[i], color=colorlist[n])
            plt.plot(x_list[i], y_list[i], color=colorlist[n])

        plt.grid(True)
        plt.xlim(0, 2820)
        plt.ylim(0, 2)
        new_ticks = np.linspace(0.5, 1.5, 2)
        xgra = ['predict', 'ground truth']
        plt.yticks(new_ticks, xgra)
        plt.title(title)
        plt.xlabel(x_lable)
        # plt.ylabel(y_lable)

    plt.savefig("%s%s.jpg" % (save_path, title))
    plt.close()
'''
## 0926055755
# truthxlist = []

## 0926064454
# truthxlist = [[2479, 2530], [2554, 2559], [2566, 2582], [2588, 2598], [2625, 2642], [2668, 2673], [2705, 2713], [2720, 2736], [2738, 2745], [2748, 2756], [2761, 2769], [2775, 2781]]   # 一秒
# truthxlist = [[2481, 2528], [2565, 2580], [2589, 2596], [2625, 2640], [2669, 2672], [2705, 2712], [2721, 2736], [2737, 2744], [2749, 2756], [2761, 2768], [2777, 2780]]  # 四秒

## 0926073154
# truthxlist = [[11, 19], [878, 885], [895, 906], [909, 913], [918, 926], [944, 947], [979, 1001], [1011, 1046], [1059, 1071], [1089, 1097], [1116, 1129], [1228, 1236], [1590, 1593], [1595, 1607], [1616, 1626], [1636, 1641], [1649, 1658], [1671, 1675],[1705, 1721], [1733, 1738], [2232, 2246], [2272, 2304]]
# truthxlist = [[13, 20], [877, 884], [897, 904], [909, 912], [917, 924], [945, 948], [981, 1000], [1013, 1044], [1061, 1072], [1089, 1096], [1117, 1128], [1229, 1236], [1593, 1608], [1617, 1624], [1637, 1640], [1649, 1656], [1673, 1676], [1705, 1720], [1733, 1736], [2233, 2244], [2273, 2304]]

## 0926081852
# truthxlist = [[916, 923], [930, 941], [947, 950], [960, 981], [1111, 1118], [1120, 1124], [1420, 1423], [1428, 1433], [1445, 1465]]
# truthxlist = [[917, 924], [929, 940], [961, 980], [1113, 1116], [1121, 1124], [1425, 1432], [1445, 1464]]

## 0926090550
# truthxlist = [[1085, 1096], [1266, 1270], [1316, 1320], [1334, 1340], [1472, 1475], [1501, 1506]]
# truthxlist = [[1085, 1096], [1153, 1156], [1253, 1256], [1317, 1320], [1473, 1476], [1501, 1504]]

## 0926095249
# truthxlist = [[169, 173], [250, 256], [316, 324], [1448, 1452], [1458, 1476], [1544, 1548], [1609, 1614], [1646, 1655], [2714, 2717]]
truthxlist = [[169, 172], [249, 256], [317, 324], [1449, 1452], [1457, 1476], [1497, 1500], [1545, 1548], [1609, 1612], [1645, 1652], [2713, 2716]]

## 0926103947
# truthxlist = [[432, 434], [1274, 1278], [1335, 1340]]
# truthxlist = [[1273, 1276], [1337, 1340]]

## 0926112645
# truthxlist = [[800, 804], [980, 983], [2060, 2065], [2136, 2139], [2209, 2218]]
# truthxlist = [[801, 804], [2061, 2064], [2209, 2216]]

## 0926121343
# truthxlist = []
'''
# 连接
sign = 1
while sign == 1:
    l = len(archxlist) - 1
    i = 0
    sign = 0
    while i < l:
        if archxlist[i+1][0] - archxlist[i][1] == 1:
            first = archxlist[i][0]
            last = archxlist[i+1][1]
            archxlist[i] = [first, last]
            del archxlist[i+1]
            sign = 1
            l -= 1
        i += 1

for i in range(len(archxlist)):
    archylist.append([1, 1])

# lines_list = [archxlist, truthxlist]
print('archxlist', archxlist)
'''
print('truthxlist', truthxlist)
plot_sequence_diagram(lines_list, ['predict', 'ground truth'], "0926095249", "time(s)", "y_lable", 'E:/locate_picture/')

# 一秒 评价指标FN、FP、TN、TP
FN = 0
FP = 0
TN = 0
TP = 0
for i in range(1, 2821):
    judge = 0
    truth = 0
    for j in archxlist:
        if j[0] <= i <= j[1]:
            judge = 1
            break
    for k in truthxlist:
        if k[0] <= i <= k[1]:
            truth = 1
            break
    # print(i, judge, truth)
    if judge == 1 and truth == 1:
        TP += 1
    elif judge == 0 and truth == 0:
        TN += 1
    elif judge == 1 and truth == 0:
        FP += 1
    else:
        FN += 1
print('TP, TN, FP, FN:', TP, TN, FP, FN)
print('Accuracy', (TP+TN)/(TP+TN+FP+FN))
print('Precision', TP/(TP+FP))
print('Recall', TP/(TP+FN))

# 四秒
print('TP, TN, FP, FN:', TP/4, TN/4, FP/4, FN/4)
print('Accuracy', (TP+TN)/(TP+TN+FP+FN))
print('Precision', TP/(TP+FP))
print('Recall', TP/(TP+FN))
'''