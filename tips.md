### 机器学习的一些技巧

#### 参考

\[1] [菊安酱机器学习实战](https://www.bilibili.com/video/BV18J411h7tQ)

[2] [菜菜的Sklearn机器学习](https://www.bilibili.com/video/BV1MA411J7wm)

#### 前言

`pd` 表示`pandas`包

`df`表示`DataFrame`类型的数据集

`plt`表示`matplotlib.pyplot`

`np`代表`numpy`包

以下代码都没有导包操作，读者使用时需要自行进行导包，若读者存在不懂的知识，如**卡方过滤、互信息法等**，请自行百度。

#### 1. 查看数据集的信息 是否由缺失值等情况

``` python
# 查看比较全面的信息
df.info()

# 可以直观的显示
df.isnull().sum()

# 查看某一列数据值的分布
data['Age'].value_counts()

#如果求和为0可以彻底确认是否有NaN
pd.DataFrame(data).isnull().sum()
```

#### 2. 观察用散点图特征之间的关系

```python
x = list(df['balance'].value_counts().index)
y = list(df['balance'].value_counts())
# bankSet['balance'].value_counts().index
# 画图的尺寸
plt.figure(figsize=(10,7))
plt.scatter(x, y)
plt.show()
```

#### 3. 将连续性变量处理为分类特征[1]

``` python
from sklearn.preprocessing import KBinsDiscretizer

#将某个字段分为三分类变量，使用'ordinal'编码方式，和kmeans聚类策略
est1 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
```

![image-20201115114320239](https://starsjsm-images.oss-cn-beijing.aliyuncs.com/img/image-20201115114320239.png)

#### 4. 将分类特征转换为分类数值[1]

```python
# 至少要输入2维以上的数据，一维可以用LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
df.iloc[:,:] = OrdinalEncoder().fit_transform(df)

#1维变量
from sklearn.preprocessing import LabelEncoder
data.loc[:,'Sex'] = LabelEncoder().fit_transform(data.loc[:,'Sex'])
```

#### 5. 手动将分类特征转换为分类数值[1]

``` python
#将二分类变量转换为数值型变量
data['Sex']=(data['Sex']=='male').astype('int')

#将三分类变量转换为数值型变量
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))
```

#### 哑变量转换

``` python
sex_dum = pd.pd.get_dummies(data['Sex'])
data.drop(['Sex'], axis=1, inplace=True)
data = pd.concat([data,sex_dum], axis=1)
data.head()
```

#### 特征选择[2]

##### 方差过滤

``` python
# 消除方差为0的特征
from sklearn.feature_selection import VarianceThreshold
X_var0 = VarianceThreshold().fit_transform(X)
pd.DataFrame(X_var0).shape

# 消除中位数以下的特征
# X.var()#每一列的方差
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
X_fsvar.shape

#若特征是伯努利随机变量，假设p=0.8，即二分类特征中某种分类占到80%以上的时候删除特征
X_bvar = VarianceThreshold(.8 * (1 - .8)).fit_transform(X)
X_bvar.shape
```

##### 卡方过滤

``` python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 计算特征的卡方和P值
chivalue, pvalues_chi = chi2(X_fsvar,y)
 
#k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
 
X_fschi = SelectKBest(chi2, k=填写具体的k).fit_transform(X_fsvar, y)
```

##### F检验

``` python
from sklearn.feature_selection import f_classif
 
F, pvalues_f = f_classif(X_fsvar,y)

k = F.shape[0] - (pvalues_f > 0.05).sum()
 
X_fsF = SelectKBest(f_classif, k=填写具体的k).fit_transform(X_fsvar, y)
```

##### 互信息法

``` python
from sklearn.feature_selection import mutual_info_classif as MIC
result = MIC(X_fsvar,y)
k = result.shape[0] - sum(result <= 0)
X_fsmic = SelectKBest(MIC, k=填写具体的k).fit_transform(X_fsvar, y)
```

##### 嵌入法

``` python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC

RFC = RFC(n_estimators =10,random_state=0)
X_embedded = SelectFromModel(RFC,threshold=0.005).fit_transform(X,y)

# k-折交叉验证
cross_val_score(RFC,X_embedded,y,cv=5).mean()
```

#### 6. 数据去量钢化

``` python
from sklearn.preprocessing import StandardScaler
codedf = StandardScaler().fit_transform(codedf)
codedf = pd.DataFrame(codedf)
```

#### 7. 分割数据集

``` python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(bankSet.iloc[:,:-1]
                                                 ,bankSet.iloc[:,-1]
                                                 ,test_size=0.25
                                                 ,random_state=0
                                                )

#修正测试集和训练集的索引[1]
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
    
# 在pd.Dataframe中选择反索引
x_test = x_train.loc[x_train.index.difference(train_list)]
x_test.shape
```

#### 8. 画数据的散点图，查看是否有规律

``` python
def dataPlot(dataSet):
    m,n=dataSet.shape
    fig = plt.figure(figsize=(8,20),dpi=100)
    colormap = mpl.cm.rainbow(np.linspace(0, 1, n))
    for i in range(n):
        fig_ = fig.add_subplot(n,1,i+1)
        plt.scatter(range(m),dataSet.iloc[:,i].values,s=2,c=colormap[i])
        plt.title(dataSet.columns[i])
        plt.tight_layout(pad=1.2)
```

#### 9. 评分

``` python
# 第一种
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

# 第二种
score = model.score(Xtest, Ytest)

# 交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ada, Xtrain, Ytrain, cv=5)
print(f'这个模型的准确率为{round(scores.mean() * 100,2)}% (+/- {round(scores.std()*2 *100,2)}%)')
```

