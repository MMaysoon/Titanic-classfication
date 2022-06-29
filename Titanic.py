#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# can not use merge ,connact not same n.of features
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
combine=[train,test]


# In[3]:


train.columns.values


# In[4]:


train.head()


# In[5]:


train.tail()


# In[6]:


train.isnull().sum()


# In[7]:


test.isnull().sum()


# In[8]:


train.shape


# In[9]:


# describe numerical
train.describe()


# In[10]:


# describe categories  (captial O)
train.describe(include=['O'])


# In[11]:


#as_index=False #sort_values(by=)
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[12]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[13]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[14]:


train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[15]:


g=sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)


# In[16]:


g=sns.FacetGrid(train,col='Survived',row='Pclass',size=2.2,aspect=1.6)
g.map(plt.hist,'Age',bins=20,alpha=.5)


# In[17]:


g=sns.FacetGrid(train,row='Embarked',size=2.2,aspect=1.6)
g.map(sns.pointplot,'Pclass', 'Survived', 'Sex',palette='deep')
g.add_legend()


# In[18]:


g=sns.FacetGrid(train,row='Embarked',col='Survived',size=2.2,aspect=1.6)
g.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)
g.add_legend()


# In[19]:


#drop ticket,cabin
train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)


# In[20]:


train.head()


# In[21]:


combine=[train,test]


# In[22]:


# combine list 0--train,1--test
combine[0].head()


# In[23]:


for dataset in combine:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)


# In[24]:


train['Title']


# In[25]:


pd.crosstab(train['Title'],train['Sex'])


# In[26]:


# replace some title with Rare
for x in combine:
    x['Title']=x['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    x['Title'] = x['Title'].replace('Mlle', 'Miss')
    x['Title'] = x['Title'].replace('Ms', 'Miss')
    x['Title'] = x['Title'].replace('Mme', 'Mrs')


# In[27]:


train[['Title','Survived']].groupby(['Title'],as_index=False).mean()


# In[28]:


# map label for title
mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for x in combine:
    x['Title']=x['Title'].map(mapping)
    x['Title']=x['Title'].fillna(0)


# In[29]:


train=train.drop(['Name', 'PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
combine=[train,test]


# In[30]:


# convert sex to numerical
for x in combine:
    x['Sex']=x['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[31]:


train.head()


# In[32]:


g = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', alpha=.5, bins=20)
g.add_legend()


# In[33]:


# not know age --guess by pclass and gender
# famele in plcess 1 null-- replace by fample gender median 
# 2 gender , 3 plcees-------6


# In[34]:


ages=np.zeros((2,3))


# In[35]:


for x in combine :
    for i in range(0,2):
        for j in range (0,3):
            agex=x[(x['Sex']==i) & (x['Pclass']==j+1)]['Age'].dropna()
            y=agex.median()
            ages[i,j]=int(y/0.5+0.5)*0.5
            
    for i in range (0,2):
        for j in range (0,3):
            x.loc[(x.Age.isnull())&(x.Sex==i)&(x.Pclass==j+1),'Age']=ages[i,j]
            
    x['Age']=x['Age'].astype(int)
        
            
           
    


# In[36]:


train.isnull().sum()


# In[37]:


train['AgeBand']=pd.cut(train['Age'],5)


# In[38]:


train['AgeBand'].value_counts()


# In[39]:


train[['AgeBand','Survived']].groupby(['AgeBand']).mean().sort_values(by='AgeBand',ascending=True)


# In[40]:



for x in combine :
    x.loc[x['Age']<=16,'Age']=0
    x.loc[(x['Age'] > 16) & (x['Age'] <= 32), 'Age'] = 1
    x.loc[(x['Age'] > 32) & (x['Age'] <= 48), 'Age'] = 2
    x.loc[(x['Age'] > 48) & (x['Age'] <= 64), 'Age'] = 3
    x.loc[ x['Age'] > 64, 'Age']=4


# In[41]:


train['Age'].value_counts()


# In[42]:


train.tail()


# In[43]:


train=train.drop(['AgeBand'],axis=1)


# In[44]:


combine=[train,test]


# In[45]:


#familysize 1--same person
for x in combine :
    x['family']=x['SibSp'] + x['Parch'] + 1


# In[46]:


train[['family', 'Survived']].groupby(['family'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[47]:


# is alone
for x in combine :
    x['Alone']=0
    x.loc[x['family']==1,'Alone']=1
    
train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()


# In[48]:


train = train.drop(['Parch', 'SibSp', 'family'], axis=1)
test = test.drop(['Parch', 'SibSp', 'family'], axis=1)
combine = [train, test]


# In[49]:


# age and pclass
for x in combine :
    x['age*class']=x.Age*x.Pclass
    


# In[50]:


train.loc[:,['age*class','Age','Pclass']].head(10)


# In[51]:


z=train.Embarked.dropna().mode()[0]


# In[52]:


for x in combine:
    x['Embarked'] = x['Embarked'].fillna(z)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[53]:


for x in combine:
    x['Embarked'] = x['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()


# In[54]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[55]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
train['FareBand']


# In[56]:


train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[57]:


for x in combine:
    x.loc[ x['Fare'] <= 7.91, 'Fare'] = 0
    x.loc[(x['Fare'] > 7.91) & (x['Fare'] <= 14.454), 'Fare'] = 1
    x.loc[(x['Fare'] > 14.454) & (x['Fare'] <= 31), 'Fare']   = 2
    x.loc[ x['Fare'] > 31, 'Fare'] = 3
    x['Fare'] = x['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)


# In[79]:


x_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
x_test  = test.drop("PassengerId", axis=1).copy()


# In[59]:


#logistic regrassion
from sklearn.linear_model import LogisticRegression


# In[60]:


lr=LogisticRegression()


# In[61]:


lr.fit(x_train,y_train)


# In[104]:


c=lr.score(x_train,y_train)
c


# In[65]:


# correlaion
coef=pd.DataFrame(train.columns.delete(0))


# In[66]:


coef


# In[69]:


coef.columns=['feature']
coef


# In[72]:


coef['correlation']=pd.Series(lr.coef_[0])


# In[74]:


coef.sort_values(by='correlation',ascending=False)


# In[95]:


# vector machine
from sklearn.svm import SVC,LinearSVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
a=svc.score(x_train,y_train)
a


# In[96]:


#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_test)
b=knn.score(x_train, y_train)
b


# In[97]:


#naviebayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
e=gaussian.score(x_train, y_train)
e


# In[98]:


#perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_test)
f=perceptron.score(x_train, y_train)
f


# In[99]:


# linear svc
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
h=linear_svc.score(x_train, y_train) 
h


# In[100]:


# gd
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)
g=sgd.score(x_train, y_train) 
g


# In[101]:


#dt
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test)
i=decision_tree.score(x_train, y_train) 
i


# In[102]:


#rf
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
d=random_forest.score(x_train, y_train)
d


# In[105]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [a, b, c, 
              d, e, f, 
              g, h, i]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




