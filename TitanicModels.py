# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:18:35 2017

@author: satya naidu
"""

        

# importing libraries
import pandas as pd
import numpy as np

#visualizing libraries
import matplotlib.pyplot as plt
import seaborn as sns



#machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline

import requests,json
from sklearn.externals import joblib
# importing data
dataset=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#data exploration
#dataset.columns.values
dataset.describe()
dataset.info()
dataset.head()
dataset.isnull().sum()



dataset['Survived'].value_counts().plot.pie()
sns.countplot(dataset['Survived'])


## categorical feature
dataset.groupby(['Sex','Survived'])['Survived'].count()
dataset[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sns.countplot('Sex',hue='Survived',data=dataset)

dataset[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar()
sns.factorplot('Embarked','Survived',data=dataset)
sns.countplot('Embarked',data=dataset)
sns.countplot('Embarked',hue='Sex',data=dataset)
sns.countplot('Embarked',hue='Survived',data=dataset)
sns.countplot('Embarked',hue='Pclass',data=dataset)
## ordinal feature
pd.crosstab(dataset.Pclass,dataset.Survived,margins=True)
dataset['Pclass'].value_counts().plot.bar()
sns.countplot('Pclass',hue='Survived',data=dataset)
pd.crosstab([dataset.Sex,dataset.Survived],dataset.Pclass,margins=True)

sns.factorplot('Pclass','Survived',hue='Sex',data=dataset)

## continous feature

dataset['Age'].describe()
sns.violinplot('Sex','Age',hue='Survived',data=dataset)
sns.boxplot('Survived','Age',data=dataset)
dataset[dataset['Survived']==0].Age.plot.hist()
#g=sns.FacetGrid(dataset,col='Survived')
#g.map(plt.hist(dataset['Age'],bins=20))

pd.crosstab(dataset.Parch,dataset.Pclass)

## Fare feature
dataset['Fare'].describe()
sns.distplot(dataset[dataset['Pclass']==1].Fare)

sns.heatmap(dataset.corr(),annot=True)

#dataset.loc[dataset['Age']<=16,'Age_Band']=0
#dataset.loc[dataset['Age']>16,'Age_Band']=1
dataset=dataset.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)
test=test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

X=np.array(X)
sns.countplot(dataset['Survived'])



from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy="mean",axis=0)
#imputer1=Imputer(missing_values='NaN',strategy="most_frequent",axis=0)
X[:,2]=np.ravel(imputer.fit_transform(X[:,2].reshape(-1,1)))
#X[:,6]=imputer1.fit_transform(X[:,6])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_1=LabelEncoder()
X[:,1]=labelencoder_1.fit_transform(X[:,1])
#labelencoder_2=LabelEncoder()
#X[:,6]=labelencoder_2.fit_transform(X[:,6])
hotencoder=OneHotEncoder(categorical_features=[1])
X=hotencoder.fit_transform(X).toarray()
X=X[:,1:]

sns.heatmap(dataset.corr(),annot=True)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)




#logistic regression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)
classifier.score(X_train,Y_train)
#Linear SVC
svc=SVC(kernel="linear",C=0.1,gamma=0.1)
svc.fit(X_train,Y_train)
y_pred=svc.predict(X_test)
svc.score(X_train,Y_train)

#Kernal SVC
ksvc=SVC(kernel="rbf",C=0.1,gamma=0.1)
ksvc.fit(X_train,Y_train)
y_pred=ksvc.predict(X_test)
ksvc.score(X_train,Y_train)

#Decisio Tree
dc=DecisionTreeClassifier()
dc.fit(X_train,Y_train)
y_pred=dc.predict(X_test)
dc.score(X_train,Y_train)

#random forest
random=RandomForestClassifier(n_estimators=100)
random.fit(X_train,Y_train)
y_pred= random.predict(X_test)
random.score(X_train,Y_train)

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
knn.score(X_train,Y_train)

#nave bayes

naive= GaussianNB()
naive.fit(X_train,Y_train)
y_pred=naive.predict(X_test)
naive.score(X_train,Y_train)

# applying K-fold to evalaute model

from sklearn.model_selection import cross_val_score, cross_val_predict,KFold
xyz=[]
accuracy=[]
std=[]
lists=['Logistic Regression','Linear Svm','Radial Svm','Decision Tree','Random Forest','KNN','Naive Bayes']
estimators=[classifier,svc,ksvc,dc,random,knn,naive]
for i in estimators:
    accuracies= cross_val_score(i,X=X,y=y,cv=10,scoring="accuracy")
    xyz.append(accuracies.mean())
    std.append(accuracies.std())
    accuracy.append(accuracies)
new = pd.DataFrame({'CV Mean':xyz,'Std':std},index=lists)


## applying grid search to imporve model

from sklearn.model_selection import GridSearchCV
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,y)
print(gd.best_score_)
print(gd.best_estimator_)



joblib.dump(classifier,"logistic_regression_model.pkl")

## Ensembling 
#voting classifier, bagging, Boosting
#VotingClassifier
#BaggingClassifier
#AdaBoostClassifier
#GrdientBoostingClassifier
#XGBClassifier
