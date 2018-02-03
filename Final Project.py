"""
implement Naive Bayes, SVM, Logistic regression, Random Forest
"""

# Importing the libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, StandardScaler
from sklearn.model_selection import cross_val_score


# Importing the dataset
dat=pd.read_csv('Test.csv')

sns.set(font_scale=.6)
corr = dat.corr()
cover= np.zeros_like(corr)
cover[np.triu_indices_from(cover)]=True
# Heatmap
with sns.axes_style("white"):
    sns.heatmap(corr, mask=cover, square=True,cmap="RdBu", annot=True)


#encode categorical columns
vars=['hour','alcohol involved','road system','road surface type','light condition','road divided by','cell phone in use flag', 'year of vehicle']
for var in vars:
    combine='var'+'_'+var
    combine = pd.get_dummies(dat[var], prefix=var)
    temp=dat.join(combine)
    dat=temp
    
data_vars=dat.columns.values.tolist()
to_keep=[i for i in data_vars if i not in vars]
final=dat[to_keep]
final.columns.values

#set x and y
X=final.iloc[:,final.columns!='severity'].values
y=final.iloc[:,5].values
            
#impute
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:,:5])
X[:,:5]=imputer.transform(X[:,:5])
          
#split training and test set
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=0)

#feature scaling numeric variables
sc=StandardScaler()
X_train[:,:5]=sc.fit_transform(X_train[:,:5])
X_test[:,:5]=sc.transform(X_test[:,:5])

sns.pairplot(dat, hue='severity')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Logistic Regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
classifier1=LogisticRegression(multi_class='multinomial',solver='newton-cg')
classifier1.fit(X_train, y_train)

#predict test
y_pred1=classifier1.predict(X_test)

#confusion matrix
cm1=confusion_matrix(y_test, y_pred1)
print(cm1)

#10 fold validation
accuracies= cross_val_score(estimator=classifier1, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
print('accuracies: ',accuracies)
print('mean: ',accuracies.mean())
print('stdev: ',accuracies.std())

print(classification_report(y_test, y_pred1))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   KNN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
classifier2=KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier2.fit(X_train, y_train)

#predict test
y_pred2=classifier2.predict(X_test)

#confusion matrix
cm2=confusion_matrix(y_test, y_pred2)
print(cm2)

#10 fold validation
accuracies= cross_val_score(estimator=classifier2, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
print('accuracies: ',accuracies)
print('mean: ',accuracies.mean())
print('stdev: ',accuracies.std())

print(classification_report(y_test, y_pred2))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   SVM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classifier3=SVC(kernel='rbf',random_state=0)
classifier3.fit(X_train, y_train)

#predict test
y_pred3=classifier3.predict(X_test)

#confusion matrix
cm3=confusion_matrix(y_test, y_pred3)
print(cm3)

print(classification_report(y_test, y_pred3))




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Random Forest
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classifier4=RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier4.fit(X_train, y_train)

#predict test
y_pred4=classifier4.predict(X_test)

#confusion matrix
cm4=confusion_matrix(y_test, y_pred4)
print(cm4)


print(classification_report(y_test, y_pred4))
