#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#load data and see different aspects of data
loans = pd.read_csv('loan_data.csv')
loans.info()
loans.head()
loans.describe()

#Data analysis by different plots like histogram plot between credit.policy column and fico column in data
plt.figure(figsize=(10,4))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,bins=30,color='blue',label='credit.policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,bins=30,color='red',label='credit.policy=0')
plt.legend()
plt.xlabel('FICO')
      
#histogram plot for data using  not.fully.paid and fico columns
plt.figure(figsize=(10,4))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,bins=30,color='blue',label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,bins=30,color='red',label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#countplot for 'not.fully.paid' column using "purpose" as xlabel
plt.figure(figsize=(10,4))
sns.countplot(x="purpose",hue='not.fully.paid',data=loans,color='blue',palette='Set2')

#jointplot for "fico" and "int.rate" columns
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

#lmplot for fico and int.rate using hue as not.fully.paid on credit.policy column
plt.figure(figsize=(20,10))
sns.lmplot(x='fico',y='int.rate',hue='not.fully.paid',data=loans,palette='Set1',col='credit.policy')


#data customization step using pd.get_dummies method to convert string values to float values of purpose column
loans.info()
cat_feats= ['purpose']
final_data =pd.get_dummies(loans,columns=cat_feats,drop_first=True)


final_data.info()
final_data.head()
 
#train test split using sklearn
from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#apply DecisionTreeClassifier on data
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction1=dtree.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction1))
print('\n')
print(classification_report(y_test,prediction1))

#apply randomforest classifier 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train,y_train)


prediction2=rfc.predict(X_test)
print(confusion_matrix(y_test,prediction2))
print('\n')
print(classification_report(y_test,prediction2))

#Thank you

