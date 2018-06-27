#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#load and analyse the data
df=pd.read_csv('kyphosis.csv')
df.head()
df.info()
sns.pairplot(df,hue='Kyphosis',palette='bright',kind='scatter')

#train_test_split of data
from sklearn.model_selection import train_test_split
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#apply DecisionTree classifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

#import confusion_matrix and classification_report and apply them to see scores
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


#visulization of decision tree using pydot and graphviz libraries
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 



#apply RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
prediction1= rfc.predict(X_test)


#apply confusion_matrix and classification_report for scores
print(confusion_matrix(prediction1,y_test))
print("\n")
print(classification_report(prediction1,y_test))


