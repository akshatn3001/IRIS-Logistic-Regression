import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('Iris.csv')
df=df[df['Species']!='Iris-setosa']
df['Species']=df['Species'].map({'Iris-versicolor':1,'Iris-virginica':0})
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
parameters={'penalty':['l1','l2'],'C':[1,2,3,4,5,6,7,8,9,10,20,30,40],'max_iter':[100,200,300]}
cl_reg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
cl_reg.fit(X_train,y_train)
y_pred=cl_reg.predict(X_test)
print(cl_reg.predict([[101,6.3,3.3,6.0,2.5]]))