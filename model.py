import pandas as pd
import numpy as np

dataframe = pd.read_csv('Placementsample1.csv')
def convert_to_int(word):
    word_dict = {'Good':1, 'Bad':0,'yes':1,'no':0,'completed':1,'none':0}
    return word_dict[word]
X=dataframe.loc[:,['comSkill','ssc_p','hsc_p','degree_p','No_certi','internships','etest_p','sports','placetest_p']]
y=dataframe.loc[:,['status']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
regressor=RandomForestClassifier()
regressor=regressor.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print(scorce)
import joblib  # Save to file

joblib_file = 'model.pkl'   
joblib.dump(regressor, joblib_file)

#print(model.predict([[convert_to_int('Good'),69.6,68.4,78.3,5,convert_to_int('completed'),60,convert_to_int('yes'),80]]))
