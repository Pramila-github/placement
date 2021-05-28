import numpy as np
import pandas as pd
import pickle
dataset = pd.read_csv('placementdata.csv')
def convert_to_int(word):
    word_dict = {'Good':1, 'Bad':0,'Yes':1,'No':0,'completed':1,'none':0}
    return word_dict[word]


dataset = dataset.drop('sl_no', axis=1)


# catgorising col for further labelling
dataset["gender"] = dataset["gender"].astype('category')
dataset["comSkill"] = dataset["comSkill"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["Internship"] = dataset["Internship"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["sports"] = dataset["sports"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')

# labelling the columns
dataset["gender"] = dataset["gender"].cat.codes
dataset["comSkill"] = dataset["comSkill"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["Internship"] = dataset["Internship"].cat.codes
dataset["sports"] = dataset["sports"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes


# selecting the features and labels
X = dataset.iloc[:, :-1].values
Y=dataset.loc[:,['status']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#importing the random forest classifier model and training it on the dataset
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier()
regressor=regressor.fit(X_train,Y_train)

#print(regressor.score(X_test, Y_test))

# Saving model to disk
pickle.dump(regressor, open('final_model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('final_model.pkl','rb'))
#print(model.predict([[0,convert_to_int('Good'),39,0,58,0,0,58,0,4,convert_to_int('none'),58,58,convert_to_int('No')]]))
