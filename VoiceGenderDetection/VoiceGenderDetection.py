import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dtr = pd.read_csv('C://Users//jatot//OneDrive//Desktop//Null Class Tasks//Voice Gender Detection//Dataset//train.csv')
# dtr.drop(labels = ['mindom', 'median'], axis = 1, inplace= True)
dtr.info()
X = dtr.iloc[:,1:21].values
Y = dtr.iloc[:,21].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.95, random_state = 0)

X_train = X
Y_train = Y
from sklearn.preprocessing import StandardScaler
FS = StandardScaler()
X_train = FS.fit_transform(X_train)
X_test = FS.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state = 0)
print(classifier.fit(X_train, Y_train))
forest = classifier[6]
importances = pd.DataFrame({'feature':dtr.iloc[:, 1:21].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)
Y_pred = classifier.predict(X_test)
dte = pd.read_csv("C://Users//jatot//OneDrive//Desktop//Null Class Tasks//Voice Gender Detection//Dataset//test.csv")   # Read the test dataset provided to us
# dte.drop(labels = ['mindom', 'median'], axis = 1, inplace= True)
print(dte)
X_TEST = dte.iloc[:,1:].values           # Same procedure is to be followed as it was followed in case of training dataset
FST = StandardScaler()
X_TEST = FST.fit_transform(X_TEST)
Y_PRED = classifier.predict(X_TEST)
Predicted = pd.DataFrame(Y_PRED)   # New Dataframe was created
ss = pd.read_csv("C://Users//jatot//OneDrive//Desktop//Null Class Tasks//Voice Gender Detection//Dataset//sample-submission.csv")
dataset = pd.concat([ss['Id'], Predicted], axis=1)     # Concatenation (Merging) with sample-submission dataset and creating a new dataset
dataset.columns = ['Id', 'label']      # Column names were assigned for new dataset formed
dataset.to_csv('sample_submission-rf.csv', index = False)   # Exported the new dataset with name
