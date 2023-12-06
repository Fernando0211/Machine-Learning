import pandas as pd
from sklearn.linear_model import Perceptron

df = pd.read_csv("Social_Network_Ads.csv")

df = df.drop(columns = "User ID")

dummy = {
    'Gender': {
        'prefix': 'Gender',
        'sep': ';'
    }} 

for column_name, dummy_data in dummy.items():
    
    dummies = df[column_name].str.get_dummies(sep=dummy_data['sep'])

    dummies.columns = map(lambda col: f'{dummy_data["prefix"]}_{col}', dummies.columns)
    
    df = pd.concat([dummies, df], axis=1)
    
df = df.drop(columns = "Gender")

x_train = df.iloc[0:320, 0:4]

x_test = df.iloc[320:400, 0:4]

y_train = df.iloc[0:320, 4]

y_test = df.iloc[320:400, 4] 

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
Perceptron()
test_score = clf.score(x_test, y_test)
train_score = clf.score(x_train, y_train)

print("Score using seen data: ", test_score)
print("Score using unseen data: ", train_score)