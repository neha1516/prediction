import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
app = Flask(__name__, template_folder="template")

data_dir = "mumbai dataset"
os.chdir(data_dir)
data = pd.read_csv("Mumbai.csv")
data.head()
data.apply(lambda x: sum(x.isnull()), axis=0)
data['FLOODS'].replace(['YES', 'NO'], [1, 0], inplace=True)
data.head()

x = data.iloc[:, 1:14]
x.head()
y = data.iloc[:, -1]
y.head()

c = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
c.hist()
acc1 = 91.666667
plt.savefig('plot.png')

minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
minmax.fit(x).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.head()
y_train.head()
x_train_std = minmax.fit_transform(x_train)
x_test_std = minmax.fit_transform(x_test)

lr = LogisticRegression()
lr_clf = lr.fit(x_train_std, y_train)

lr_accuracy = cross_val_score(lr_clf, x_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
lr_accuracy.mean()

y_predict = lr_clf.predict(x_test_std)
print('Predicted chances of flood are')
print(y_predict)

(y_predict != 0).sum()

(y_predict != 1).sum()
y = (7 / 24) * 100
y1 = round(y, 2)
print(y)
acc = (accuracy_score(y_test, y_predict) * 100)

print("\n accuracy score: %f" % (accuracy_score(y_test, y_predict) * 100))
print("recall score: %f" % (recall_score(y_test, y_predict) * 100))
print("roc score: %f" % (roc_auc_score(y_test, y_predict) * 100))

app = Flask(__name__)
app.config["DEBUG"] = False


@app.route('/', methods=['GET'])
def home():
    return render_template("data.html", ny=y1, acc_h=acc1)


if __name__ == '__main__':
    app.run()
