import json
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS

def plotPoints(DataSetPath):
    DataSet = ParseJson(DataSetPath, False)
    x_1 = []
    y_1 = []
    x_0 = []
    y_0 = []
    fig = plt.figure()
    for row in DataSet:
        if row[1] > 1:
            x_1.append(row[0])
            y_1.append(row[1])
        else:
            x_0.append(row[0])
            y_0.append(row[1])
    # fig.add_artist(lines.Line2D([-100, 0], [100, 0]))
    # fig.add_artist(lines.Line2D([0, -100], [0, 100]))
    plt.scatter(x_1, y_1, color="green")
    plt.scatter(x_0, y_0, color="red")
    plt.plot([100, -100], [0, 0], color='k')
    plt.plot([0, 0], [100, -100], color='k')
    plt.show()


def createDataSet():
    points = []
    while len(points) != 1000:
        x = np.random.randint(-10000, 10001)
        y = np.random.randint(-10000, 10001)
        if (x / 100, y / 100) not in points:
            points.append((x / 100, y / 100))
    return points


def saveDataSet(DataSetPoints):
    data = {}
    for i in range(len(DataSetPoints)):
        data[str(i)] = {"x": DataSetPoints[i][0], "y": DataSetPoints[i][1]}
    with open('test.json', 'w') as f:
        json.dump(data, f)
        print("saved")


def saveDataToCsv(data: list):
    file_exists = exists("tableB.csv")
    rows = []
    for row in data:
        if file_exists:
            rows.append(
                {"X": row['X'], "Y": row['Y'], "Value": row['Value'], "DeltaW1": row['DeltaW1'],
                 "DeltaW2": row['DeltaW2'],
                 "DeltaBias": row['DeltaBias'],
                 "MSE": row['MSE']})
    if file_exists:
        rows.append(
            {"X": '---', "Y": '---', "Value": '---', "DeltaW1": '---', "DeltaW2": '---',
             "DeltaBias": '---',
             "MSE": '---'})
        df = pd.DataFrame(rows)
        df_old = pd.read_csv("tableB.csv")
        df = pd.concat([df_old, df])
        df = df.drop(columns=['Unnamed: 0'])
        df.to_csv("tableB.csv")
    else:
        df = pd.DataFrame(rows)
        df.to_csv("tableB.csv")


def ParseJson(path: str, condition: bool):
    points = []
    jsonfile = "" + str(path) + ".json"
    if jsonfile is not None:
        with open(jsonfile, 'r') as jsonfile:
            json_object = json.load(jsonfile)
            jsonfile.close()
            for i in range(len(json_object)):
                p = json_object[str(i)]
                if condition:
                    # if we are in case A the condition is whether y is greater then 1
                    if p['y'] > 1:
                        single_point = (p['x'], p['y'], 1)
                    else:
                        single_point = (p['x'], p['y'], -1)
                    points.append(single_point)
                else:
                    # if we are in case A the condition is whether 4<=x^2+y^2<=9
                    cond = (p['x'] ** 2) + (p['y'] ** 2)
                    if 4 <= cond <= 9:
                        single_point = (p['x'], p['y'], 1)
                    else:
                        single_point = (p['x'], p['y'], -1)
                    points.append(single_point)
    return points


def forward(classifier, X, layers=None):
    if layers is None or layers == 0:
        layers = classifier.n_layers_

    # Initialize first layer
    activation = X

    # Forward propagate
    hidden_activation = ACTIVATIONS[classifier.activation]
    for i in range(layers - 1):
        weight_i, bias_i = classifier.coefs_[i], classifier.intercepts_[i]
        activation = (activation @ weight_i) + bias_i
        if i != layers - 2:
            hidden_activation(activation)
    if activation.shape[1] > 1:
        ans = []
        for j in range(activation.shape[1]):
            ans.append(classifier._label_binarizer.inverse_transform(activation[:, j]))
        return ans
    hidden_activation(activation)
    return classifier._label_binarizer.inverse_transform(activation)
class MODEL:
    def __init__(self, jsonfile: str, condition: bool):
        # class member points which is a list of points. each point is a tuple ->(x,y, value 1 or -1)
        self.learning_rate = 0.1
        np.random.seed(10)

        self.w1 = np.random.normal(scale=0.5, size=(2, 2))
        self.w2 = np.random.normal(scale=0.5, size=(2, 2))

        # read the points from the json file
        self.points = ParseJson(jsonfile, condition)
        self.train_x = np.asarray([[x, y] for x, y, z in self.points])
        self.train_y = np.asarray([[z] if z == 1 else [0] for x, y, z in self.points])
        # self.train_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in self.train_y])
        self.N = self.train_y.size

if __name__ == '__main__':
    model = MODEL("dataSets/test2", False)
    clf = MLPClassifier(solver= 'adam',
                    hidden_layer_sizes=(4, 8),
                    max_iter = 100,
                    activation='relu',
                    random_state=42)

    clf.fit(model.train_x, model.train_y)
    # double = model.test("dataSets/test", False)
    # x_test = double[0]
    # y_test = double[1]
    # y_predict = clf.predict(x_test)

    layer_i = forward(clf, model.train_x, 2)
    # print("Accuracy of BP (train):  %.2f precent" % (metrics.accuracy_score(y_test, y_predict) * 100))
    print("Score of correct prediction: ", clf.score(model.train_x, model.train_y) * 100, "%")
