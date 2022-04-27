import json
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.size)


def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


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


class Backpropagation:
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
        self.train_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in self.train_y])
        self.N = self.train_y.size

    def train(self):
        iterNumber = 0
        errors = []
        while iterNumber < 5000:
            print(iterNumber)
            # feed forward
            z_1 = np.dot(self.train_x, self.w1)
            a_1 = sigmoid(z_1)

            z_2 = np.dot(a_1, self.w2)
            a_2 = sigmoid(z_2)

            # calculate cost
            cost = mean_squared_error(a_2, self.train_y)
            errors.append(cost)

            # back propagation
            d_w1 = (a_2 - self.train_y) * a_2 * (1 - a_2)
            d_w2 = np.dot(d_w1, self.w2.T) * a_1 * (1 - a_1)

            W2_update = np.dot(a_1.T, d_w1) / self.N
            W1_update = np.dot(self.train_x.T, d_w2) / self.N
            # correction
            self.w2 -= self.learning_rate * W2_update
            self.w1 -= self.learning_rate * W1_update

            iterNumber += 1

    def test(self, DataSetPath: str, condition: bool):
        DataSet = ParseJson(DataSetPath, condition)

        test_x = np.asarray([[x, y] for x, y, z in DataSet])
        test_y = np.asarray([z if z == 1 else 0 for x, y, z in DataSet])
        test_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in test_y])

        Z1 = np.dot(test_x, self.w1)
        A1 = sigmoid(Z1)

        # on output layer
        Z2 = np.dot(A1, self.w2)
        A2 = sigmoid(Z2)
        mse = mean_squared_error(A2, test_y)
        acc = accuracy(A2, test_y)
        return A2, mse, acc

    # def valid(self, DataSetPath: str, condition: bool) -> float:
    #     DataSet = ParseJson(DataSetPath, condition)
    #     test_y = np.asarray([z if z == 1 else 0 for x, y, z in DataSet])
    #     test_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in test_y])
    #     mse = mean_squared_error(A2, test_y)
    #     acc = accuracy(A2, test_y)


if __name__ == '__main__':
    model = Backpropagation("dataSets/dataSet2", True)
    model.train()
    model.test("dataSets/test2", True)
    # plotPoints("dataSets/dataSet2")
    # print(model.valid("dataSets/dataSet1", True))
    # print(model.valid("dataSets/dataSet2", True))
    # print(model.valid("dataSets/dataSet3", True))

    # model = Backpropagation("dataSets/dataSet1", False)
    # model.train()
    # print(model.valid("dataSets/dataSet1", False))
    # print(model.valid("dataSets/dataSet2", False))
    # print(model.valid("dataSets/dataSet3", False))
