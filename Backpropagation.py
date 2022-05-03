import json
import random
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
        np.random.seed(23)

        self.w1 = np.random.normal(scale=0.5, size=(2, 8))
        self.w2 = np.random.normal(scale=0.5, size=(8, 4))
        self.w3 = np.random.normal(scale=0.5, size=(4, 2))
        self.w4 = np.random.normal(scale=0.5, size=(2, 1))

        # read the points from the json file
        self.points = ParseJson(jsonfile, condition)
        random.shuffle(self.points)
        self.train_x = np.asarray([[x, y] for x, y, z in self.points])
        self.train_y = np.asarray([[z] for x, y, z in self.points])
        # self.train_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in self.train_y])
        self.N = self.train_y.size

    def train(self):
        iterNumber = 0
        errors = []
        while iterNumber < 5000:
            # feed forward
            print(iterNumber)
            z_1 = np.dot(self.train_x, self.w1)
            a_1 = sigmoid(z_1)

            z_2 = np.dot(a_1, self.w2)
            a_2 = sigmoid(z_2)

            z_3 = np.dot(a_2, self.w3)
            a_3 = sigmoid(z_3)

            z_4 = np.dot(a_3, self.w4)
            a_4 = sigmoid(z_4)
            # calculate cost
            cost = mean_squared_error(a_4, self.train_y)
            errors.append(cost)
            # if len(errors) >= 2 and errors[-1] > errors[-2]:
            #     break

            # back propagation
            # this weights are listed backwards i.e d_w1 is between the output and the 1 from last layer
            d_w1 = (a_4 - self.train_y) * a_4 * (1 - a_4)
            d_w2 = np.dot(d_w1, self.w4.T) * a_3 * (1 - a_3)
            d_w3 = np.dot(d_w2, self.w3.T) * a_2 * (1 - a_2)
            d_w4 = np.dot(d_w3, self.w2.T) * a_1 * (1 - a_1)

            W4_update = np.dot(a_3.T, d_w1) / self.N
            W3_update = np.dot(a_2.T, d_w2) / self.N
            W2_update = np.dot(a_1.T, d_w3) / self.N
            W1_update = np.dot(self.train_x.T, d_w4) / self.N

            # correction
            self.w4 -= self.learning_rate * W4_update
            self.w3 -= self.learning_rate * W3_update
            self.w2 -= self.learning_rate * W2_update
            self.w1 -= self.learning_rate * W1_update

            iterNumber += 1

    def test(self, DataSetPath: str, condition: bool):
        DataSet = ParseJson(DataSetPath, condition)
        test_x = np.asarray([[x, y] for x, y, z in DataSet])
        test_y = np.asarray([[z] for x, y, z in DataSet])
        test_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in test_y])
        Z1 = np.dot(test_x, self.w1)
        A1 = sigmoid(Z1)

        # hidden layers
        Z2 = np.dot(A1, self.w2)
        A2 = sigmoid(Z2)

        # on output layer
        Z3 = np.dot(A2, self.w3)
        A3 = sigmoid(Z3)

        # Z4 = np.dot(A3, self.w4)
        # A4 = sigmoid(Z4)
        # A4 = [1 if x > 0.5 else -1 for x in A4]

        mse = mean_squared_error(A3, test_y)
        rightAnswer = 0
        # for i in range(len(test_y)):
        #     if A4[i] == test_y[1]:
        #         rightAnswer += 1
        acc = accuracy(A3, test_y)
        print(acc)
        return A3

    # def valid(self, DataSetPath: str, condition: bool) -> float:
    #     DataSet = ParseJson(DataSetPath, condition)
    #     test_y = np.asarray([z if z == 1 else 0 for x, y, z in DataSet])
    #     test_y = np.asarray([[1, 0] if z == 1 else [0, 1] for z in test_y])
    #     mse = mean_squared_error(A2, test_y)
    #     acc = accuracy(A2, test_y)


if __name__ == '__main__':
    model = Backpropagation("dataSets/dataSet4", False)
    model.train()
    model.test("dataSets/test2", False)
    # plotPoints("dataSets/dataSet2")
    # print(model.valid("dataSets/dataSet1", True))
    # print(model.valid("dataSets/dataSet2", True))
    # print(model.valid("dataSets/dataSet3", True))

    # model = Backpropagation("dataSets/dataSet1", False)
    # model.train()
    # print(model.valid("dataSets/dataSet1", False))
    # print(model.valid("dataSets/dataSet2", False))
    # print(model.valid("dataSets/dataSet3", False))
