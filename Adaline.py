import json
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met
import seaborn as sns


def plotPoints(DataSetPath):
    DataSet = ParseJson("dataSets/test2", True)
    adaline = Adaline(DataSetPath, True)
    adaline.train()
    test = adaline.test("dataSets/test2", True)
    fig, axes = plt.subplots(1, 2)
    x_1 = []
    y_1 = []
    x_0 = []
    y_0 = []
    for row in DataSet:
        if row[2] == 1:
            x_1.append(row[0])
            y_1.append(row[1])
        else:
            x_0.append(row[0])
            y_0.append(row[1])
    axes[0].scatter(x_0, y_0, color="red")
    axes[0].scatter(x_1, y_1, color="green")
    x_test_1 = []
    y_test_1 = []
    x_test_0 = []
    y_test_0 = []
    for row in test:
        if row[2] == 1:
            x_test_1.append(row[0])
            y_test_1.append(row[1])
        else:
            x_test_0.append(row[0])
            y_test_0.append(row[1])
    axes[1].scatter(x_test_0, y_test_0, color="red")
    axes[1].scatter(x_test_1, y_test_1, color="green")
    axes[0].plot([100, -100], [0, 0], color='k')
    axes[0].plot([0, 0], [100, -100], color='k')
    axes[1].plot([100, -100], [0, 0], color='k')
    axes[1].plot([0, 0], [100, -100], color='k')
    axes[0].title.set_text("Correct answer")
    axes[1].title.set_text("predicted answer")
    plt.suptitle("True is when y > 1, test 2")
    plt.savefig("conditionATest2.jpg")
    plt.show()


def createConfusionMatrix(DataSetPath):
    DataSet = ParseJson("dataSets/test2", True)
    # train on DataSetPath
    adaline = Adaline(DataSetPath, True)
    adaline.train()
    test = adaline.test("dataSets/test2", True)
    target = [row[2] for row in DataSet]
    pred = [row[2] for row in test]
    conf = met.confusion_matrix(target, pred)
    conf_mat = np.zeros((2, 2))
    try:
        for i in range(2):
            for j in range(2):
                conf_mat[i, j] = conf[i, j]
    except IndexError as e:
        pass
    conf = conf_mat
    labels = ['True Neg\n' + str(conf[0, 0]), 'False Pos\n' + str(conf[0, 1]), 'False Neg\n' + str(conf[1, 0]),
              'True Pos\n' + str(conf[1, 1])]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(conf, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion matrix for first condition on test2\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig("confMatTest2CondA.jpg")
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


class Adaline:
    """
    Constructor for Adaline class
    :param self: Our Adaline Network
    :param jsonfile: The data set of points as a json file
    :param condition: which case are we using now , part A or B
    case A: y > 1
    case B: 4 <= x^2 + y^2 <=9
    The data set of points as a list
    """

    def __init__(self, jsonfile: str, condition: bool):
        # class member points which is a list of points. each point is a tuple ->(x,y, value 1 or -1)
        self.points = []
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.eps = 0.5
        # read the points from the json file
        self.points = ParseJson(jsonfile, condition)

    def train(self):
        w1 = w2 = b = float("{0:.3f}".format(np.random.rand()))
        alpha = 0.1 / 2
        mseTotal = []
        iterNum = 0
        while True:
            iterNum += 1
            delta = []
            mse = []
            rows = []
            for point in self.points:
                Yin = b + (w1 * point[0] / 100) + (w2 * point[1] / 100)
                error = point[2] - Yin
                if error != 0:
                    w1 = w1 + alpha * error * point[0] / 100
                    w2 = w2 + alpha * error * point[1] / 100
                    b = b + alpha * error
                    delta.append((alpha * error * point[0] / 100, alpha * error * point[1] / 100, alpha * error))
                else:
                    delta.append((0, 0, 0))
                mse.append(error ** 2)
                rows.append(
                    {"X": point[0], "Y": point[1], "Value": point[2], "DeltaW1": delta[-1][0], "DeltaW2": delta[-1][1],
                     "DeltaBias": delta[-1][2],
                     "MSE": mse[-1]})
            mseTotal.append(np.sum(mse))
            saveDataToCsv(rows)
            if mseTotal[-1] < 10 or (len(mseTotal) >= 2 and abs(mseTotal[-1] - mseTotal[-2]) < 0.001) or iterNum >= 50:
                self.w1 = w1
                self.w2 = w2
                self.b = b
                break

    def test(self, DataSetPath: str, condition: bool) -> list:
        DataSet = ParseJson(DataSetPath, condition)
        ans = []
        for point in DataSet:
            pred = 1 if self.b + self.w1 * point[0] + self.w2 * point[1] >= 0 else -1
            ans.append((point[0], point[1], pred))
        return ans

    def valid(self, DataSetPath: str, condition: bool) -> float:
        DataSet = ParseJson(DataSetPath, condition)
        correct_ans = 0
        pred = self.test(DataSetPath, condition)
        for i in range(len(DataSet)):
            if pred[i][2] == DataSet[i][2]:
                correct_ans += 1
        return correct_ans / len(DataSet)


if __name__ == '__main__':
    model = Adaline("dataSets/dataSet2", False)
    model.train()
    plotPoints("dataSets/dataSet2")
    print(model.valid("dataSets/dataSet1", True))
    print(model.valid("dataSets/dataSet2", True))
    print(model.valid("dataSets/dataSet3", True))

    model = Adaline("dataSets/dataSet1", False)
    model.train()
    print(model.valid("dataSets/dataSet1", False))
    print(model.valid("dataSets/dataSet2", False))
    print(model.valid("dataSets/dataSet3", False))

