import json
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met
import seaborn as sns
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

def createConfusionMatrix(DataSetPath):
    # DataSet = ParseJson("dataSets/test2", True)
    # train on DataSetPath
    model = MODEL(DataSetPath, False)
    clf = MLPClassifier(solver='adam',
                        hidden_layer_sizes=(4, 8),
                        max_iter=100,
                        activation='relu',
                        random_state=42)

    clf.fit(model.train_x, model.train_y)
    # test = adaline.test("dataSets/test2", True)
    double = ParseJson("dataSets/test2", False)
    x_test = [[x,y] for x,y,z in double]
    y_test =  [[z] for x,y,z in double]
    y_predict = clf.predict(x_test)
    y_predict = [x if x == 1 else -1 for x in y_predict]
    # target = [row[2] for row in DataSet]
    # pred = [row[2] for row in test]
    conf = met.confusion_matrix(y_test, y_predict)
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

def neuron_diagram(classifier, x, i, curr_layer, output_y):
    for neuron in curr_layer:
        x_true = x[neuron == 1, 1]
        y_true = x[neuron == 1, 0]
        x_false = x[neuron == 0, 1]
        y_false = x[neuron == 0, 0]
        plt.scatter(x=x_true, y=y_true, c='yellow')
        plt.scatter(x=x_false, y=y_false, c='green')
        plt.show()
    if output_y:
        output_layer = forward(classifier, x)
        output_layer_x_true = x[output_layer == 1, 1]
        output_layer_y_true = x[output_layer == 1, 0]
        output_layer_x_false = x[output_layer == 0, 1]
        output_layer_y_false = x[output_layer == 0, 0]
        plt.scatter(x=output_layer_x_true, y=output_layer_y_true, c='yellow')
        plt.scatter(x=output_layer_x_false, y=output_layer_y_false, c='green')
        plt.show()


def find_layer(mlp, x, layers_num):
    hidden_act_func = ACTIVATIONS[mlp.activation]
    neurons_activation = x
    layers = []

    # feed forward the layers
    for i in range(0, layers_num - 2):
        weight, bias = mlp.coefs_[i], mlp.intercepts_[i]
        neurons_activation = np.matmul(neurons_activation, weight) + bias
        hidden_act_func(neurons_activation)

    # in the last iterate we wont activate the function
    if layers_num > 1:
        weight, bias = mlp.coefs_[layers_num - 2], mlp.intercepts_[layers_num - 2]
        neurons_activation = np.matmul(neurons_activation, weight) + bias

    # check if there are 2 or more
    if neurons_activation.shape[1] >= 2:
        for i in range(0, neurons_activation.shape[1]):
            layers.append(mlp._label_binarizer.inverse_transform(neurons_activation[:, i]))
        return layers

    hidden_act_func(neurons_activation)
    layer = mlp._label_binarizer.inverse_transform(neurons_activation)
    return layer



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
    createConfusionMatrix("dataSets/dataSet4")
    model = MODEL("dataSets/test2", False)
    clf = MLPClassifier(solver='adam',
                        hidden_layer_sizes=(4, 8),
                        max_iter=100,
                        activation='relu',
                        random_state=42)

    clf.fit(model.train_x, model.train_y)
    for layer in range(2, clf.n_layers_):
        layer_i = forward(clf, model.train_x, layer)
        if layer == clf.n_layers_ -1:
            neuron_diagram(clf,model.train_x,layer-1,layer_i ,True)
        else:
            neuron_diagram(clf, model.train_x, layer - 1, layer_i, False)
    # double = model.test("dataSets/test", False)
    # x_test = double[0]
    # y_test = double[1]
    # y_predict = clf.predict(x_test)

    # layer_i = forward(clf, model.train_x, 2)
    # print("Accuracy of BP (train):  %.2f precent" % (metrics.accuracy_score(y_test, y_predict) * 100))
    print("Score of correct prediction: ", clf.score(model.train_x, model.train_y) * 100, "%")




    # layer_i = find_layer(clf, model.train_x, 4)
    # diff=layer_i-layer_f
    # zipped = zip(layer_f, layer_f)
    # ans = []
    # for a, b in zipped:
    #     ans.append(a - b)
    # s=0
    # for a in ans:
    #     s+=a.sum()
    # # print("Accuracy of BP (train):  %.2f precent" % (metrics.accuracy_score(y_test, y_predict) * 100))
    # print("Score of correct prediction: ", clf.score(model.train_x, model.train_y) * 100, "%")
