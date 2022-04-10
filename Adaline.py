import json
import numpy as np


class Adaline:
    """
    Constuctor for Adaline class
    :param self: Adaline
    :param jsonfile: T1
    :param condition: T2
    :return: opencv solution, my implementation
    """

    def __init__(self, jsonfile: str, condition: bool):
        # class member points which is a list of points. each point is a tuple ->(x,y, value 1 or -1)
        self.points = []
        # read the points from the json file
        jsonfile = "" + str(jsonfile) + ".json"
        if jsonfile is not None:
            with open(jsonfile, 'r') as jsonfile:
                json_object = json.load(jsonfile)
                jsonfile.close()
                for i in range(len(json_object)):
                    p = json_object[str(i)]
                    if condition:
                        if p['y'] > 1:
                            single_point = (p['x'], p['y'], 1)
                        else:
                            single_point = (p['x'], p['y'], -1)
                        self.points.append(single_point)


def saveDataToCSV():
    print()


def train(DataSet: list):
    w1 = w2 = b = alpha = float("{0:.3f}".format(np.random.rand()))
    mseTotal = []
    table = []
    while True:
        delta = []
        mse = []
        for point in DataSet:
            Yin = b + w1 * point[0] + w2 * point[1]
            error = point[2] - Yin
            if error != 0:
                w1 = w1 + alpha * error * point[0]
                w2 = w2 + alpha * error * point[1]
                b = b + alpha * error
                delta.append((alpha * error * point[0], alpha * error * point[1], alpha * error))
            else:
                delta.append((0, 0, 0))
            mse.append(error ** 2)
        mseTotal.append(np.sum(mse))
        if mseTotal[-1] < 10:
            break


def test(DataSet: list):
    print()


if __name__ == '__main__':
    data = Adaline("dataSets/dataSet1", True)
