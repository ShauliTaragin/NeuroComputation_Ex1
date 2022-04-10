import json
import numpy as np


class Adaline:
    """
    Constructor for Adaline class
    :param self: Our Adaline Network
    :param jsonfile: The data set of points as a json file
    :param condition: which case are we using now , part A or B
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
        jsonfile = "" + str(jsonfile) + ".json"
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
                        self.points.append(single_point)
                    else:
                        # if we are in case A the condition is whether 4<=x^2+y^2<=9
                        cond = (p['x'] ** 2) + (p['y'] ** 2)
                        if 4 <= cond <= 9:
                            single_point = (p['x'], p['y'], 1)
                        else:
                            single_point = (p['x'], p['y'], -1)
                        self.points.append(single_point)

    def saveDataToCSV(self):
        print()

    def train(self, DataSet: list):
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
                self.w1 = w1
                self.w2 = w2
                self.b = b
                break

    def test(self, DataSet: list) -> list:
        ans = []
        for point in DataSet:
            pred = self.b + self.w1 * point[0] + self.w2 * point[1]
            ans.append((point[0], point[1], pred))
        return ans

    def valid(self, DataSet: list, condition: bool)->float:

        correct_ans = 0
        pred = self.test(DataSet)
        for point in DataSet:
            if abs(pred[2] - point[2]) < self.eps:
                correct_ans += 1
        return correct_ans / len(DataSet)


if __name__ == '__main__':
    data = Adaline("dataSets/dataSet1", True)
