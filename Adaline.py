import numpy as np


def saveDataToCSV():


def train(DataSet: list):
    w1 = w2 = b = alpha = float("{0:.3f}".format(np.random.rand()))
    mseTotal = []
    table=[]
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

def test(DataSet:list):


if __name__ == '__main__':
    print()
