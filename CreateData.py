# This is a sample Python script.
import numpy as np
import json
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def createDataSet():
    points = []
    while len(points) != 1000:
        x = np.random.randint(-10000, 10001)
        y = np.random.randint(-10000, 10001)
        if (x/100, y/100) not in points:
            points.append((x/100, y/100))
    return points


def saveDataSet(DataSetPoints):
    data = {}
    for i in range(len(DataSetPoints)):
        data[str(i)] = {"x": DataSetPoints[i][0], "y": DataSetPoints[i][1]}
    with open('test.json', 'w') as f:
        json.dump(data, f)
        print("saved")


if __name__ == '__main__':
    points = createDataSet()
    saveDataSet(points)
