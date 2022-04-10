import numpy as np
import json
from os.path import exists
import pandas as pd


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


def saveDataToCsv(data):
    file_exists = exists("tableA.csv")
    fields = ["x", "y", "value", "Delta1", "Delta2", "Bias", "MSE"]
    rows = []
    for row in data:
        rows.append({"X": row[0], "Y": row[1], "Value": row[2], "Delta1": row[3], "Delta2": row[4], "Bias": row[5],
                     "MSE": row[6]})
    if file_exists:
        df = pd.read_csv("TableA.csv")
        for row in rows:
            df = df.append(row, ignore_index=True)
        df.to_csv("TableA.csv")
    else:
        df = pd.DataFrame(columns=fields)
        for row in rows:
            df = df.append(row, ignore_index=True)
        df.to_csv("TableA.csv")


if __name__ == '__main__':
    print("hello")
