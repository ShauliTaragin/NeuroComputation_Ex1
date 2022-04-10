import json


class Adaline:
    """
    Constuctor for Adaline class
    :param self: Adaline
    :param jsonfile: T1
    :param condition: T2
    :return: opencv solution, my implementation
    """
    def __init__(self , jsonfile:str , condition:bool):
        # class member points which is a list of points. each point is a tuple ->(x,y, value 1 or -1)
        self.points = []
        # read the points from the json file
        jsonfile = "" + str(jsonfile) + ".json"
        if jsonfile is not None:
            with open(jsonfile , 'r') as jsonfile:
                json_object =  json.load(jsonfile)
                jsonfile.close()
                for i in range(len(json_object)):
                    p = json_object[str(i)]
                    if condition:
                        if p['y'] > 1:
                            single_point = (p['x'] , p['y'] , 1)
                        else:
                            single_point = (p['x'], p['y'], -1)
                        self.points.append(single_point)




def train(DataSet:dict):

    return



if __name__ == '__main__':
    data = Adaline("dataSets/dataSet1" , True)