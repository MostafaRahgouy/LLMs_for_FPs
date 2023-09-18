import json


class DataLoader:
    def __init__(self, path, dataset_name):
        self.path = path
        self.dataset_name = dataset_name

    def get_train(self):
        path = f'{self.path}/{self.dataset_name}/train.json'
        return self.read(path)

    def get_val(self):
        path = f'{self.path}/{self.dataset_name}/val.json'
        return self.read(path)

    def get_test(self):
        path = f'{self.path}/{self.dataset_name}/test.json'
        return self.read(path)

    @staticmethod
    def read(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def write(path, data):
        formatter = {"indent": 4, "separators": (",", ": ")}
        with open(path, 'w') as f:
            json.dump(data, f, **formatter)


def read_data(path):
    with open(path, encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def write(path, data):
    formatter = {"indent": 4, "separators": (",", ": ")}
    with open(path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, **formatter)
