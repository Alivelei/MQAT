# _*_ coding: utf-8 _*_
"""
    @Time : 2021/9/29 14:22 
    @Author : smile 笑
    @File : main.py
    @desc :
"""


import toml
import argparse
# from network.compute_rgb import compute_norm
import json
import os
import pickle


def compute_img_norm(configure):
    xm_path = configure["dataset"]["dataset_path"]
    queries = os.listdir(xm_path)
    img_path = [os.path.join(xm_path, query) for query in queries]
    compute_norm(img_path)


class Sort2Id(object):
    def __init__(self, configure):
        self.configure = configure
        self.sort_dict = {}
        self.inverse_sort_dict = dict()

    def building_classify(self):
        queries = json.load(open(self.configure["dataset"]["dataset_json_path"], encoding="utf-8"))
        locations = set([query["category"] for query in queries])

        for location in locations:
            self.sort_dict[location] = len(self.sort_dict)
        print(self.sort_dict)  # {'organ': 0, 'modality': 1, 'plane': 2, 'abnormality': 3}
        self.inverse_sort_dict = dict(zip(self.sort_dict.values(), self.sort_dict.keys()))

    def sort_to_id(self, location):
        return self.sort_dict.get(location)

    def id_to_sort(self, location_id):
        return self.inverse_sort_dict.get(location_id)


parser = argparse.ArgumentParser()
parser.add_argument("--train_config_path", default="./config/train.toml")
parser.add_argument("--test_config_path", default="./config/test.toml")

args = parser.parse_args()

train_configure = toml.load(args.train_config_path)
test_configure = toml.load(args.test_config_path)


if __name__ == '__main__':
    # compute_sort_number(train_configure)
    sortid = Sort2Id(train_configure)
    sortid.building_classify()
    pickle.dump(sortid, open(train_configure["config"]["sort_ws_path"], "wb"))
    # compute_img_norm(train_configure)


