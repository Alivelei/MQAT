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
    xm_path = configure["dataset"]["dataset_xm_path"]
    queries = json.load(open(configure["dataset"]["dataset_path"], encoding="utf-8"))
    img_path = [os.path.join(xm_path + str(query["img_id"]), "source.jpg") for query in queries]
    compute_norm(img_path)


class Sort2Id(object):
    UNK_TAG = 0

    def __init__(self, configure):
        self.configure = configure
        self.sort_dict = {"UNK": 0}
        self.inverse_sort_dict = dict()

    def building_classify(self):
        queries = json.load(open(self.configure["dataset"]["dataset_path"], encoding="utf-8"))
        locations = set([query["location"] for query in queries])

        for location in locations:
            self.sort_dict[location] = len(self.sort_dict)
        print(self.sort_dict)
        self.inverse_sort_dict = dict(zip(self.sort_dict.values(), self.sort_dict.keys()))

    def sort_to_id(self, location):
        return self.sort_dict.get(location, 0)

    def id_to_sort(self, location_id):
        return self.inverse_sort_dict.get(location_id, 0)


parser = argparse.ArgumentParser()
parser.add_argument("--train_config_path", default="./config/train.toml")
parser.add_argument("--test_config_path", default="./config/test.toml")
parser.add_argument("--valid_config_path", default="./config/valid.toml")

args = parser.parse_args()

train_configure = toml.load(args.train_config_path)
test_configure = toml.load(args.test_config_path)
valid_configure = toml.load(args.valid_config_path)


if __name__ == '__main__':
    # compute_sort_number(train_configure)
    sortid = Sort2Id(train_configure)
    sortid.building_classify()
    # pickle.dump(sortid, open(train_configure["config"]["sort_ws_path"], "wb"))
    pass


