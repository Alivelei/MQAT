# _*_ coding: utf-8 _*_

"""
    @Time : 2021/9/23 9:29 
    @Author : smile 笑
    @File : datasets_text.py
    @desc :
"""


import pickle
from torch.utils.data import DataLoader, Dataset
from word_sequence import Word2Sequence, SaveWord2Vec
import torchvision.transforms as tfs
import os
from PIL import Image
import json
import torch
import numpy as np
import toml
import argparse
from main import train_configure, test_configure, Sort2Id
from word_sequence import sentence_to_word


def train_aug_img(img, configure):
    aug = tfs.Compose([
        tfs.RandomResizedCrop(configure["image"]["img_height"], scale=(configure["image"]["resized_crop_left"], configure["image"]["resized_crop_right"])),
        tfs.RandomApply([tfs.GaussianBlur(kernel_size=configure["image"]["b_size"], sigma=configure["image"]["blur"])], p=configure["image"]["blur_p"]),
        tfs.RandomGrayscale(p=configure["image"]["grayscale"]),
        tfs.RandomApply([
            tfs.ColorJitter(configure["image"]["brightness"], configure["image"]["contrast"], configure["image"]["saturation"], configure["image"]["hue"])],
            p=configure["image"]["apply_p"]
        ),
        tfs.RandomRotation(configure["image"]["img_rotation"]),
        tfs.RandomHorizontalFlip(configure["image"]["img_flip"]),
        tfs.ToTensor(),
        tfs.Normalize(configure["image"]["img_mean"], configure["image"]["img_std"])
    ])

    return aug(img)


def test_aug_img(img, configure):
    aug = tfs.Compose([
        tfs.Resize([configure["image"]["img_height"], configure["image"]["img_width"]]),
        tfs.ToTensor(),
        tfs.Normalize(configure["image"]["img_mean"], configure["image"]["img_std"])
    ])

    return aug(img)


class VQADataset(Dataset):
    def __init__(self, configure):
        self.configure = configure
        self.images_path = configure["dataset"]["images_path"]
        self.run_mode = configure["run_mode"]  # 得到运行方式

        self.queries = json.load(open(configure["dataset"]["dataset_path"], encoding="utf-8"))
        self.queries = [query for query in self.queries]  # 3064、451
        self.qus_ws = pickle.load(open(configure["config"]["en_qus_ws_path"], "rb"))
        self.ans_ws = pickle.load(open(configure["config"]["en_ans_ws_path"], "rb"))
        self.max_seq_len = self.configure["config"]["en_qus_seq_len"]

        self.sort_ws = pickle.load(open(configure["config"]["sort_ws_path"], "rb"))

    def __getitem__(self, idx):
        query = self.queries[idx]  # 随机抽取一个

        img_path = os.path.join(self.images_path + str(query["image_name"]))

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(query["answer"], False)

        if self.run_mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.configure)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.configure)

        location = query["image_organ"]
        location_id = self.sort_ws.sort_to_id(location)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            ans_type_id = self.configure["config"]["answer_open"]
        else:
            ans_type_id = self.configure["config"]["answer_close"]

        qus_id = self.qus_ws.transform(question, max_len=self.max_seq_len)
        ans_id = self.ans_ws.transform([answer])

        return image, qus_id, ans_id, location_id, ans_type_id

    def __len__(self):
        return len(self.queries)


def collate_fn(batch):
    image, qus_id, ans_id, location_id, ans_type_id = list(zip(*batch))
    image = torch.stack(image).cuda(train_configure["config"]["device_ids"][0])
    qus_id = torch.tensor(qus_id, dtype=torch.int64).cuda(train_configure["config"]["device_ids"][0])
    ans_id = torch.tensor(ans_id, dtype=torch.int64).cuda(train_configure["config"]["device_ids"][0])
    location_id = torch.tensor(location_id, dtype=torch.int64).cuda(train_configure["config"]["device_ids"][0])
    ans_type_id = torch.tensor(ans_type_id, dtype=torch.int64).cuda(train_configure["config"]["device_ids"][0])

    return image, qus_id, ans_id, location_id, ans_type_id


def get_dataloader(configure):
    db = VQADataset(configure)
    if configure["config"]["num_workers"] >= 1:
        dl = DataLoader(db, batch_size=configure["config"]["batch_size"], shuffle=configure["config"]["shuffle"],
                        collate_fn=collate_fn, num_workers=configure["config"]["num_workers"], prefetch_factor=True, drop_last=True)
    else:
        dl = DataLoader(db, batch_size=configure["config"]["batch_size"], shuffle=configure["config"]["shuffle"],
                        collate_fn=collate_fn, drop_last=True)

    return dl


if __name__ == '__main__':
    for idx, (img, qus, ans, location, ans_type) in enumerate(get_dataloader(train_configure)):
        print(img.shape)
        print(qus)
        print(ans)
        print(location)
        print(ans_type)
        break





