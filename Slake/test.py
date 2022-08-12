# _*_ coding: utf-8 _*_

"""
    @Time : 2021/9/26 10:59
    @Author : smile 笑
    @File : test.py
    @desc :
"""


import torch
from main import test_configure, Sort2Id
import os
from model import VILT
from torch import nn
from dataset import get_dataloader
from tqdm import tqdm
import numpy as np
from train import compute_batch_score
from word_sequence import Word2Sequence, SaveWord2Vec
import pickle


os.environ["CUDA_VSIABLE_DEVICES"] = "0"


def test(configure, model, model_path):
    torch.cuda.set_device(0)

    model = nn.DataParallel(model, device_ids=configure["config"]["device_ids"]).cuda()
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)

    model.eval()

    db = get_dataloader(test_configure)
    bar = tqdm(db, total=len(db))

    ans_losses = []

    close_acces = 0
    open_acces = 0
    close_nums = 0
    open_nums = 0
    total_acces = 0
    total_nums = 0

    for idx, (image, qus, ans, location, ans_type) in enumerate(bar):
        predict_ans = model(image, qus)
        ans_loss = criterion(predict_ans, ans.view(-1))

        _, ans_pred = predict_ans.max(-1)  # 取出预测值

        # 计算一个batch的open和close准确率
        open_batch_acc, close_batch_acc, total_batch_acc, open_len, close_len, total_len = compute_batch_score(ans_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        open_acces += open_batch_acc
        open_nums += open_len
        close_acces += close_batch_acc
        close_nums += close_len
        total_acces += total_batch_acc
        total_nums += total_len

        ans_losses.append(ans_loss.cpu().item())

        open_acc = open_acces / (open_nums + 1e-10)
        close_acc = close_acces / (close_nums + 1e-10)  # 加一个足够小的数防止出现0
        total_acc = total_acces / (total_nums + 1e-10)

        bar.set_description("test_idx:{}, loss:{:.5f}, m_loss:{:.5f}, open_acc:{:.5f}, close_acc:{:.5f}, total_acc:{:.5f}".format(idx, ans_loss.cpu().item(), np.mean(ans_losses), open_acc, close_acc, total_acc))


if __name__ == '__main__':
    model = VILT(test_configure)
    model_path = test_configure["config"]["test_best_model_path"]
    parameter_path = test_configure["config"]["test_best_parameter_path"]
    device_ids = test_configure["config"]["device_ids"]

    test(test_configure, model, model_path)





