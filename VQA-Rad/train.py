# _*_ coding: utf-8 _*_

"""
    @Time : 2021/9/20 9:15
    @Author : smile 笑
    @File : train2.py
    @desc :
"""


from torch import nn, optim
import torch
from word_sequence import SaveWord2Vec, Word2Sequence
from dataset import get_dataloader
from tqdm import tqdm
import os
import numpy as np
from tensorboardX import SummaryWriter
from model import VILT
from main import train_configure, test_configure, Sort2Id
from network.mixup import image_mix_up
from network.method import set_seed_everything, load_pretrained_model, compute_batch_score
from epoch_res import get_epoch_min_mode, save_epoch_res


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, device_ids, optimizer, epochs, mix_probability, start_epoch_save, save_frequency, latest_model_path=None, latest_parameter_path=None, log_path=None):
    torch.cuda.set_device(0)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if not os.path.exists(model_path):
        start_epoch = 0
    else:
        model.load_state_dict(torch.load(latest_model_path))
        parameter = torch.load(latest_parameter_path)
        optimizer.load_state_dict(parameter["optimizer"])
        start_epoch = parameter["epoch"] + 1

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_path)

    for epoch in range(start_epoch, epochs):
        db = get_dataloader(train_configure)
        bar = tqdm(db, total=len(db))

        ans_losses = []

        close_acces = 0
        open_acces = 0
        close_nums = 0
        open_nums = 0
        total_acces = 0
        total_nums = 0

        model.train()
        for idx, (image, qus, ans, location, ans_type) in enumerate(bar):
            # 使用mix_up数据增强
            if np.random.random() <= mix_probability:
                ans_loss, predict_ans = image_mix_up(model, image, qus, ans, criterion)
            else:
                predict_ans = model(image, qus)
                ans_loss = criterion(predict_ans, ans.view(-1))

            _, ans_pred = predict_ans.max(-1)  # 取出预测值

            # 计算一个batch的open和close准确率
            open_batch_acc, close_batch_acc, total_batch_acc, open_len, close_len, total_len = compute_batch_score(ans_pred, ans.view(-1), ans_type)

            # 3.backward
            optimizer.zero_grad()  # reset gradient
            ans_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.025)  # 设定阈值，防止梯度消失、梯度爆炸
            optimizer.step()  # update parameters of net

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

            bar.set_description("train_e:{}, loss:{:.5f}, m_loss:{:.5f}, open_acc:{:.5f}, close_acc:{:.5f}, total_acc:{:.5f}".format(epoch, ans_loss.cpu().item(), np.mean(ans_losses), open_acc, close_acc, total_acc))

        # 将每个epoch的loss添加到log日志中
        writer.add_scalars("ans_loss", {"ans_loss": np.mean(ans_losses)}, epoch)
        writer.add_scalars("open_acc", {"open_acc": np.mean(open_acc)}, epoch)
        writer.add_scalars("close_acc", {"close_acc": np.mean(close_acc)}, epoch)
        writer.add_scalars("total_acc", {"total_acc": np.mean(total_acc)}, epoch)

        # 每五轮保存一次模型，保证训练断掉后，还能接上
        if epoch % 5 == 0:
            state_dict = {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "ans_loss": np.mean(ans_losses),
                "total_acc": total_acc,
                "open_acc": open_acc,
                "close_acc": close_acc
            }
            torch.save(model.state_dict(), latest_model_path)
            torch.save(state_dict, latest_parameter_path)

        # 将每轮模型的结果保存在json中
        state_dict = {"train_e": epoch, "m_loss": np.mean(ans_losses), "open_acc": open_acc, "close_acc": close_acc,
                      "total_acc": total_acc}
        save_epoch_res(train_configure["config"]["train_epoch_effect_path"], state_dict)

        # 当大于250轮时，保存最佳模型
        if epoch > start_epoch_save:
            ans_loss = np.mean(ans_losses)
            min_m_loss = get_epoch_min_mode(train_configure["config"]["train_epoch_effect_path"], "loss")
            if ans_loss < min_m_loss:
                # 保存最佳模型  当模型跑了250轮以上后保存loss最小模型
                state_dict = {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "ans_loss": ans_loss,
                    "total_acc": total_acc,
                    "open_acc": open_acc,
                    "close_acc": close_acc
                }
                torch.save(model.state_dict(), train_configure["config"]["best_model_path"])
                torch.save(state_dict, train_configure["config"]["best_parameter_path"])

        if epoch >= start_epoch_save and epoch % save_frequency == 0:
            # 大于250个epoch每15轮保存一个模型
            m_loss = np.mean(ans_losses)
            torch.save(model.state_dict(), os.path.join("./model/", "model_e{}_l{:.3f}.pth".format(epoch, m_loss)))

        # 进入验证集
        if epoch % 5 == 0:
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

            # 将每轮模型的结果保存在json中
            state_dict = {"test_e": epoch, "m_loss": np.mean(ans_losses), "open_acc": open_acc, "close_acc": close_acc, "total_acc": total_acc}
            save_epoch_res(train_configure["config"]["test_epoch_effect_path"], state_dict)

            if epoch > start_epoch_save:
                max_total_acc = get_epoch_min_mode(train_configure["config"]["test_epoch_effect_path"], "acc")
                if total_acc >= max_total_acc:
                    # 保存最佳模型  当模型跑了250轮以上后保存测试集准确率最大模型
                    state_dict = {
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "ans_loss": np.mean(ans_losses),
                        "total_acc": total_acc,
                        "open_acc": open_acc,
                        "close_acc": close_acc
                    }
                    torch.save(model.state_dict(), train_configure["config"]["test_best_model_path"])
                    torch.save(state_dict, train_configure["config"]["test_best_parameter_path"])


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    set_seed_everything(train_configure["model"]["random_seed"])

    model = VILT(train_configure)
    # 对可梯度更新的进行更新
    optimizer = optim.AdamW(model.parameters(), lr=train_configure["model"]["learning_rate"], weight_decay=1e-4)
    device_ids = train_configure["config"]["device_ids"]

    model_path = train_configure["config"]["latest_model_path"]
    parameter_path = train_configure["config"]["latest_parameter_path"]

    epochs = train_configure["config"]["epochs"]
    log_path = train_configure["config"]["log_path"]

    mix_probability = train_configure["image"]["mix_up_probability"]

    start_epoch_save, save_frequency = train_configure["model"]["start_epoch_save"], train_configure["model"]["save_frequency"]

    train(model, device_ids, optimizer, epochs, mix_probability, start_epoch_save, save_frequency, model_path, parameter_path, log_path)


