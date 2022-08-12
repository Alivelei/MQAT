# _*_ coding: utf-8 _*_

"""
    @Time : 2021/12/10 14:50
    @Author : smile 笑
    @File : patch_embedding.py
    @desc :
"""


from torch import nn
import torch
from network.trans_encoder import TransformerEncoder, ClassificationHead
from einops.layers.torch import Rearrange, Reduce
from network.res_block import BasicBlock
from network.res_block import BasicBlock, conv3x3
from main import train_configure
from network.word_embedding import WordEmbedding


class ImgPatchEmbedding(nn.Module):
    def __init__(self, emb_size=784, seq_len=49, ngf_conv=64):
        super(ImgPatchEmbedding, self).__init__()
        self.feature_extractor = nn.Sequential(
            BasicBlock(3, ngf_conv, 2, downsample=conv3x3(3, ngf_conv, 2)),
            BasicBlock(ngf_conv, ngf_conv * 2, 2, downsample=conv3x3(ngf_conv, ngf_conv * 2, 2)),
            BasicBlock(ngf_conv * 2, ngf_conv * 2, 2, downsample=conv3x3(ngf_conv * 2, ngf_conv * 2, 2)),
            BasicBlock(ngf_conv * 2, ngf_conv * 2, 2, downsample=conv3x3(ngf_conv * 2, ngf_conv * 2, 2)),
            BasicBlock(ngf_conv * 2, emb_size, 2, downsample=conv3x3(ngf_conv * 2, emb_size, 2)),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.positions = nn.Parameter(torch.randn(seq_len, emb_size))  # 加入位置编码

    def forward(self, x):
        x = self.feature_extractor(x)  # torch.Size([2, 49, 768, 768])

        x = x + self.positions  # 加入位置编码
        x = x + 0.3  # 为图像加入图像属性编码 0.5，表示是图像部分

        return x


class TextPatchEmbedding(nn.Module):
    def __init__(self, word_size, embedding_size, hidden_size, seq_len, glove_path):
        super(TextPatchEmbedding, self).__init__()

        self.embedding = WordEmbedding(word_size, embedding_size, 0.0, False)
        self.embedding.init_embedding(glove_path)
        self.linear = nn.Linear(embedding_size, hidden_size)
        # 位置编码信息，一共有20个位置向量
        self.positions = nn.Parameter(torch.randn(seq_len, hidden_size))

    def forward(self, qus):
        text_embedding = self.embedding(qus)

        text_x = self.linear(text_embedding)

        text_x = text_x + self.positions  # 文本加入位置编码

        text_x = text_x - 0.3  # 文本加入类别编码

        return text_x


class VILT(nn.Module):
    def __init__(self, configure):
        super(VILT, self).__init__()

        qus_word_size = configure["config"]["en_qus_word_size"]  # 270
        n_classes = configure["config"]["en_ans_word_size"]
        emb_size = configure["model"]["emb_size"]
        qus_embedding_dim = configure["model"]["qus_embedding_dim"]
        text_seq_len = configure["config"]["en_qus_seq_len"]
        depth = configure["model"]["depth"]
        num_heads = configure["model"]["num_heads"]
        dropout = configure["model"]["dropout"]
        seq_len = configure["model"]["seq_len"]
        ngf_conv = configure["model"]["ngf_conv"]
        glove_path = configure["config"]["glove_path"]

        self.img_model = ImgPatchEmbedding(emb_size=emb_size, seq_len=seq_len, ngf_conv=ngf_conv)
        self.text_projector = TextPatchEmbedding(qus_word_size, qus_embedding_dim, emb_size, text_seq_len, glove_path)
        self.trans_enc = TransformerEncoder(depth, num_heads=num_heads, drop_p=dropout, forward_drop_p=dropout)
        self.cls_linear = ClassificationHead(emb_size, n_classes)

    def forward(self, img, qus):
        img_patch_features = self.img_model(img)  # torch.Size([2, 49, 768])
        qus_embedding = self.text_projector(qus)  # torch.Size([2, 20, 768])
        batch_size, _, dim = qus_embedding.shape

        sort_tokens = torch.zeros([batch_size, 1, dim], requires_grad=False).cuda()  # 加入类别区分  并且不对其训练

        patch_embedding = torch.cat([qus_embedding, sort_tokens], dim=1)
        patch_embedding = torch.cat([patch_embedding, img_patch_features], dim=1)

        res = self.cls_linear(self.trans_enc(patch_embedding))

        return res


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.ones([2, 20], dtype=torch.int64).cuda()
    vit = VILT(train_configure).cuda()
    print(vit(a, b).shape)
    # torch.save(vit.state_dict(), "./1.pth")  # 360M

