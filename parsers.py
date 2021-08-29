import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from loss import *

# 将str转换为bool值
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 获取实验参数
def get_args():
    ## 1、创建参数解析器
    parser = argparse.ArgumentParser()
    ## 2、添加参数
    # dir
    parser.add_argument('-save_dir', type=str, default='DNNs/')                     # 实验模型和结果的保存路径
    parser.add_argument('-wav1_dev', type=str, default='DB/VoxCeleb1/wav_dev/')      #  即vox1的dev set用作train
    parser.add_argument('-wav1_test', type=str, default='DB/VoxCeleb1/wav_test/')    #  即vox1的test集用作test
    parser.add_argument('-wav1_emb', type=str, default='DB/VoxCeleb1/emb/')
    parser.add_argument('-face1_dev', type=str, default='DB/Vox1_Face/face_dev/')
    parser.add_argument('-face1_test', type=str, default='DB/Vox1_Face/face_test/')
    parser.add_argument('-face1_emb', type=str, default='DB/Vox1_Face/emb/')
    parser.add_argument('-wav2_dev', type=str, default='DB/VoxCeleb2/wav_dev/')      #  即vox2的dev set用作train
    parser.add_argument('-wav2_test', type=str, default='DB/VoxCeleb2/wav_test/')    #  即vox2的test集用作test
    parser.add_argument('-wav2_emb', type=str, default='DB/VoxCeleb2/emb/')
    parser.add_argument('-face2_dev', type=str, default='DB/Vox2_Face/face_dev/')
    parser.add_argument('-face2_test', type=str, default='DB/Vox2_Face/face_test/')
    parser.add_argument('-face2_emb', type=str, default='DB/Vox2_Face/emb/')
    # hyper-params
    parser.add_argument('-nb_D_embd_samples', type=int, default=80000)  # train_trial数量,然后会产生2倍的cat嵌入向量用于训练
    parser.add_argument('-nb_valid_trial', type=int, default=20000)     # valid_trial数量,然后会产生2倍的cat嵌入向量用于验证
    parser.add_argument('-nb_eval_trial', type=int, default=20000)      # eval_trial数量,然后会产生2倍的cat嵌入向量用于测试
    parser.add_argument('-epoch', type=int, default=70)                 # epoc大小
    parser.add_argument('-train_bs', type = int, default = 64)          # train batch size  越小反而越差
    parser.add_argument('-valid_bs', type = int, default = 32)          # valid batch size  越大反而越差
    parser.add_argument('-eval_bs', type=int, default = 100)            # test  batch size
    parser.add_argument('-lr', type = float, default = 0.001)           # 学习率
    parser.add_argument('-lr_decay', type=str, default='keras')         # 学习率衰减
    parser.add_argument('-weight_decay', type = float, default=0.0001)  # 权重衰减
    parser.add_argument('-optimizer', type = str, default = 'Adam')     # 优化器
    parser.add_argument('-nb_worker', type = int, default = 8)          # pytorch worker数量
    parser.add_argument('-seed', type = int, default = 1234)            # 随机种子
    parser.add_argument('-load_model_dir', type = str, default='DNNs/models/epoch20_0.6203.pt')
    parser.add_argument('-load_model_opt_dir', type = str, default='DNNs/models/best_optimizer_eval.pt')
    # DNN args
    parser.add_argument('-m_cat_emb_dim', type=int, default=1536)  # 人脸512+语音1024 = 1536
    parser.add_argument('-m_nb_fc1_node', type=int, default=2048)  # 第一个全连接层输出为2048，所以嵌入维度也为2048
    parser.add_argument('-m_nb_fc2_node', type=int, default=1024)  # 第二个全连接层输出为1024，所以嵌入维度也为1024
    parser.add_argument('-m_nb_classes',  type=int, default=1211)   # vox1數據集的類別
    # flag
    parser.add_argument('-amsgrad', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-make_val_trial', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-save_best_only', type=str2bool, nargs='?', const=True, default=False) # 是否保存最优模型
    parser.add_argument('-load_model', type=str2bool, nargs='?', const=True, default=False)  # 是否导入预训练模型
    parser.add_argument('-do_lr_decay', type=str2bool, nargs='?', const=True, default=True)  # 是否执行学习率衰减
    parser.add_argument('-mg', type=str2bool, nargs='?', const=True, default=False)          # 应该指的是是否有多个GPU ？
    parser.add_argument('-reproducible', type=str2bool, nargs='?', const=True, default=True) # 实验是否可重复进行（需设定一个固定的随机种子）
    ## 3、解析参数
    args = parser.parse_args()
    # 将DNN args参数专门存到Dict，后面需要用
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':          # 找到以m_开头的
            args.model[k[2:]] = v  # 跳过前两个字母
    return args  # 返回一个字典






