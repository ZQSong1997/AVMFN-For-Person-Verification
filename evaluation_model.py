####    该文件用于评估预训练模型的性能！！！ ####
import argparse
import numpy as np
import pickle as pk
import torch
import torch.nn as nn
from torch.utils import data

from parsers import get_args
from dataloader import Val_Embd_Dataset
from trainer import evaluate_model
from utils import *

def eval():
    # 参数解析
    args = get_args()
    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # print('Device: {}'.format(device))
    # 导入test set生成的embedding dataset
    face_test_emb_npy, face_test_path_list = load_dataset(args.face1_emb + 'vox1_face_T_embedding.npy',
                                                          args.face1_emb + 'vox1_face_T_label.txt')
    with open(args.wav1_emb + 'TTA_vox1_eval.pk', 'rb') as f:  # .pk由原始wav数据集经过Rawnet2模型生成的embedding数据集
        wav_test_emb_pk = pk.load(f)            # 长度为4874    test_set embedding的数量
    # Vox1 face测试集
    face_test_emb_dict = {}
    for i in range(len(face_test_emb_npy)):
        face_test_emb_dict[face_test_path_list[i]] = list(face_test_emb_npy[i, 0, :])
    # Vox1 wav测试集
    wav_test_emb_dict = {}
    for key, value in wav_test_emb_pk.items():
        wav_test_emb_dict[str(key) + '.wav'] = list(value)
    # 制作“id与paths一对多”的NewDict，keys是说话人的id，values是说话人所有wav或face文件的paths
    Wav_Test_Id_Dict, Face_Test_Id_Dict = Id_to_mPath_Dict(wav_path_list=list(wav_test_emb_dict.keys()),
                                                           face_path_list=list(face_test_emb_dict.keys()))
    # 自定义一个dataset generators，用于生成 cat_embeds, cat_labels，cat_paths
    cat_v_embeds, cat_v_paths = Val_Cat_Emb(wav_id_dict=Wav_Test_Id_Dict, face_id_dict=Face_Test_Id_Dict,
                                            wav_emb_dict=wav_test_emb_dict, face_emb_dict=face_test_emb_dict,
                                            nb_trial=args.nb_eval_trial, dir_val_trial='trials/test_trials.txt')
    # 加载预训练模型
    model = torch.load('Pre-trained_model/best_model.pkl')

##  3、计算 Vox1 original evaluation EER    包含40 speakers 和 37720 trials （由4874个utterances自由组合）
    f_eer = open('trials/Evaluation_EERs.txt', 'a', buffering=1)
    # 用官方提供的trials 测试文件（权威）进行评估，自己训练的时候是要自己随机生成的（仅train用）。其实原理都是一样的。
    with open('trials/test_trials.txt', 'r') as f:  # trial pairs为40000
        val_trial_txt = f.readlines()
    # with open('../trials/vox_original.txt', 'r') as f:   # 相当于将Vox1的test集，随机组合成trials pair后作为Vox2的evaluation集
    #     vox1_original_test_trial = f.readlines()
    # 用自定义的数据生成器处理embedding数据集并导入，用作Evaluation
    # 生成喂入model的dataset(Face+Wav)，得到cat_embedding X 和 label y
    F_W_eval_set = Val_Embd_Dataset(cat_embeds=cat_v_embeds)
    # 调用DataLoader类进行封装，以便进行迭代
    valset_gen = data.DataLoader(F_W_eval_set,batch_size=args.eval_bs,shuffle=False,drop_last=False,num_workers=args.nb_worker)
    # 将封装好的生成数据送入evaluation模型进行评估，得到一个数值
    eval_eer = evaluate_model(model = model,
                              evalset_gen = valset_gen,  # 导入经处理后的test数据集 # 包含嵌入值 X ，但不包括label Y
                              eval_trial_txt = val_trial_txt, # 导入trial，其中 1 表示Positive sample pair，0 表示Negative sample pair
                              cat_paths = cat_v_paths,   # 根据自定义Cat向量数据集的wav+jpg路径生成的cat_path list
                              save_dir = args.save_dir,  # DNNs/
                              epoch = 1,
                              device = device)
    # 记录所得的EER分数
    text = 'Vox1 Evaluation EER: {}'.format(eval_eer)
    print(text)
    f_eer.write(text+'\n')
    f_eer.close()
    return eval_eer


num = 20
all_eval_eer = 0
print('begin evaluation')
for i in range(num):
    eval_eer = eval()
    all_eval_eer += eval_eer
Average_Eval_EER =  all_eval_eer / num
print('Average_Eval_EER：{}'.format(Average_Eval_EER))
