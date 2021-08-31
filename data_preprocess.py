import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pk

from parsers import *
from utils import *
from dataloader import *
from trainer import evaluate_model


######      数据导入和预处理模块      ######
args = get_args()
##  【1】、获取说话人的face和wav的embedding数据集和path label
#   1、导入并处理face和wav embeddings数据集和path list
# 导入face数据集   # face_emb_npy为原始face数据集经过FaceNet模型生成的embedding数据集  # face_path_list里面是与embeddings一一对应的path
face_dev_emb_npy, face_dev_path_list = load_dataset(args.face1_emb + 'vox1_face_D_embedding.npy',
                                                    args.face1_emb + 'vox1_face_D_label.txt')
face_test_emb_npy, face_test_path_list = load_dataset(args.face1_emb + 'vox1_face_T_embedding.npy',
                                                    args.face1_emb + 'vox1_face_T_label.txt')
# 导入wav数据集    # .pk由原始wav数据集经过Rawnet2模型生成的embedding数据集
with open(args.wav1_emb + 'TTA_vox1_dev.pk', 'rb') as f:
    wav_dev_emb_pk  = pk.load(f)                       # 长度为148642  dev_set embedding的数量，即说话人utterances的数量
with open(args.wav1_emb + 'TTA_vox1_eval.pk', 'rb') as f:
    wav_test_emb_pk = pk.load(f)                       # 长度为4874    test_set embedding的数量
# 其中wav_dev_emb_pk和wav_test_emb_pk 都为<class 'dict'>
#   2、转换为“path与embedding一一对应的Dict”
# face训练集
face_dev_emb_dict = {}
for i in range(len(face_dev_emb_npy)):
    face_dev_emb_dict[face_dev_path_list[i]] = list(face_dev_emb_npy[i, 0, :])
# face测试集
face_test_emb_dict = {}
for i in range(len(face_test_emb_npy)):
    face_test_emb_dict[face_test_path_list[i]] = list(face_test_emb_npy[i, 0, :])
# wav训练集
wav_dev_emb_dict = {}
for key, value in wav_dev_emb_pk.items():
    wav_dev_emb_dict[str(key) + '.wav'] = list(value)  # 需要转换成list才能得到全部内容，目的是方便查看完整的embedding数据
    # key的内容示例： 'id10001/1zcIwhmdeo4/00001'   # value为1024维的向量
    # print('wav_dev_emb_dict的keys为：{}'.format(wav_dev_emb_dict.keys()))  # [...,'id10426/UUpSDo9_Qi4/00025.wav', 'id10426/WIbIl2skqjk/00001.wav',...]
# wav测试集
wav_test_emb_dict = {}
for key, value in wav_test_emb_pk.items():
    wav_test_emb_dict[str(key) + '.wav'] = list(value)
# 验证一下数据是否匹配
print('face_dev_dict的emb数量为：{}'.format(len(face_dev_emb_dict)))    # 149519
print('face_test_dict的emb数量为：{}'.format(len(face_test_emb_dict)))  # 4826
print('wav_dev_dict的emb数量为：{}'.format(len(wav_dev_emb_dict)))    #  148642
print('wav_test_dict的emb数量为：{}\n'.format(len(wav_test_emb_dict)))  #  4874
#   3、生成 “vox1_dev_set” 的label，用于train, 其中speakers的id作为keys, speakers的class为values
Vox_dev_label_dict = get_label_dic(list(wav_dev_emb_dict.keys()))   # 返回一个字典，如 {'id10001':0,'id10002':1, ... , 'id11251':1250}
print('The train set have {} classes'.format( len(list(Vox_dev_label_dict.keys()))))    # 1211
#   4、制作“id与paths一对多”的New Dict，keys是说话人的id，values是说话人所有wav或face文件的paths
Wav_Dev_Id_Dict, Face_Dev_Id_Dict = Id_to_mPath_Dict(wav_path_list=list(wav_dev_emb_dict.keys()),
                                                     face_path_list=list(face_dev_emb_dict.keys()))
Wav_Test_Id_Dict, Face_Test_Id_Dict = Id_to_mPath_Dict(wav_path_list=list(wav_test_emb_dict.keys()),
                                                       face_path_list=list(face_test_emb_dict.keys()))
##  【2】、定义dataset generators
#   1、自定义一个dataset generators，用于生成 cat_embeds, cat_labels，cat_paths
cat_d_embeds, cat_d_paths, cat_d_labels = Dev_Cat_Emb(wav_id_dict=Wav_Dev_Id_Dict, face_id_dict=Face_Dev_Id_Dict,
                                                  wav_emb_dict=wav_dev_emb_dict, face_emb_dict=face_dev_emb_dict,
                                                  embd_samples=args.nb_D_embd_samples, labels_dict=Vox_dev_label_dict)
cat_t_embeds, cat_t_paths = Val_Cat_Emb(wav_id_dict=Wav_Test_Id_Dict, face_id_dict=Face_Test_Id_Dict,
                                         wav_emb_dict=wav_test_emb_dict, face_emb_dict=face_test_emb_dict,
                                         nb_trial=args.nb_valid_trial, dir_val_trial='trials/veri_test.txt')
with open('trials/veri_test.txt', 'r') as f:
    eval_trial_txt = f.readlines()
#   2、生成喂入model的dataset(Face+Wav)，得到cat_embedding X 和 label y    # 继承Dataset类
F_W_dev_set = Dev_Embd_Dataset(cat_embeds=cat_d_embeds, cat_labels=cat_d_labels)
F_W_eval_set = Val_Embd_Dataset(cat_embeds=cat_t_embeds)
#   3、利用DataLoader类进行封装，以便进行迭代
devset_gen = data.DataLoader(F_W_dev_set, batch_size=args.train_bs, shuffle=True, drop_last=True, num_workers=args.nb_worker)
evalset_gen = data.DataLoader(F_W_eval_set, batch_size=args.valid_bs, shuffle=False, drop_last=False, num_workers=args.nb_worker) # batch_size有改動
print('devset_gen长度为：{}'.format(len(devset_gen)))  # 如果数据样本总量一共10W，有100个batch，那每个batch就是1000
print('evalset_gen长度为：{}'.format(len(evalset_gen)))  # 如果数据样本总量一共4W，但只有1个batch，那每个batch就是40000

