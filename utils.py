import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import random

## 设置学习率衰减
def keras_lr_decay(step, decay = 0.0001):
    return 1./(1. + decay * step)

def init_weights(m):
    print(m)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        else:
            print('no weight',m)

def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_val_utts(l_val_trial):
    l_utt = []
    for line in l_val_trial:
        _, utt_a, utt_b = line.strip().split(' ')
        if utt_a not in l_utt: l_utt.append(utt_a)
        if utt_b not in l_utt: l_utt.append(utt_b)
    return l_utt

def read_data(filename, split=" "):
    with open(filename, mode="r", encoding='utf-8') as f:
        content_list = f.readlines()
        if split is None:
            content_list = [content.rstrip() for content in content_list]
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]
    return content_list
# 加载face embedding
def load_dataset(dataset_path, filename):
    embeddings = np.load(dataset_path)
    path_list = read_data(filename, split=None)  # read_data() 返回txt的数据列表
    return embeddings, path_list

## 遍历face数据集中的所有图片jpg，并将path作为标签存到一个列表中
def get_face_list(src_dir):
    l_face = []
    for path, dirs, files in os.walk(src_dir):    # src_dir = DB/Vox1_Face/dev_face/
        # 其中 path为：DB/Vox1_Face/dev_face/id10270/5r0dWxy17C8       # dirs为：[]      # files为：0000600.jpg
        base = '/'.join(path.split('/')[-2:]) + '/'  # 将path以'/'进行分割，并取最后两段;然后以'/'进行拼接，最后再加一个'/'。
        for file in files:
            if file[-3:] != 'jpg':       # file 为0000600.jpg、0000700.jpg
                continue                 # 如果文件不是jpg格式，则跳过该循环
            l_face.append(base + file)   # base 正常应该为：id10270/5r0dWxy17C8/   # file为：0000600.jpg
            # l_face内容为：['id10270/5r0dWxy17C8/0000600.jpg', 'id10270/5r0dWxy17C8/0000700.jpg', .....]
    return l_face
## 遍历wav数据集中的所有音频wav，并将path作为标签存到一个列表中
def get_utt_list(src_dir):
    l_utt = []
    for path, dirs, files in os.walk(src_dir):    # src_dir = DB/VoxCeleb1/test_wav/ 等
        # 其中 path为：DB/VoxCeleb1/test_wav/id10270/5r0dWxy17C8  # dirs为：[]  # files为：00001.wav
        base = '/'.join(path.split('/')[-2:]) + '/'  # 将path以'/'进行分割，并取最后两段;然后以'/'进行拼接，最后再加一个'/'。
        # print('当前base为：{}'.format(base))  # id10270/5r0dWxy17C8/
        for file in files:
            if file[-3:] != 'wav':       # file 为00001.wav、00002.wav
                continue                 # 如果文件不是wav格式，则跳过该循环
            l_utt.append(base+file)      # base 示例：id10270/5r0dWxy17C8/  # file为：00001.wav
            # l_utt内容为：['id10270/5r0dWxy17C8/00001.wav', 'id10270/5r0dWxy17C8/00002.wav', 'id10270/5r0dWxy17C8/00003.wav', .....]
    return l_utt
## 用于生成 “vox1_dev_set训练集” 的label，其中speakers的id作为键，index作为值
def get_label_dic(l_utt):    # l_utt是dev set中每个wav文件的path
    d_label = {}
    idx_counter = 0
    # 以speakers’id为keys，classes index为values生成一个dict
    for utt in l_utt:                    # utt示例：id10001/1zcIwhmdeo4/00001.wav
        spk = utt.split('/')[0]          # 将speakers id号作为说话人的label，如 id10001
        if spk not in d_label:           # 避免重复
            d_label[spk] = idx_counter   # 如 {'id10012':11}
            idx_counter += 1
    return d_label                       # 返回一个字典，如 {'id10001':0,'id10002':1, ... , 'id11251':1250}



## 该函数用于将生成的embedding数据统一转换成Dict格式，并且以speakers的id作为键keys，值values为一个二维list(当中每一个元素也是一个list，即该说话人的embedding)
# 如：{'id10001':[ [521维的向量],[521维的向量],...,[521维的向量] ], 'id10002':[ [521维的向量],[521维的向量],...,[521维的向量] ], ..., }
# Wav_SpkId_Dict, Face_SpkId_Dict = Embd_to_Dict(wav_emb_dict=dev_wav_emb_pk, face_emb_list=face_dev_embd, face_path_list=face_dev_path)
def Embd_to_Dict(wav_emb_dict, face_emb_list, face_path_list):
## 参数说明：
# wav_emb_dict 是Vox1_wav_set的embedding文件，以.pk格式保存，数据结构为Dict，其中keys为：'id10001/1zcIwhmdeo4/00001'，values为文件对应的embedding向量
# face_emb_list 是Vox1_face_set的embedding文件，数据结构为List格式
# face_path_list 是

    ## 【1】、先处理Vox1 wav的embedding文件得到一个新的Dict，目的是得到完整的embedding数据
    vox1_wav_dict = {}
    for key, value in wav_emb_dict.items():
        vox1_wav_dict[key] = list(value)  # 需要转换成list才能得到全部内容
        # key的内容示例： 'id10001/1zcIwhmdeo4/00001'   # value的维度为 1024
    ## 【2】、制作一个字典，keys是说话人的id，values是说话人所属的wav或face的embeddings
    # Wav dict
    Wav_SpkId_Dict = {}
    for path, emb in vox1_wav_dict.items():
        spk_id = path.split('/')[0]         # path示例：id10001/1zcIwhmdeo4/00001  spkid示例：id10001
        # 说话人不存在则添加，并且以id为键，添加一个 空list 便于后续存 embedding
        if spk_id not in Wav_SpkId_Dict:
            Wav_SpkId_Dict[spk_id] = []     # 如：{'id10001':[], 'id10002':[], ..., 'id11251':[] }
        # 将同一个说话人的不同wav文件的embedding 一一添加到同一个 空list 中，这里list是键的值。
        Wav_SpkId_Dict[spk_id].append(emb)  # 如：{'id10001':[[向量1],[向量2],[向量3],...,[向量n]],  ..., }  ，其中向量为1024维的embedding
    # Face dict
    Face_SpkId_Dict = {}
    for path, emb in zip(face_path_list, face_emb_list):
        face_id = path.rstrip().split('/')[0]  # path示例：id10001/1zcIwhmdeo4/0000600.jpg    face_id示例：id10270
        if face_id not in Face_SpkId_Dict:
            Face_SpkId_Dict[face_id] = []
        Face_SpkId_Dict[face_id].append(emb)
    return Wav_SpkId_Dict,Face_SpkId_Dict

## 制作一个新字典，keys是说话人的id，values是说话人所有wav或face文件的paths
def Id_to_mPath_Dict(wav_path_list, face_path_list):  # 示例：wav_dev_dict.keys(), face_dev_dict.keys()
    Wav_Id_mPath_Dict = {}
    for wav_path in wav_path_list:
        spk_id = wav_path.split('/')[0]       # wav_path示例：id10270/5r0dWxy17C8/00001.wav  spk_id示例：id10270
        # 说话人不存在则添加，并且以id为键，添加一个 空list 便于后续存 path
        if spk_id not in Wav_Id_mPath_Dict:
            Wav_Id_mPath_Dict[spk_id] = []      # 如：{'id10270':[], 'id10271':[], ..., 'id10309':[] }
        # 将同一个说话人的不同wav文件的path 一一添加到同一个 空list 中，这里list是键的值。
        Wav_Id_mPath_Dict[spk_id].append(wav_path)    # 如：{'id10270':[ [path1],[path2],[path3],...,[path n] ],  ..., }
    Face_Id_mPath_Dict = {}
    for face_path in face_path_list:
        face_id = face_path.split('/')[0]  # face_path示例：id10270/5r0dWxy17C8/0000600.jpg    face_id示例：id10270
        if face_id not in Face_Id_mPath_Dict:
            Face_Id_mPath_Dict[face_id] = []      # 如：{'id10270':[], 'id10271':[], ..., 'id10309':[] }
        # 将同一个说话人的不同jpg文件的path 一一添加到同一个list中，这里list是键的值。
        Face_Id_mPath_Dict[face_id].append(face_path)
    return Wav_Id_mPath_Dict, Face_Id_mPath_Dict


## 采用Vox1_test数据集，制作用于”训练中评估“的trial pairs，label 为 1 表示 positive sample pair，0 表示 negative sample pair
## 函数生成一个txt文件
def make_validation_trial(Wav_Id_mPath_Dict, Face_Id_mPath_Dict, nb_trial, dir_val_trial):
## 参数说明：
# Wav_Id_mPath_Dict, Face_Id_mPath_Dict 都是字典，keys是说话人的id，values是说话人所有wav或face文件的paths
# nb_trial 是trial pairs的数量，其中 1 表示 positive sample pair，0 表示 negative sample pair
# dir_val_trial 是保存trial pairs和label的文件，以txt文件格式保存，具体为 'trials/veri_test.txt'
    # 【1】、获取 speakers id list, 将字典的所有键，即speakers id取出来
    l_spk = list(Wav_Id_mPath_Dict.keys())  # 注意需转换成list  长度应该为40
    # print('speakers id为：{}'.format(l_spk))     # 具体为：['id10270', 'id10271', 'id10272', 'id10273',...,'id10309']
    f_val_trial = open(dir_val_trial, 'w')   # 将 trial pairs 保存为txt文件
    # 【2】、构造 positive trial pairs   比较同一个speaker的trial pair，标签取1
    # 将nb_trial的数量对半分, 即正样本对和负样本对各占一半, 我们设为40000
    nb_trg_trl = int(nb_trial / 2)
    # 首先随机选取l_spk列表中指定数量的speakers id。 其中replace为True表示可以重复选取；随机种子已经设为1234,以便每次实验选取结果相同
    selected_spks = np.random.choice(l_spk, size=nb_trg_trl, replace=True)   # selected_spks 元素数量为40000个,包含很多重复的id
    for spk in selected_spks:
        l_cur_wav = Wav_Id_mPath_Dict[spk]   # 然后根据该speaker id去查找键值，即一个存有该说话人所有wav path文件的list
        l_cur_face = Face_Id_mPath_Dict[spk] # 然后根据该speaker id去查找键值，即一个存有该说话人所有jpg path文件的list
        wav_a, wav_b = np.random.choice(l_cur_wav, size=2, replace=False)  # 接着又在该list中随机选取2个元素，作为a positive trial pair
        jpg_a, jpg_b = np.random.choice(l_cur_face, size=2, replace=False)
        f_val_trial.write('1 %s_%s %s_%s\n'%(wav_a,jpg_a,wav_b,jpg_b))        # 最后，写入txt文件，方便后面的程序使用
    # 【3】、构造 negative trial pairs  比较不同speaker的trial pair，标签取0
    # 选取同样数量的 trial pairs
    for i in range(nb_trg_trl):
        # 首先随机选取l_spk列表中 2 个speakers id。 并且数值不能相同，因为这里要构造 negative trial pairs
        spks_cur = np.random.choice(l_spk, size=2, replace=False)   # spks_cur 元素数量为 2 个
        # 分别根据这两个speaker id为键，查询得到他们的文件list，再各从文件list中随机选取1个元素，组成 a negative trial pair
        wav_a = np.random.choice(Wav_Id_mPath_Dict[spks_cur[0]], size=1)[0]
        wav_b = np.random.choice(Wav_Id_mPath_Dict[spks_cur[1]], size=1)[0]
        jpg_a = np.random.choice(Face_Id_mPath_Dict[spks_cur[0]], size=1)[0]
        jpg_b = np.random.choice(Face_Id_mPath_Dict[spks_cur[1]], size=1)[0]
        f_val_trial.write('0 %s_%s %s_%s\n'%(wav_a,jpg_a,wav_b,jpg_b))
    f_val_trial.close()
    return

## 用于调试
# l_eval = sorted(get_utt_list('DB/VoxCeleb1/vox1_wav_T_set/'))
# print(l_eval)
# make_validation_trial(l_utt=l_eval, nb_trial=40, dir_val_trial='DB/VoxCeleb1/veri_test.txt')
## 函数介绍：
# np.random.choice(a, size=None, replace=True, p=None)  用法:从给定的1维数组a中随机采样,注意必须是一维的！
# a : 如果是一维数组，就表示从这个一维数组中随机采样；如果是int型，就表示从0到a-1这个序列中随机采样。
# size : 采样结果的数量，默认为1。可以是整数，表示要采样的数量；也可以为tuple，如(m, n, k)，则要采样的数量为m * n * k，size为(m, n, k)。
# replace : boolean型，采样的样本是否要更换？ 意思是抽样之后还放不放回去。当replace指定为True时，采样的元素可能会有重复；当replace指定为False时，采样不会重复。
# p : 一个一维数组，制定了a中每个元素采样的概率，若为默认的None，则a中每个元素被采样的概率相同。


# dataset generators，用于生成 cat_embeds, cat_paths, cat_labels
def Dev_Cat_Emb(wav_id_dict, face_id_dict, wav_emb_dict, face_emb_dict, embd_samples, labels_dict):
## 参数说明：
# wav_id_dict, face_id_dict：是“id与paths一对多”的Dict，keys是说话人的id，values是说话人所有wav或face文件的paths
# wav_emb_dict, face_emb_dict：是“path与embedding一一对应”的Dict   # key的内容示例： 'id10001/1zcIwhmdeo4/00001.wav' ，value为1024维的向量
# labels_dict：是Vox1_dev_set的label，用于训练, 其中speakers的id作为keys, speakers的class为values，如 {'id10001':0,'id10002':1, ... , 'id11251':1250}
# embd_samples：指定输入模型的cat_embeds 的样本数量
# Return: cat_embeds, cat_paths, cat_labels
    # 先将字典的所有键，即speakers id取出来
    l_spk = list(wav_id_dict.keys())    # l_spk 具体为：['id10001', 'id10002', ...,'id11250']
    selected_spks = np.random.choice(l_spk, size=embd_samples, replace=True)  # 有很多重复的id, 如['id10156', 'id10762', ...,'id10542']
    cat_embeds, cat_paths, cat_labels = [], [], []
    # 构造cat embeddings
    for spk in selected_spks:    # spk是一个id号
        ## 1、根据speakers id去查找键值，即一个存有该说话人所有wav或face paths的list
        l_cur_wav = wav_id_dict[spk]
        l_cur_face = face_id_dict[spk]  # l_cur_wav和l_cur_face是该id的所有paths(多个)
        ## 2、从wav和face paths中随机选取其中一个path
        wav_path = random.choice(l_cur_wav)  # 一个path字符串 如 'id10001/1zcIwhmdeo4/00001.wav'
        face_path = random.choice(l_cur_face)
        ## 3、根据path为key去找到对应的embeddings value
        wav_emb = wav_emb_dict[wav_path]     # 一个1024维的向量
        face_emb = face_emb_dict[face_path]  # 一个512维的向量
        ## 4、进行向量拼接前，需对两个embedding进行L2-normalized
        wav_emb_re = np.array(wav_emb).reshape(1, -1)   # 需要先转换成二维数组
        face_emb_re = np.array(face_emb).reshape(1, -1)
        wav_emb_l2 = preprocessing.normalize(wav_emb_re, norm='l2')   # L2正则化后的数值范围为[-1,1]
        face_emb_l2 = preprocessing.normalize(face_emb_re, norm='l2')
        wav_emb_L2 = wav_emb_l2[0]  # 变回一维
        face_emb_L2 = face_emb_l2[0]
        # print('wav_emb_L2的length是：{}'.format(len(wav_emb_L2)))   # 1024  # <class 'numpy.ndarray'>
        # print('face_emb_L2的length是：{}'.format(len(face_emb_L2))) # 512
        ## 5、然后进行向量拼接，得到一个统一的嵌入 X
        Cat_emb = np.concatenate((wav_emb_L2, face_emb_L2))
        Cat_Emb = Cat_emb.astype(np.float32)    # 必须进行这一步，将数据转为float，不然进行forward时可能报错
        # print('拼接后的embedding的shape为：{}'.format(np.array(Cat_emb).shape))  # (1536,)  即1024 +512
        ## 6、生成cat向量 X 的cat_path
        Cat_Path = str(wav_path) + '__' + str(face_path)  # 以'__'为连接符，拼接wav path和jpg path
        ## 7、爲train模式，需產生labels
        Class_Label = labels_dict[spk]   # y 为标签 0-1250, labels 内容如 {'id10001':0,'id10002':1, ... , 'id11251':1250}
        ## 8、添加到list
        cat_embeds.append(Cat_Emb)
        cat_paths.append(Cat_Path)
        cat_labels.append(Class_Label)
        # print('拼接后路径为{}的向量{}的label为{}'.format(Cat_path,Cat_emb, Class_Label))
# 拼接后路径为id10003/EGPV-Xa0LGk/00001.wav__id10003/K5zRxtXc27s/0003200.jpg的向量[-0.03436462  0.00641493 -0.01543195 ... -0.00822573 -0.02959702 0.05503625]的label为2
# 拼接后路径为id10848/UsA4VLmofgo/00004.wav__id10848/yE10PUOtORM/0000700.jpg的向量[ 0.04604127  0.03042512  0.00607507 ... -0.03349253  0.01749844 -0.00830342]的label为807
    return cat_embeds, cat_paths, cat_labels

# Test dataset generators，用于生成 cat_embeds, cat_paths
def Val_Cat_Emb(wav_id_dict, face_id_dict, wav_emb_dict, face_emb_dict, nb_trial, dir_val_trial):
    ## 参数说明：
    # wav_id_dict, face_id_dict：是“id与paths一对多”的Dict，keys是说话人的id，values是说话人所有wav或face文件的paths
    # wav_emb_dict, face_emb_dict：是“path与embedding一一对应”的Dict   # key的内容示例： 'id10001/1zcIwhmdeo4/00001.wav' ，value为1024维的向量
    # nb_trial 是trial pairs的数量，其中 1 表示 positive sample pair，0 表示 negative sample pair
    # dir_val_trial 是保存trial pairs和label的文件，以txt文件格式保存，具体为 'trials/veri_test.txt'
    # Return: cat_embeds, cat_paths

##  【1】、获取 speakers id list, 将字典的所有键，即speakers id取出来
    l_spk = list(wav_id_dict.keys())  # 注意需转换成list 长度为40  # 具体为：['id10270', 'id10271', 'id10272', 'id10273',...,'id10309']
    nb_trg_trl = int(nb_trial / 2)    # 将nb_trial的数量20000 对半分, 即正样本对和负样本对各占一半
    cat_embeds, cat_paths = [], []
    f_val_trial = open(dir_val_trial, 'w')  # 将 trial pairs 保存为txt文件
##  【2】、构造positive trial pairs，比较同一个speaker的trial pair，标签取1。 这将会产生两倍数量的cat-embeddings喂入到网络进行学习，以得到一个预测值。
    # 首先随机选取l_spk列表中指定数量的speakers id。 其中replace为True表示可以重复选取；随机种子已经设为1234,以便每次实验选取结果相同。
    selected_spks = np.random.choice(l_spk, size=nb_trg_trl, replace=True)  # 有很多重复的id, 如['id10156', 'id10762', ...,'id10542']
    for spk in selected_spks:    # spk是一个id号
        ## 1、根据speakers id去查找键值，即一个存有该说话人所有wav或face paths的list
        l_cur_wav = wav_id_dict[spk]
        l_cur_face = face_id_dict[spk]  # l_cur_wav和l_cur_face是该id的所有paths(多个)
        ## 2、从wav和face paths中随机选取其中一个path
        wav_path_a, wav_path_b = np.random.choice(l_cur_wav, size=2, replace=False)  # 接着又在该list中随机选取2个元素，作为a positive trial pair
        face_path_a, face_path_b = np.random.choice(l_cur_face, size=2, replace=False)
        ## 3、根据path为key去找到对应的embeddings value
        wav_emb_a = wav_emb_dict[wav_path_a]        # 一个1024维的向量
        wav_emb_b = wav_emb_dict[wav_path_b]        # 一个1024维的向量
        face_emb_a = face_emb_dict[face_path_a]     # 一个512维的向量
        face_emb_b = face_emb_dict[face_path_b]     # 一个512维的向量
        ## 4、进行向量拼接前，需对两个embedding进行L2-normalized
        wav_emb_re_a = np.array(wav_emb_a).reshape(1, -1)   # 需要先转换成二维数组
        wav_emb_re_b = np.array(wav_emb_b).reshape(1, -1)
        face_emb_re_a = np.array(face_emb_a).reshape(1, -1)
        face_emb_re_b = np.array(face_emb_b).reshape(1, -1)
        wav_emb_l2_a = preprocessing.normalize(wav_emb_re_a, norm='l2')   # L2正则化后的数值范围为[-1,1]
        wav_emb_l2_b = preprocessing.normalize(wav_emb_re_b, norm='l2')
        face_emb_l2_a = preprocessing.normalize(face_emb_re_a, norm='l2')
        face_emb_l2_b = preprocessing.normalize(face_emb_re_b, norm='l2')
        wav_emb_L2_a = wav_emb_l2_a[0]  # 变回一维
        wav_emb_L2_b = wav_emb_l2_b[0]
        face_emb_L2_a = face_emb_l2_a[0]
        face_emb_L2_b = face_emb_l2_b[0]
        # print('wav_emb_L2的length是：{}'.format(len(wav_emb_L2_a)))    # 1024 # <class 'numpy.ndarray'>
        # print('face_emb_L2的length是：{}'.format(len(face_emb_L2_a)))  # 512
        ## 5、然后进行向量拼接，得到一个统一的嵌入 X
        Cat_emb_a = np.concatenate((wav_emb_L2_a, face_emb_L2_a))
        Cat_emb_b = np.concatenate((wav_emb_L2_b, face_emb_L2_b))
        # Cat_Emb_A = Cat_emb_a.astype(np.float32)  # 必须进行这一步，将数据转为float，不然进行forward时可能报错
        # Cat_Emb_B = Cat_emb_b.astype(np.float32)
        # print('拼接后的embedding的shape为：{}'.format(np.array(Cat_emb_a).shape))  # (1536,)  即1024 +512
        ## 6、生成cat向量 X 的cat_path
        Cat_path_a = str(wav_path_a) + '__' + str(face_path_a)  # 以'__'为连接符，拼接wav path和jpg path
        Cat_path_b = str(wav_path_b) + '__' + str(face_path_b)  # 以'__'为连接符，拼接wav path和jpg path
        ## 7、添加到list
        cat_embeds.append(Cat_emb_a)
        cat_embeds.append(Cat_emb_b)
        cat_paths.append(Cat_path_a)
        cat_paths.append(Cat_path_b)
        # print('拼接后路径为：{}的向量是：{}'.format(Cat_path,Cat_emb))
# 拼接后路径为：id10277/znxUWA2QAGs/00007.wav_id10277/0rpfN7wThsg/0000275.jpg的向量是：[ 0.07338596 -0.01595096  0.02492375 ... -0.01297332  0.10624083 0.05338817]
        f_val_trial.write('1 %s %s\n' % (Cat_path_a, Cat_path_b))
##  【3】、构造同样数量的 negative trial pairs  比较不同speaker的trial pair，标签取0。
    for i in range(nb_trg_trl):
        # 1、首先随机选取l_spk列表中 2 个speakers id，并且数值不能相同，因为这里要构造 negative trial pairs
        spks_cur = np.random.choice(l_spk, size=2, replace=False)  # spks_cur 元素数量为2个，是speaker id号
        # 2、分别根据这两个speaker id为键，查询得到他们的path list，再从各path list中随机选取1个元素，组成 a negative trial pair
        wav_path_a = np.random.choice(wav_id_dict[spks_cur[0]], size=1)[0]  # a和b不是同一个说话人
        wav_path_b = np.random.choice(wav_id_dict[spks_cur[1]], size=1)[0]
        face_path_a = np.random.choice(face_id_dict[spks_cur[0]], size=1)[0]
        face_path_b = np.random.choice(face_id_dict[spks_cur[1]], size=1)[0]
        ## 3、根据path为key去找到对应的embeddings value
        wav_emb_a = wav_emb_dict[wav_path_a]  # 一个1024维的向量
        wav_emb_b = wav_emb_dict[wav_path_b]  # 一个1024维的向量
        face_emb_a = face_emb_dict[face_path_a]  # 一个512维的向量
        face_emb_b = face_emb_dict[face_path_b]  # 一个512维的向量
        ## 4、进行向量拼接前，需对两个embedding进行L2-normalized
        wav_emb_re_a = np.array(wav_emb_a).reshape(1, -1)  # 需要先转换成二维数组
        wav_emb_re_b = np.array(wav_emb_b).reshape(1, -1)
        face_emb_re_a = np.array(face_emb_a).reshape(1, -1)
        face_emb_re_b = np.array(face_emb_b).reshape(1, -1)
        wav_emb_l2_a = preprocessing.normalize(wav_emb_re_a, norm='l2')  # L2正则化后的数值范围为[-1,1]
        wav_emb_l2_b = preprocessing.normalize(wav_emb_re_b, norm='l2')
        face_emb_l2_a = preprocessing.normalize(face_emb_re_a, norm='l2')
        face_emb_l2_b = preprocessing.normalize(face_emb_re_b, norm='l2')
        wav_emb_L2_a = wav_emb_l2_a[0]  # 变回一维
        wav_emb_L2_b = wav_emb_l2_b[0]
        face_emb_L2_a = face_emb_l2_a[0]
        face_emb_L2_b = face_emb_l2_b[0]
        # print('wav_emb_L2的type是：{}'.format(type(wav_emb_L2_a)))     # <class 'numpy.ndarray'>
        # print('wav_emb_L2的length是：{}'.format(len(wav_emb_L2_a)))    # 1024
        # print('face_emb_L2的length是：{}'.format(len(face_emb_L2_a)))  # 512
        ## 5、然后进行向量拼接，得到一个统一的嵌入 X
        Cat_emb_a = np.concatenate((wav_emb_L2_a, face_emb_L2_a))
        Cat_emb_b = np.concatenate((wav_emb_L2_b, face_emb_L2_b))
        # Cat_Emb_A = Cat_emb_a.astype(np.float32)  # 必须进行这一步，将数据转为float，不然进行forward时可能报错
        # Cat_Emb_B = Cat_emb_b.astype(np.float32)
        # print('拼接后的embedding的shape为：{}'.format(np.array(Cat_emb_a).shape))  # (1536,)  即1024 +512
        ## 6、生成cat向量 X 的cat_path
        Cat_path_a = str(wav_path_a) + '__' + str(face_path_a)  # 以'__'为连接符，拼接wav path和jpg path
        Cat_path_b = str(wav_path_b) + '__' + str(face_path_b)  # 以'__'为连接符，拼接wav path和jpg path
        ## 7、添加到list
        cat_embeds.append(Cat_emb_a)
        cat_embeds.append(Cat_emb_b)
        cat_paths.append(Cat_path_a)
        cat_paths.append(Cat_path_b)
        # print('拼接后路径为：{}的向量是：{}'.format(Cat_path,Cat_emb))
        # 拼接后路径为：id10277/znxUWA2QAGs/00007.wav_id10277/0rpfN7wThsg/0000275.jpg的向量是：[ 0.07338596 -0.01595096  0.02492375 ... -0.01297332  0.10624083 0.05338817]
        f_val_trial.write('0 %s %s\n' % (Cat_path_a, Cat_path_b))
    f_val_trial.close()
    return cat_embeds, cat_paths








