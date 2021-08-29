import os
import numpy as np
import pickle as pk
import itertools

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

from dataloader import *
from model import *
from trainer import *
from parsers import *
from utils import *
from loss import *


def read_data(filename, split=" "):
    with open(filename, mode="r", encoding='utf-8') as f:
        content_list = f.readlines()
        if split is None:
            content_list = [content.rstrip() for content in content_list]
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]
    return content_list
# 加载人脸数据库(embedding之后的)
def load_dataset(dataset_path, filename):
    embeddings = np.load(dataset_path)
    path_list = read_data(filename, split=None)  # read_data() 返回txt的数据列表
    return embeddings, path_list


def main():
##  【1】、解析参数
    args = get_args()
##  【2】、make experiment reproducible if specified  如有指定，使实验可重复进行（通过设定一个固定的随机种子）
    if args.reproducible:
        torch.manual_seed(args.seed)         # reproducible   # args.seed —————— default = 1234
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
##  【3】、device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Device: {}'.format(device))

##  【4】、获取说话人的face和wav的embedding数据集和path label
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
    # 其中wav_dev_emb_pk和wav_test_emb_pk 为 <class 'dict'> 类型

#   3、转换为“path与embedding一一对应的Dict”
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
    # 验证数据是否匹配
    print('face_dev_dict的emb数量为：{}'.format(len(face_dev_emb_dict)))    # 149519
    print('face_test_dict的emb数量为：{}'.format(len(face_test_emb_dict)))  # 4826
    print('wav_dev_dict的emb数量为：{}'.format(len(wav_dev_emb_dict)))    #  148642 
    print('wav_test_dict的emb数量为：{}\n'.format(len(wav_test_emb_dict)))  #  4874
   
#   4、生成 “vox1_dev_set” 的label，用于train, 其中speakers的id作为keys, speakers的class为values
    Vox_dev_label_dict = get_label_dic(list(wav_dev_emb_dict.keys()))   # 返回一个字典，如 {'id10001':0,'id10002':1, ... , 'id11251':1250}
    args.model['nb_classes'] = len(list(Vox_dev_label_dict.keys()))
    print('The train set have {} classes'.format(args.model['nb_classes']))  # 1211
    
#   5、制作“id与paths一对多”的New Dict，keys是说话人的id，values是说话人所有wav或face文件的paths
    Wav_Dev_Id_Dict, Face_Dev_Id_Dict = Id_to_mPath_Dict(wav_path_list=list(wav_dev_emb_dict.keys()),
                                                         face_path_list=list(face_dev_emb_dict.keys()))
    Wav_Test_Id_Dict, Face_Test_Id_Dict = Id_to_mPath_Dict(wav_path_list=list(wav_test_emb_dict.keys()),
                                                           face_path_list=list(face_test_emb_dict.keys()))

##  【6】、定义dataset generators
#   1、自定义一个dataset generators，用于生成 cat_embeds, cat_labels，cat_paths
    cat_d_embeds, cat_d_paths, cat_d_labels = Dev_Cat_Emb(wav_id_dict=Wav_Dev_Id_Dict, face_id_dict=Face_Dev_Id_Dict,
                                                      wav_emb_dict=wav_dev_emb_dict, face_emb_dict=face_dev_emb_dict,
                                                      embd_samples=args.nb_D_embd_samples, labels_dict=Vox_dev_label_dict)
    # cat_d_embeds 第一個元素为：[ 0.04119347  0.05533383 -0.0199192  ...  0.00145087 -0.05227421 -0.04864043] ，type爲 <class 'numpy.ndarray'>     
    # cat_d_embeds長度爲：100000，type为：<class 'list'>
    # cat_d_paths長度爲：100000，第一個元素为： id10856/URycIz_nE_I/00002.wav__id10856/MMC_b0dsXQc/0002425.jpg
    # cat_d_labels長度爲：100000，第一個元素为：815

    cat_t_embeds, cat_t_paths = Val_Cat_Emb(wav_id_dict=Wav_Test_Id_Dict, face_id_dict=Face_Test_Id_Dict,
                                             wav_emb_dict=wav_test_emb_dict, face_emb_dict=face_test_emb_dict,
                                             nb_trial=args.nb_valid_trial, dir_val_trial='trials/veri_test.txt')
    print('cat_t_embeds的元素數量爲：{}，第一個元素維度爲：{}，內容爲：{}'.format(len(cat_t_embeds), len(cat_t_embeds[0]), cat_t_embeds[0]))  
    # cat_t_embeds的長度爲：40000，第一個長度爲：1536，內容爲：[-0.03308959 -0.01561352  0.00579559 ... -0.00332587  0.07292533 -0.01478813]
    print('cat_t_paths的元素數量爲：{}，第一個元素爲：{}'.format(len(cat_t_paths), cat_t_paths[0]))
    # cat_t_paths的元素數量爲：40000，第一個元素爲：id10301/VIAnMw7J5tI/00010.wav__id10301/VIAnMw7J5tI/0001650.jpg
    with open('trials/veri_test.txt', 'r') as f:  # trial pairs为40000
        eval_trial_txt = f.readlines()
#   2、生成喂入model的dataset(Face+Wav)，得到cat_embedding X 和 label y    # 继承Dataset类 <class 'dataloader.Dataset_embd'>
    F_W_dev_set = Dev_Embd_Dataset(cat_embeds=cat_d_embeds, cat_labels=cat_d_labels)
    F_W_eval_set = Val_Embd_Dataset(cat_embeds=cat_t_embeds)
#   3、利用DataLoader类进行封装，以便进行迭代   default：batch_size = 100
    devset_gen = data.DataLoader(F_W_dev_set, batch_size=args.train_bs, shuffle=True, drop_last=True, num_workers=args.nb_worker)
    evalset_gen = data.DataLoader(F_W_eval_set, batch_size=args.valid_bs, shuffle=False, drop_last=False, num_workers=args.nb_worker) # batch_size有改動
    print('devset_gen长度为：{}'.format(len(devset_gen))) # 1000  因为数据量一共10W，有100个batch，所以每个batch就是1000
    print('evalset_gen长度为：{}'.format(len(evalset_gen))) # 40000  因为数据量一共4W，但只有1个batch

##  【7】、设置模型和参数结果的保存路径
    save_dir = args.save_dir  # 即 DNNs/
    # 如果下面的路径不存在，则先创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'results/'):
        os.makedirs(save_dir+'results/')
    if not os.path.exists(save_dir+'models/'):
        os.makedirs(save_dir+'models/')
##  【8】、将实验参数记录下来
    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in sorted(vars(args).items()):        # dir、hyper-params、flag等参数
        f_params.write('{}:\t{}\n'.format(k, v))
    for k, v in sorted(args.model.items()):        # DNN args 参数
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()

##  【9】、定义 Model
#   先判断是否支持多卡训练
    if bool(args.mg):   # args.mg 应该指是否有多张GPU？ 为True则可以进行多卡并行训练     # 这里的设置默认为True
#   1、先实例化Feature_Fusion模型
        model_1gpu = Feature_Fusion(args.model)  # args.model 是一个model内参字典
        AngularProto_net = AngularProtoLoss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        Arcface_net = ArcFaceNetloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        AAMsoftmax_net = AAMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        AMsoftmax_net = AMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        Centerloss_net = CenterLoss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        # 如果有训练好的模型，则直接导入
        if args.load_model:    # default = False  如果需要使用预训练模型就需要将其改为Ture
#   2、加载模型 model.load_state_dict(torch.load(PATH)) # 它是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系(如model的每一层的weights及偏置等等)
            model_1gpu.load_state_dict(torch.load(args.load_model_dir))
            print('多卡并行训练，加载 {} 预训练模型，Continue training！'.format(args.load_model_dir)) # load_model_dir为预训练模型文件
        # 计算model的参数量
        nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
        print('参数总量为: {}'.format(nb_params))  # Rawnet2: 29,404,572  # 我的: 6,487,227
        # 利用 nn.DataParallel函数 可以调用多个GPU，帮助加速训练
        model = nn.DataParallel(model_1gpu).to(device)
#   用单张卡进行训练
    else:
        model = Feature_Fusion(args.model).to(device)  # 将拼接后的嵌入作为融合网络model的input
        AngularProto_net = AngularProtoLoss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        Arcface_net = ArcFaceNetloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        AAMsoftmax_net = AAMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        AMsoftmax_net = AMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        Centerloss_net = CenterLoss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model_dir))
            print('单卡训练，加载 {} 预训练模型，Continue training！'.format(args.load_model_dir))
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print('参数总量为: {}'.format(nb_params))  # Rawnet2: 29,404,572  # 我的: 6,487,227
#   如果找不到预训练模型就需要从0开始训练
    if not args.load_model:
        print('没有预训练模型,从零开始训练！')
        model.apply(init_weights)   # 调用model.apply() 初始化模型的参数，并且会自动将参数信息都print出来
##  【10】、设置优化器参数
    # params = [
    #     { 'params': [
    #             param for name, param in model.named_parameters()
    #             if 'bn' not in name ] },
    #     { 'params': [
    #             param for name, param in model.named_parameters()
    #             if 'bn' in name
    #         ], 'weight_decay':0 },
    # ]
#   挑选优化器
    if args.optimizer.lower() == 'adam':    # lower() 将字符串中的所有大写字母转换为小写字母
#   1、初始化优化器        # default: lr=0.001, weight_decay = 0.0001, amsgrad = True
        # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
        optimizer = torch.optim.Adam(itertools.chain( model.parameters(),
            AngularProto_net.parameters(), Arcface_net.parameters(), AAMsoftmax_net.parameters(), AMsoftmax_net.parameters(), 
            Centerloss_net.parameters() ), 
            lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(itertools.chain(model.parameters()), lr=args.lr, momentum=args.opt_mom,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        raise NotImplementedError('Add other optimizers if needed')
#   如果有训练好的模型，则可以直接导入优化器
    if args.load_model:   # default = False
#   2、加载优化器参数，同model类一样也可以用 load_state_dict() 导入
        optimizer.load_state_dict(torch.load(args.load_model_opt_dir))    # args.load_model_opt_dir 为优化器模型的路径
##  【11】、设定学习率衰减
#   先判断是否要执行学习率衰减
    if bool(args.do_lr_decay):           # default = True  # bool()函数 Returns True when the argument x is true, False otherwise.
        if args.lr_decay == 'keras':     # default = 'keras'
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
            # torch.optim.lr_scheduler 模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
        elif args.lr_decay == 'cosine':
            raise NotImplementedError('Not implemented yet')
        else:
            raise NotImplementedError('Not implemented yet')
##  【12】、设置目标函数
    criterion = {}  # 评估准则
    criterion['CE_loss'] = nn.CrossEntropyLoss()  # 选用交叉熵损失函数
    criterion['NLL_loss'] = nn.NLLLoss()
    criterion['Arc_loss'] = Arcface_net
    criterion['AAMsoftmax_loss'] = AAMsoftmax_net
    criterion['AMsoftmax_loss'] = AMsoftmax_net
    criterion['Center_loss'] = Centerloss_net
    criterion['AP_loss'] = AngularProto_net

###############################
###### 开始训练 Train ##########
##############################
    # 设置一个初始EER值，后面会不断更新
    best_eval_eer = 99.
    # 用于保存每个epoch的train_loss和eval_eer结果
    f_eer = open(save_dir + 'Eval_EER.txt', 'a', buffering=1)  # 目录为：DNNs/eers.txt  # buffering设置为1时，表示在文本模式下使用行缓冲区方
    tb_writer = SummaryWriter('./log')
    # 开始分epoch进行训练
    for epoch in tqdm(range(args.epoch)):    # args.epoch：default = 80    # tqdm 是个显示进度条的库
        # 如果存在预训练模型，可以自行选择从第几个epoch继续训练
        # if epoch < 20:
        #     continue        # train phase
        # train phase   调用自定义的 train_model进行模型训练，返回的是一个loss分数
        train_model(model = model,       # 导入设置好的模型（已经参数化）
            devset_gen = devset_gen,         # 导入经处理后的train数据集，DataLoader类封装好的 # 包含 Cat嵌入X 和 label Y
            optimizer = optimizer,       # 导入优化器
            lr_scheduler = lr_scheduler, # 导入学习率调度器
            criterion = criterion,       # 导入损失函数
            epoch = epoch,               # 导入训练的epoch数
            device = device,             # 导入训练的device
            args = args)                 # 导入所有的参数
        # evaluate phase  调用自定义的evaluate_model进行模型评估，返回的是一个EER分数
        eval_eer = evaluate_model(model = model,
            evalset_gen = evalset_gen,    # 导入经处理后的test数据集 # 包含嵌入值 X ，但不包括label Y
            eval_trial_txt = eval_trial_txt,    # 从DB/VoxCeleb1/veri_test.txt 中导入trial，其中 1 表示Positive sample pair，0 表示Negative sample pair
            cat_paths = cat_t_paths,  # 根据自定义Cat向量数据集的wav+jpg路径生成的cat_path list
            save_dir = save_dir,      # DNNs/
            epoch = epoch,
            device = device)
        # 将当前epoch的eval_eer结果实时保存起来
        tb_writer.add_scalars('Evaluation', {'EER':eval_eer}, epoch)
        f_eer.write('epoch:%d, eval_eer:%.4f\n'%(epoch, eval_eer))
        # 若当目前epoch的最新EER值低于之前最好的EER，则更新，EER越小越好
        if float(eval_eer) < best_eval_eer:
            best_eval_eer = float(eval_eer)
            print('New best EER: %f' % float(eval_eer))
            # 将当前epoch的 model参数和optimizer参数 都保存为.pt文件
            save_model_dict = model_1gpu.state_dict() if args.mg else model.state_dict()
            torch.save(save_model_dict, save_dir+'models/epoch%d_%.4f.pt'%(epoch, eval_eer) ) # 路径形如：DNNs/models/epoch1_0.1965.pt
            torch.save(optimizer.state_dict(), save_dir+'models/best_optimizer_eval.pt') # 路径形如：DNNs/models/best_optimizer_eval.pt
            # 保存整个网络
            torch.save(model, 'DNNs/models/best_pre_model.pkl')
    f_eer.close()

if __name__ == '__main__':
    main()
