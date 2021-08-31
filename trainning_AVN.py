import itertools
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import *
from data_preprocess import *
from trainer import *

'''
######      model训练模块     ######
'''
def main():
    ##  【1】、解析参数
    args = get_args()
    ##  【2】、device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Device: {}'.format(device))
    ##  【3】、通过设定一个固定的随机种子,使实验可重复进行
    if args.reproducible:
        torch.manual_seed(args.seed)         # reproducible   # args.seed —————— default = 1234
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ##  【4】、设置模型和参数结果的保存路径
    save_dir = args.save_dir  # 即 'DNNs/'
    # 如果路径不存在，则先创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'results/'):
        os.makedirs(save_dir+'results/')
    if not os.path.exists(save_dir+'models/'):
        os.makedirs(save_dir+'models/')
    ##  【5】、记录实验参数
    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in sorted(vars(args).items()):        # dir、hyper-params、flag等参数
        f_params.write('{}:\t{}\n'.format(k, v))
    for k, v in sorted(args.model.items()):        # DNN args 参数
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()
    ##  【6】、设置目标函数
    AAMsoftmax_net = AAMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes'], margin=0.1, scale=60).to(device)
    AMsoftmax_net = AMsoftmaxloss(feature_dim=1024, cls_num=args.model['nb_classes'], margin=0.1, scale=80).to(device)
    Centerloss_net = CenterLoss(feature_dim=1024, cls_num=args.model['nb_classes']).to(device)
    ##  【7】、生成一个字典来存储各类评估准则
    criterion = {}
    criterion['CE_loss'] = nn.CrossEntropyLoss()  # 选用交叉熵损失函数
    criterion['AAMsoftmax_loss'] = AAMsoftmax_net
    criterion['AMsoftmax_loss'] = AMsoftmax_net
    criterion['Center_loss'] = Centerloss_net

##  【1】、初始化model
#   先判断是否支持多卡训练
    if bool(args.mg):   # args.mg 应该指是否有多张GPU？ 为True则可以进行多卡并行训练，默认为True
        model_1gpu = Gated_Feature_Fusion(args.model)  # 先实例化model，args.model 是一个model内参字典
        # model_1gpu = Attention_Feature_Fusion(args.model).to(device)
        # model_1gpu = Cat_Feature_Fusion(args.model).to(device)
        if args.load_model:    # 为Ture则导入预训练模型，默认为False
            model_1gpu.load_state_dict(torch.load(args.load_model_dir))  # 加载模型参数
            print('多卡并行训练，加载 {} 预训练模型，Continue training！'.format(args.load_model_dir)) # load_model_dir为预训练模型文件
        # 计算model的参数量
        nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
        print('参数总量为: {}'.format(nb_params))
        # 利用 nn.DataParallel函数调用多个GPU，帮助加速训练
        model = nn.DataParallel(model_1gpu).to(device)
#   用单张卡进行训练
    else:
        model = Gated_Feature_Fusion(args.model).to(device)  # 将拼接后的嵌入作为融合网络model的input
        # model = Attention_Feature_Fusion(args.model).to(device) 
        # model = Cat_Feature_Fusion(args.model).to(device)
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model_dir))
            print('单卡训练，加载 {} 预训练模型，Continue training！'.format(args.load_model_dir))
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print('参数总量为: {}'.format(nb_params))  # Rawnet2: 29,404,572  # 我的: 6,487,227
#   如果找不到预训练模型就需要从头开始训练
    if not args.load_model:
        model.apply(init_weights)   # 调用model.apply()初始化模型的参数，并且会自动将参数信息都print出来
##  【2】、optimizer设置
    if args.optimizer.lower() == 'adam':    # lower()将字符串中的所有大写字母转换为小写字母
        optimizer = torch.optim.Adam(itertools.chain( model.parameters(),
            AAMsoftmax_net.parameters(), AMsoftmax_net.parameters(), Centerloss_net.parameters() ),
            lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(itertools.chain(model.parameters()), lr=args.lr, momentum=args.opt_mom,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        raise NotImplementedError('Add other optimizers if needed')
#   如果有训练好的模型，则可以直接导入optimizer
    if args.load_model:   # default=False
        optimizer.load_state_dict(torch.load(args.load_model_opt_dir))    # args.load_model_opt_dir 为optimizer参数的保存路径
##  【3】、设定学习率衰减
    if bool(args.do_lr_decay):           # default = True  # bool()函数 Returns True when the argument x is true, False otherwise.
        if args.lr_decay == 'keras':     # default = 'keras'
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
            # torch.optim.lr_scheduler 模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
        elif args.lr_decay == 'cosine':
            raise NotImplementedError('Not implemented yet')
        else:
            raise NotImplementedError('Not implemented yet')
##  【4】、开始分epoch进行训练
    best_eval_eer = 99.   # 预设一个初始EER值，后面再不断更新
    # 保存每个epoch的train loss和eval_eer
    f_eer = open(save_dir + 'Eval_EER.txt', 'a', buffering=1)   # buffering设置为1时，表示在文本模式下使用行缓冲区
    tb_writer = SummaryWriter('./log')
    for epoch in tqdm(range(args.epoch)):    # args.epoch：default = 80    # tqdm 是个显示进度条的库
        # 如果存在预训练模型，可以自行选择从第几个epoch继续训练
        # if epoch < 20:
        #     continue
        # train phase   调用自定义的 train_model进行模型训练，返回一个loss分数
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
            # 将当前epoch的 model和optimizer的参数保存为.pt文件
            save_model_dict = model_1gpu.state_dict() if args.mg else model.state_dict()
            torch.save(save_model_dict, save_dir+'models/epoch%d_%.4f.pt'%(epoch, eval_eer) ) # 路径形如：DNNs/models/epoch10_0.6133.pt
            torch.save(optimizer.state_dict(), save_dir+'models/best_optimizer_eval.pt') # 路径形如：DNNs/models/best_optimizer_eval.pt
            # 保存整个网络
            torch.save(model, 'Pre-trained_model/best_model.pkl')
    f_eer.close()

if __name__ == '__main__':
    main()
