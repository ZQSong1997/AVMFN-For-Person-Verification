import torch
import numpy as np
from torch.nn.functional import embedding

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from utils import cos_sim

def train_model(model, devset_gen, optimizer, lr_scheduler, criterion, epoch, device, args):
    model.train()
    tb_writer = SummaryWriter('./log')
    f_loss = open('DNNs/Train_Loss.txt', 'a', buffering=1)
    with tqdm(total=len(devset_gen), ncols=70) as pbar:   # len(db_gen)为400
        for batch_index, (m_batch, m_label) in enumerate(devset_gen):
            n_iter = epoch * len(devset_gen) + batch_index
            m_batch, m_label = m_batch.to(device), m_label.to(device)  # 将该batch数据都先指定device
            embeddings, output = model(m_batch)
            # 定義各種损失函數
            out = criterion['Arc_loss'](embeddings)  # out是class_number, embeddings是1024
            Arcface_loss = criterion['NLL_loss'](out, m_label)  # 采用arcface_loss计算output和label之间的距离  # NLL_loss是交叉熵損失函數
            # Arcface_loss = criterion['Arc_loss'](embeddings, m_label)
            AAMsoftmax_loss = criterion['AAMsoftmax_loss'](embeddings, m_label)  # 等價與上面兩行，AAMsoftmax即arcface_loss
            AMsoftmax_loss = criterion['AMsoftmax_loss'](embeddings, m_label)
            Center_loss = criterion['Center_loss'](embeddings, m_label)
            CE_loss = criterion['CE_loss'](output, m_label)  # 采用cross entropy loss计算output和label之间的距离
            # 計算loss
            alpha = 0.6  # 不同損失的权重比例
            lam_bda = 0.2
            loss = AMsoftmax_loss
            # loss = alpha*CE_loss + (1-alpha)*Center_loss 
            # loss = CE_loss + lam_bda*Center_loss 
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward()     # backpropagation, compute gradients
            optimizer.step()    # apply gradients
            # 自定义进度条的显示内容
            pbar.set_description('epoch: %d, cce_loss:%.3f' % (epoch, loss))
            pbar.update(1)      # 更新进度条的长度
            if args.do_lr_decay:
                if args.lr_decay == 'keras':
                    lr_scheduler.step()
            tb_writer.add_scalars('Train loss', {'CE loss': loss.data.item()}, n_iter)
            f_loss.write('epoch:%d, train_loss:%.4f\n' % (epoch, loss))
    f_loss.close()
    

# evalset_gen：由DataLoader封装后的test集，仅包含嵌入值X，不包含label  # 本文随机组合了1W个嵌入向量
# eval_trial_txt：从DB/VoxCeleb1/veri_test.txt 中导入trial，其中 1 表示Positive sample pair，0 表示Negative sample pair
# cat_paths：cat_t_paths 根据自定义Cat向量数据集的wav+jpg路径生成的cat_paths list
# save_dir：# DNNs/
def evaluate_model(model, evalset_gen, eval_trial_txt, cat_paths, save_dir, epoch, device):
    model.eval()   # 在评估之前，务必调用model.eval()去设置 dropout 和 batch normalization 层为evaluation模式。如果不这么做，可能导致模型推断结果不一致。
    tb_writer = SummaryWriter('./log')
    with torch.set_grad_enabled(False):   # 接下来所有的tensor运算产生的新的节点都是不可求导的
## 第一步：提取说话人cat后的嵌入
        l_cat_embeddings = []
        # tqdm 用于显示进度条，total参数设置进度条的总长度，len(evalset_gen)这里应该是40000，原本是4874
        with tqdm(total=len(evalset_gen), ncols=70) as pbar:
            for m_batch in evalset_gen:  # 每个mini_batch的长度都不一样，例如 7~10。最后所有的mini_batch加起来应该等于len(evalset_gen)
                m_batch = m_batch.to(device) 
                code = model(x=m_batch, is_test=True)  # 返回的是输出层前面的全连接层的Output，维度为1024
                l_cat_embeddings.extend(code.cpu().numpy())
                pbar.update(1)  # 每次更新进度条的长度，这里相当于每处理完一个样本就进度条+1
        # 添加一个字典，以将Cat path和Cat embedding以字典形式一一对应起来
        d_embeddings = {}
        # 先判断数量是否一致，不一致会出错。
        # print('查看数量是否一致，其中cat_paths的length：{}，l_cat_embeddings的length：{}'.format( len(cat_paths), len(l_cat_embeddings)) )
        if not len(cat_paths) == len(l_cat_embeddings):   # cat_paths示例：
            print('cat_paths和l_cat_embeddings數量不一致')
            exit()
        # 将数据集的cat_path作为keys，cat_embeddings作为values以Dict的形式存储起来
        for k, v in zip(cat_paths, l_cat_embeddings):
            d_embeddings[k] = v
## 第二步，计算EER
        y_score = []     # 每个样本的预测结果
        y_label = []     # 真实的样本标签
        f_res = open(save_dir+'results/eval_epoch_{}.txt'.format(epoch), 'w')  # 如 DNNs/results/eval_epoch_10.txt
        # l_eval_trial 为DB/VoxCeleb1/veri_test.txt中的trials对，其中1表示Positive sample pair，0表示Negative sample pair
        # line的示例：1 id10289/sf4uMnkYFG8/00002.wav__id10289/8l5ZnDf-FUA/0005500.jpg id10289/sf4uMnkYFG8/00005.wav__id10289/8l5ZnDf-FUA/0003600.jpg
        for line in eval_trial_txt:
            trg, cat_utt_face_a, cat_utt_face_b = line.strip().split(' ')   # 用strip()去掉换行符,以split(' ')空格隔开
            # 将该trial pair的label添加到标签list中
            y_label.append(int(trg))  # 1或0
            # 将cat_path作为字典的key去查询到对应的value，即cat_embedding。然后利用余弦相似度函数计算两个cat_embedding之间的距离，得到预测score，最后存到list中。
            y_score.append(cos_sim(d_embeddings[cat_utt_face_a], d_embeddings[cat_utt_face_b]))
            f_res.write('{score} {target}\n'.format(score=y_score[-1],target=y_label[-1]))  # 因为是往list中append，所以最新的值存在-1处。
        f_res.close()   # 结果是：DNNs/results/eval_epoch10.txt文件中有10000行(trial pair数量),2列。第一列是预测结果y_score，第二列是真实样本标签y_label。
        # 调用ROC曲线函数
        fpr, tpr, _ = roc_curve(y_label, y_score, pos_label=1)  # pos_label=1 表示正样本的标签为1   # roc_curve() 函数有3个返回值
        # 调用 scipy.optimize.brentq() 函数找两个函数的根（即它们的曲线相交的地方）
        f = lambda x: 1. - x - interp1d(fpr, tpr)(x)
        eer = brentq(f, 0., 1.)
    return eer


### 函数介绍：
## sklearn.metrics.roc_curve() 函数的用法
# 用于绘制ROC曲线，ROC曲线是以FPR为横坐标，以TPR为纵坐标，以概率为阈值来度量模型正确识别正实例的比例
# 与模型错误的把负实例识别成正实例的比例之间的权衡，TPR的增加必定以FPR的增加为代价，ROC曲线下方的面积是模型准确率的度量
# 主要参数如下：
# y_true：真实的样本标签，默认为{0，1}或者{-1，1}。如果要设置为其它值，则 pos_label 参数要设置为特定值。例如要令样本标签为{1，2}，其中2表示正样本，则pos_label=2。
# y_score：对每个样本的预测结果。
# pos_label：正样本的标签。
# 返回值的计算：fpr：False positive rate；   tpr：True positive rate；     thresholds

## scipy.Optimize.brentq(f，a，b) 求解非线性方程，二分法找根值0
# 第一个参数f是要求解的用户定义函数的名称。接下来的两个参数a和b是包含您要查找的解决方案的x值。
# 您应该选择a和b，以便在a和b之间的区间内只有一个解。Brent method还要求f(A)和f(B)具有相反的符号；如果没有相反的符号，则返回错误消息。

##  model.train() 与 model.eval() 函数的用法
# 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
# 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
# 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。