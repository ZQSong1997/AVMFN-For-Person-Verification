# Audio Visual Multimodel Fusion Network for Person Verification

#### Overview
本项目是基于人脸+声纹的多模态生物验证算法的实现，人脸特征采用Facenet模型提取，声纹特征采用Rawnet2模型提取。融合网络采用Gated Multi-Modal Fusion Network。

#### Model architecture
![Front-end feature extractor](https://images.gitee.com/uploads/images/2021/0829/184341_dea79126_7955921.png "屏幕截图.png")
![back-end Gated Fusion Network](https://images.gitee.com/uploads/images/2021/0829/202522_84858183_7955921.png "屏幕截图.png")

#### datasets
![Voxceleb1 dataset](https://images.gitee.com/uploads/images/2021/0829/203933_f4923fee_7955921.png "屏幕截图.png")
1.  数据集采用Voxceleb1 dev set作为训练集，test set作为测试集。  [数据集链接](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
2.  Face embeddings可在'DB/Vox1_Face/emb/'获取，其中vox1_face_D_embedding.npy和vox1_face_T_embedding.npy分别是Voxcelev1训练集和测试集的face图片的embeddings，为512维的向量。vox1_face_D_label.txt和vox1_face_T_label.txt为对应的label list，如：[id10001/1zcIwhmdeo4/0000375.jpg,id10001/1zcIwhmdeo4/0000475.jpg,...]。（[百度云链接](https://pan.baidu.com/s/15T7tvXb-FUgpLU3kpQN7zw 
)提取码：38gq）
3.  Speaker embeddings可在'DB/Voxceleb1/emb/'获取，其中TTA_vox1_dev.pk和TTA_vox1_eval.pk分别是Voxcelev1训练集和测试集的wav语音的embeddings。是一个字典格式的文件，keys为Speaker id，如'id10001/1zcIwhmdeo4/00001.wav'，values为对应的1024维的embeddings。（[百度云链接](https://pan.baidu.com/s/1eNzLMzZmuKvgrxNm1XoRvQ) 提取码：bnk9）
4.  预训练模型可在'Pre-trained_model/best_model.pkl'获得。

#### Usage
1.    数据预处理：Run data_preprocess.py
2.    训练模型：Run trainning_AVN.py
3.    评估模型：Run evaluation_model.py



