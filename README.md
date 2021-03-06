# Audio Visual Multimodel Fusion Network for Person Verification

### Overview
本项目是基于人脸+声纹的多模态生物验证算法的实现，人脸特征采用Facenet模型[1]提取，声纹特征采用Rawnet2模型[2]提取。
通过在Voxceleb1数据集上训练和测试，EER指标最佳能达到0.548%，显著优于现有最先进的单模态的人脸识别或声纹识别模型，即使和同类型算法（人脸+声纹）相比，也达到了SOTA水平。

### Model architecture
![Front-end feature extractor](https://images.gitee.com/uploads/images/2021/0829/184341_dea79126_7955921.png "屏幕截图.png")
![back-end Gated Fusion Network](https://images.gitee.com/uploads/images/2021/0903/095703_c2670694_7955921.png "屏幕截图.png")

### datasets
![Voxceleb1 dataset](https://images.gitee.com/uploads/images/2021/0829/203933_f4923fee_7955921.png "屏幕截图.png")
1.  数据集采用Voxceleb1 dev set作为训练集，test set作为测试集。  [数据集官方链接](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
2.  Face embeddings可在'DB/Vox1_Face/emb/'获取，其中vox1_face_D_embedding.npy和vox1_face_T_embedding.npy分别是Voxcelev1训练集和测试集的face图片的embeddings（512维）。vox1_face_D_label.txt和vox1_face_T_label.txt为对应的label list，如：[id10001/1zcIwhmdeo4/0000375.jpg,id10001/1zcIwhmdeo4/0000475.jpg,...]。（[百度云链接](https://pan.baidu.com/s/15T7tvXb-FUgpLU3kpQN7zw 
) 提取码：38gq）
3.  Speaker embeddings可在'DB/Voxceleb1/emb/'获取，其中TTA_vox1_dev.pk和TTA_vox1_eval.pk分别是Voxcelev1训练集和测试集的wav语音的embeddings。是一个字典格式的文件，keys为Speaker id，如'id10001/1zcIwhmdeo4/00001.wav'，values为对应的embeddings（1024维）。（[百度云链接](https://pan.baidu.com/s/1eNzLMzZmuKvgrxNm1XoRvQ) 提取码：bnk9）
4.  预训练模型可在'DNNs/Pre_trained_model/best_model.pkl'获得。

### Usage
1.    数据预处理：Run data_preprocess.py
2.    训练模型：Run trainning_AVN.py
3.    评估模型：Run evaluation_model.py

### Reference
1.    [1] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.
2.    [2] Jung J, Kim S, Shim H, et al. Improved rawnet with feature map scaling for text-independent speaker verification using raw waveforms[J]. arXiv preprint arXiv:2004.00526, 2020.



