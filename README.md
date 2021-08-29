# Audio Visual Multimodel Fusion Network for Person Verification

#### 介绍
本项目是基于人脸+声纹的多模态生物验证算法的实现，人脸特征采用Facenet模型提取，声纹特征采用Rawnet2模型提取。融合网络采用Gated Multi-Modal Fusion Network。
#### 模型架构
![Front-end feature extractor](https://images.gitee.com/uploads/images/2021/0829/184341_dea79126_7955921.png "屏幕截图.png")
![back-end Gated Fusion Network](https://images.gitee.com/uploads/images/2021/0829/183620_bfce2a8d_7955921.png "屏幕截图.png")

#### 使用说明
1.  Face embeddings可在'DB/Vox1_Face/emb/'获取，其中vox1_face_D_embedding.npy和vox1_face_T_embedding.npy分别是Voxcelev1训练集和测试集的face图片的embeddings，为512维的向量。vox1_face_D_label.txt和vox1_face_T_label.txt为对应的label list，如：[id10001/1zcIwhmdeo4/0000375.jpg,id10001/1zcIwhmdeo4/0000475.jpg,...]
2.  Speaker embeddings可在'DB/Voxceleb1/emb/'获取，其中TTA_vox1_dev.pk和TTA_vox1_eval.pk分别是Voxcelev1训练集和测试集的wav语音的embeddings。是一个字典格式的文件，keys为Speaker id，如'id10001/1zcIwhmdeo4/00001.wav',values为对应的1024维的embeddings。如：[...,'id10426/UUpSDo9_Qi4/00025.wav', 'id10426/WIbIl2skqjk/00001.wav',...]
3.  预训练模型可在'Pre-trained_model/best_model.pkl'获得。

