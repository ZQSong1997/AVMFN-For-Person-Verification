# -*-coding: utf-8 -*-
"""
    # 该文件用于生成embedding数据库（人脸特征），后面待检测识别的人脸，只需要与这些embedding数据库（人脸特征）进行相似性比较，就可以识别人脸啦！！！！
"""
import time
import datetime
import numpy as np
from utils import image_processing , file_processing, debug
import face_recognition
import os
import cv2

resize_width = 160
resize_height = 160

### 获得face_embedding
def get_face_embedding(model_path, filePath_list):
    # 转换颜色空间RGB or BGR
    colorSpace = "RGB"
    # 初始化mtcnn人脸检测
    face_detect = face_recognition.Facedetection()
    # 初始化facenet
    face_net = face_recognition.facenetEmbedding(model_path) # model_path = 'models/20180408-102900' # faceNet模型路径；保存有FaceNet的Pretrained Model
    embeddings = [] # 用于保存人脸特征的数据库
    image_path_list = [] # 保存图像的路径,本工作会将其作为图片的label
    print("开始处理图片...")
    for  image_path in filePath_list:
        print("processing image：{}".format(image_path))    
        # processing image：DB/Vox2_Face/dev_face/id00017/01dfn2spqyE/001_050.jpg
        image = image_processing.read_image_gbk(image_path, colorSpace=colorSpace)
        # 进行人脸检测，获得bounding_box
        bboxes, landmarks = face_detect.detect_face(image)
        bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")
        # image_processing.show_image_boxes("image",image, bboxes)
        # 下面两种情况生成不了人脸嵌入，程序会选择跳过
        if bboxes == [] or landmarks == []:
            print("--- {} image have no face !".format(image_path))
            continue    # 注意，由于跳过，会影响嵌入生成的数量少于实际的图片数
        if len(bboxes) >= 2 or len(landmarks) >= 2:
            print("--- {} image have {} faces ".format(image_path, len(bboxes)))  # 只有一张人脸代表正常，不会print提示
            continue
        # 获得人脸区域
        face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width)
        # 人脸预处理，归一化
        face_images = image_processing.get_prewhiten_images(face_images, normalization=True)
        # 利用face_net获得人脸特征embeddings
        pred_emb = face_net.get_embedding(face_images)
        # 保存embeddings、image_path_list
        # 可以选择保存image_list或者names_list作为人脸的标签  # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
        embeddings.append(pred_emb) 
        image_path_list.append(image_path[-31:])  # 这里自己加的,注意要根据情况修改里面的数字  -31 爲 id00017/01dfn2spqyE/001_050.jpg
    print("embeddings、image_path_list已保存成功\n")
    return embeddings, image_path_list

def create_face_embedding(model_path, dataset_path, out_face_emb_path, out_face_label_path):
    # 可以获取目录dataset_path文件夹下所有文件，包括子目录下的所有文件路径（files_list），其中names_list就是子目录的文件名，一般子目录作为样本的标签。
    filePath_list, names_list = file_processing.gen_files_labels(dataset_path, postfix=['*.jpg']) # return filePath_list, name_list
    embeddings, image_path_list = get_face_embedding(model_path, filePath_list)
    # print("image_path_list:{}".format(image_path_list))       # label ，这里是图片的path ['id11001/WnFx5eZVVic/0004100.jpg', 'id11001/ZKP0nYyjg5s/0001050.jpg', ..., 'id11001/ZKP0nYyjg5s/0001150.jpg',]
    print("The Face datasets have {} label\n".format(len(image_path_list)))  # 即拥有相应数量的用户
            # The Vox1 test set have 4826 label, Vox1 dev set have 149519 label.
            # The Vox2 test set have 126665 label, Vox2 dev set have 149519 label.
    # 保存 embeddings,image_path_list
    embeddings = np.asarray(embeddings)       # <class 'numpy.ndarray'>
    print('embeddings的长度{}'.format(len(embeddings)))
    np.save(out_face_emb_path, embeddings)    # 将embeddings 保存为.npy文件
    # 利用file_processing.write_list_data()函数 保存list[]的数据到txt文件，每个元素分行
    file_processing.write_list_data(out_face_label_path, image_path_list, mode='w')   # 将用户的label（名字/id）保存为txt文件


if __name__ == '__main__':
    model_path = 'DB/facenet_pretrained_model/20180408-102900/'           # faceNet模型路径；保存有FaceNet的Pretrained Model
    in_dataset_path = 'DB/Vox2_Face/dev_face'       # 人脸数据库路径，每一类单独一个文件夹
    out_face_emb_path = 'DB/Vox2_Face/emb/vox2_face_D_embedding.npy'  # 输出embeddings的保存路径；.npy是原始数据集经Embedding后的输出结果
    out_face_label_path = 'DB/Vox2_Face/emb/vox2_face_D_label.txt'  # 输出与embeddings一一对应的path标签
    # 计算程序的处理时间
    start_date = datetime.datetime.now()
    print("开始处理时间为：{}".format(str(start_date)))
    ## 创建人脸embedding
    create_face_embedding(model_path, in_dataset_path, out_face_emb_path, out_face_label_path)
    end_date = datetime.datetime.now()
    print("数据处理总时长为：{}".format(str(start_date - end_date)))


