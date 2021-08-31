
##########      1、从本地文件夹中遍历视频文件并提取视频帧       ##########
import os
import cv2
import datetime

# 计算程序的处理时间
start_date = datetime.datetime.now()
print("开始处理时间为：{}".format(str(start_date)))

cut_frame = 50     # 多少帧截一次，自己设置就行
for root, dirs, files in os.walk(r"DB/Vox2_Face/dev/mp4/"):   # 这里就填文件夹目录就可以了
    for file in files:
        # 获取文件路径
        if ('.mp4' in file):
            path = os.path.join(root, file)  # path如：DB/Vox2_Face/dev/mp4/id10001/7gWzIy6yIIk/00274.mp4  file如：00274.mp4
            # 获得视频的格式
            video = cv2.VideoCapture(path)   # VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            # 获得视频的码率
            video_fps = int(video.get(cv2.CAP_PROP_FPS))
            # print('视频{}的fps是{}'.format(path,video_fps))  # 每秒多少帧
            current_frame = 0
            while (True):
                # 读帧
                ret, image = video.read()  # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。image就是每一帧的图像，是个三维矩阵。
                current_frame = current_frame + 1
                if ret is False:
                    video.release()
                    break
                if current_frame % cut_frame == 0:
                    save_dir = "DB/Vox2_Face/dev_face/" + path[-29:-9]   # id10001/1zcIwhmdeo4/
                    # 路径不存在，则新建
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if len(str(current_frame)) == 2:   # 针对current_frame为两位数、三位数长度不一致从而导致文件名长度不一的问题
                        cv2.imencode('.jpg',image)[1].tofile(save_dir +file[2:-4] +'_0'+str(current_frame)+'.jpg')
                        print('正在保存' +save_dir +file[2:-4] +'_0'+str(current_frame)+'.jpg')
                        # 正在保存DB/Vox2_Face/dev_face/id10001/1zcIwhmdeo4/001_050.jpg
                    if len(str(current_frame)) == 3:
                        cv2.imencode('.jpg',image)[1].tofile(save_dir +file[2:-4] +'_'+str(current_frame)+'.jpg')
                        print('正在保存' +save_dir +file[2:-4] +'_'+str(current_frame)+'.jpg')
                        # 正在保存DB/Vox2_Face/dev_face/id10001/1zcIwhmdeo4/001_350.jpg
end_date = datetime.datetime.now()
print("数据处理总时长为：{}".format(str(start_date - end_date)))
# 使用imencode()相比cv2.imwrite()支持整个路径和文件名包括中文



##########      2、遍历文件夹里面的每个文件      ##########
# import os
# # 遍历文件夹
# def walkFile(file):
#     for root, dirs, files in os.walk(file):
#      # root 表示当前正在访问的文件夹路径
#      # dirs 表示该文件夹下的子目录名list
#      # files 表示该文件夹下的文件list
#         # 遍历文件
#         print('当前根路径：{}'.format(root))
#         print('遍历所有的文件夹：')
#         # 遍历所有的文件夹
#         for d in dirs:
#             print(os.path.join(root, d))
#         print('遍历文件：')
#         for f in files:
#             print(os.path.join(root, f))
#         print('——————————————————————————')
#
# def main():
#     walkFile("DB/Vox2_Face/")
#
# if __name__ == '__main__':
#     main()



##########      3、 OpenCV从摄像头/本地视频文件中读取数据，并显示在窗口上       ##########
# import cv2
# import os

# index = 0
# gap = 60  # 间隔60帧抽取保存一帧。
# savedpath = 'image/'
# isExists = os.path.exists(savedpath)
# if not isExists:
#     os.makedirs(savedpath)
#     print("创建存储路径")
# else:
#     print("路径已经存在")
#
# cameraCapture = cv2.VideoCapture(0)  # 调用本地摄像头捕获图片  #VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
# # 从本地视频中Capture提取图片帧！！！
# # cameraCapture = cv2.VideoCapture("D:/ProjectFiles/Live-Face-Verification-Using-Deep-Learning-master/face_Verification/video67.mp4")
# cv2.namedWindow('test_camera')
#
# while (cameraCapture.isOpened()):
#
#     ASCII = cv2.waitKey(
#         1)  # waitKey()方法表示等待键盘输入。参数是1，表示延时1ms切换到下一帧图像，对于视频而言；参数为0，只显示当前帧图像，相当于视频暂停；参数过大如cv2.waitKey(1000)，会因为延时过久而卡顿感觉到卡顿。
#     if ASCII == 27:  # ASCII得到的是键盘输入的ASCII码。Esc的ASCII码为27。如果在1ms内按下了Esc，break，否则继续执行下面的代码
#         break
#     # 按帧读取视频
#     success, frame = cameraCapture.read()  # 其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
#     if not success:
#         print("读取失败")
#         break
#     else:
#         print("读取成功")
#     index += 1
#     # 显示摄像头捕获的信息
#     cv2.imshow('test_camera', frame)  # 将frame中的数据，显示到名字为”test camera“ 的窗口上。
#     # 将视频按照指定间隔帧存储图片文件。
#     if (index % gap == 0):
#         cv2.imwrite(savedpath + str(index) + '.jpg', frame)
# print("结束")
# cameraCapture.release()  # 调用release()释放摄像头
# cv2.destroyAllWindows()  # 调用destroyAllWindows()关闭所有图像窗口。









# # 待分帧的原始视频路径、分帧后需要保存视频帧的文件路径
# videos_src_path = 'DB/Vox2_Face/id10001/1zcIwhmdeo4'
# videos_save_path = 'DB/Vox2_Face/id10001/1zcIwhmdeo4'
# # 将所有视频的文件名存到videos中
# videos = os.listdir(videos_src_path)
# # videos.sort(key=lambda x: int(x[5:-4]))
# # 这一行是可选项，表示对fights文件夹下的视频进行排序，x[5:-4]表示按照文件名第5个字符到倒数第4个字符之间的符号排序，因为我的视频是newfi1.avi,newfi2.avi……的格式，我想实现的是按照数字1,2,3的顺序提取视频帧；
#
# i = 1
# # 之后的代码就是循环处理每个视频，将每个视频的视频帧保存到文件夹中。
# for each_video in videos:
#     if not os.path.exists(videos_save_path + '/' + str(i)):
#         os.mkdir(videos_save_path + '/' + str(i))
#     each_video_save_full_path = os.path.join(videos_save_path, str(i)) + '/'
#     each_video_full_path = os.path.join(videos_src_path, each_video)
#     cap = cv2.VideoCapture(each_video_full_path)
#     frame_count = 1
#     success = True
#
#     while (success):
#         success, frame = cap.read()
#     if success == True:
#         cv2.imwrite(each_video_save_full_path + "frame%d.jpg" % frame_count,
#                     frame)
#     frame_count = frame_count + 1
#     i = i + 1
#
#     cap.release()

