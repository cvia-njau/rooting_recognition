import os
import cv2
import numpy as np

def convert(bbox, shape):
    x1 = int((bbox[0] - bbox[2] / 2.0) * shape[1])
    y1 = int((bbox[1] - bbox[3] / 2.0) * shape[0])
    x2 = int((bbox[0] + bbox[2] / 2.0) * shape[1])
    y2 = int((bbox[1] + bbox[3] / 2.0) * shape[0])
    return x1, y1, x2, y2

def convertcenter(bbox, shape):
    hx_center = int(shape[1] * bbox[0])
    hy_center = int(shape[0] * bbox[1])
    hwidth = int(shape[1] * bbox[2])
    hheight = int(shape[0] * bbox[3])
    return hx_center, hy_center, hwidth, hheight

def save_image(image, addr, num):
    address = addr + str(num) + '.png'
    cv2.imwrite(address, image)

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

detectdir = "E:/yrt/D2/motiondetect/D02_20210926051056/"  # "E:/yrt/D3/motiondetect/D03_20210924122303/"
labels = ['head', 'stand', 'lie', 'sit', 'trough', 'door']

txt_total = []

# videoCapture = cv2.VideoCapture(imgdir)
# frame_num = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

txtdir = os.listdir(detectdir)
for txtname in txtdir:
    txtpath = os.path.join(detectdir, txtname)
    txt_total.append(txtpath)

cset = []  # 与地面差集
n = 0
m = 0
xcoor = []  # 确定地面区域
ycoor = []  # 确定地面区域
poslist = []
mdlist = []  # 存储头部位置变化特征(<30:0;>=30:1)

for i in range(len(txt_total)):  # len(img_total)
    ls_sign = 0
    headsign = 0  # 头部置信度存储（头部多识别，选择置信度较高的一个）
    bodysign = 0
    envisign = 0
    sline = 0  # 一个图像的标签数量（应该有4个：头部、身体姿态、食槽、门）
    h = 1329
    w = 1619
    with open(txt_total[i], "r+", encoding="utf-8", errors="ignore") as f:  # 遍历标签
        for line in f:
            sline += 1
            lineinfo = [float(a) for a in line.split(' ')]  # 按空格划分并转换float类型
            label = labels[int(lineinfo[0])]
            bbox = lineinfo[1:]  # 获取box信息
            # print(type(bbox[0]))
            (x1, y1, x2, y2) = convert(bbox, [h, w])
            if label == 'head' and headsign < lineinfo[5]:
                headsign = lineinfo[5]  # 头部置信度
                hx_center, hy_center, hwidth, hheight = convertcenter(bbox, [h, w])  # 头部中点和检测框高宽
    if i != 0:
        chxc, chyc, chw, chh = abs(hx_center - prehxc), abs(hy_center - prehyc), abs(hwidth - prehw), abs(hheight - prehh)
        # cbxc, cbyc, cbw, cbh = abs(bx_center - prebxc), abs(by_center - prebyc), abs(bwidth - prebw), abs(bheight - prebh)
        motiondis = sum([chxc, chyc, chw, chh])
        print(int(i/5), i, motiondis)
        if motiondis < 30:
            mdlist.append(0)
        else:
            mdlist.append(1)
        # print(i, cbxc, cbyc, cbw, cbh)
    prehxc, prehyc, prehw, prehh = hx_center, hy_center, hwidth, hheight

### 定位
## 秒级
mdsc = [0 for i in range(int((len(mdlist)+1)/5))]
for i in range(len(mdlist)):
    if mdlist[i] == 1:
        sc = int(i / 5)
        mdsc[sc] += 1
print("mdsc", mdsc)

print("l", len(mdsc))
## 定位
lcarch = []
# 简单定位（以0截断）
i = 0
while i < len(mdsc)-2:
    if mdsc[i] > 2:
        first = i + 1  # +1:从零开始
        if mdsc[i+1] > 2:
            while i < len(mdsc)-2 and mdsc[i+1] > 2:
                print('1', i)
                last = (i+1)+1
                i += 1
            lcarch.append([first, last])
            print('2', i)
        else:
            i += 1
    else:
        i += 1

# 连接（拱地停顿或误检）
print("lcarch", lcarch)
# archframe = []
# i = 0
exitsign = 1
while exitsign == 1:
    archframe = []
    i = 0
    exitsign = 0
    while i < len(lcarch):
        if i != len(lcarch)-1 and lcarch[i+1][0] - lcarch[i][1] <= 3:
            exitsign = 1
            archframe.append([lcarch[i][0], lcarch[i+1][1]])
            i += 2
        else:
            archframe.append(lcarch[i])
            i += 1
    lcarch = archframe
print("archframe", archframe)
# 删除少于4秒的疑似拱地
arch = []
for i in range(len(archframe)):
    if archframe[i][1] - archframe[i][0] >= 3:
        arch.append(archframe[i])
print("arch", arch)

keypl = len(arch)
print(keypl)
# 计算提取关键时序片段的总秒数，并将秒数转化为帧数进行下一阶段的拱地分类
seconds = 0
for p in arch:
    seconds += (p[1] - p[0] + 1)
print("seconds", seconds)
video_name = "E:/D2/yrt/D02_20210926051056.mp4"  # "E:/D3/yrt/D03_20210924122303.mp4"
videoCapture = cv2.VideoCapture(video_name)
img_path = 'E:/yrt/D2/keyframe_video/' + 'D02_20210926051056/'  # F:/location/img/D2/
mkdir(img_path)
success, frame = videoCapture.read()
i = 0
timeF = 1
j = 0
'''
while arch:  # 与拱地相似度高的片段是否都提取完
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        index = '%06d' % j
        frame = frame[81:1410, 1:1620]  # D1：img[81:1410, 1:1620] D2: [61:1390, 71:1690] D3: [121:1450, 1:1620]
        keypfirst = (arch[0][0]-1) * 25  # 第一个疑似拱地片段的第一帧
        keyplast = arch[0][1] * 25  # 第一个疑似拱地片段的最后一帧
        keypi = keypl - len(arch) + 1  # 存储片段序号
        if keypfirst < j <= keyplast:
            print(keypi, j)
            if j == keypfirst + 1:
                key_path = img_path + '/' + str(keypi)  # 每个关键时序片段分文件保存
                mkdir(key_path)
            save_image(frame, key_path + '/image', index)
            if j == keyplast:  # 该关键时序片段的每一帧都提取完删除
                del arch[0]
            print('save image:', i)
    success, frame = videoCapture.read()
'''
fsp = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
while arch:  # 与拱地相似度高的片段是否都提取完
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        index = '%06d' % j
        frame = frame[61:1390, 71:1690]  # D1：img[81:1410, 1:1620] D2: [61:1390, 71:1690] D3: [121:1450, 1:1620]
        keypfirst = (arch[0][0]-1) * 25  # 第一个疑似拱地片段的第一帧
        keyplast = arch[0][1] * 25  # 第一个疑似拱地片段的最后一帧
        keypi = keypl - len(arch) + 1  # 存储片段序号
        if keypfirst < j <= keyplast:
            print(keypi, j)
            if j == keypfirst + 1:
                keyvideo_path = img_path + "D02_20210926051056_" + "%06d" % j + '.mp4'  # 关键片段保存视频
                video_out = cv2.VideoWriter(keyvideo_path, fourcc, fsp, (len(frame[0]), len(frame)))
            video_out.write(frame)
            if j == keyplast:  # 该关键时序片段的每一帧都提取完删除
                del arch[0]
    success, frame = videoCapture.read()
video_out.release()
