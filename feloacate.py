import scipy.signal as sg
import os
import cv2
import math
import numpy as np
import xlwt
from numba import jit
import re

@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def mdm(magnitude, angle, a):
    fh = np.zeros(18)
    md = np.zeros(4)
    for idx in range(magnitude.shape[0]):
        for mag, ang in zip(magnitude[idx], angle[idx]):
            # 以a为0度重新计算运动方向
            if ang > a:
                ang -= a
            else:
                ang += (180-a)
            fh[int(ang // 10)] += mag/1000

    md[0] = int(fh[5] + fh[6] + fh[7] + fh[8] + fh[9] + fh[10] + fh[11] + fh[12])  # forward
    md[1] = int(fh[0] + fh[1] + fh[2] + fh[3] + fh[14] + fh[15] + fh[16] + fh[17])  # back
    md[2] = int(fh[4])  # left
    md[3] = int(fh[13])
    md = list(md)

    return md[0], md[1], md[2], md[3]

def cal_diff(head, box1):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:carea 头部不在地面区域的大小, h2b 头部与地面交集占头部比例
    """
    xmin, ymin, xmax, ymax = head
    xmin1, ymin1, xmax1, ymax1 = box1

    # 计算相交矩形
    xmin1 = max(xmin, xmin1)
    ymin1 = max(ymin, ymin1)
    xmax1 = min(xmax, xmax1)
    ymax1 = min(ymax, ymax1)

    jw1 = max(0, xmax1 - xmin1)
    jh1 = max(0, ymax1 - ymin1)
    area1 = jw1 * jh1

    hh = ymax - ymin
    hw = xmax - xmin
    harea = hh * hw
    carea = harea - area1
    h2b = round((harea-carea) / harea, 2)

    return carea, h2b

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

def hbdiffset(xcoor, ycoor, headcoor):
    '''
    计算头部与地面的差集
    :param xcoor: 食槽与门两点横坐标
    :param ycoor: 食槽与门两点纵坐标
    :param headcoor: 头部两点横纵坐标
    :return: carea 地面与头部差集
    '''
    # 确定地面区域(bx1, by1, bx2, by2)
    sxcoor = sorted(xcoor)
    sycoor = sorted(ycoor)
    bw = 980
    bh = 680
    if sxcoor[-1] > sycoor[-1]:
        bxcenter = int((sxcoor[2] - sxcoor[1]) / 2 + sxcoor[1])  # 地面横坐标中点
        y1center = (ycoor[1] - ycoor[0]) / 2 + ycoor[0]  # 食槽或门的纵坐标中点
        y2center = (ycoor[3] - ycoor[2]) / 2 + ycoor[2]
        bycenter = int((y2center - y1center) / 2 + y1center)  # 地面纵坐标中点
        bx1 = int(bxcenter - bw / 2)
        bx2 = int(bxcenter + bw / 2)
        by1 = int(bycenter - bh / 2)
        by2 = int(bycenter + bh / 2)
    else:  # 翻转
        x1center = (xcoor[1] - xcoor[0]) / 2 + xcoor[0]
        x2center = (xcoor[3] - xcoor[2]) / 2 + xcoor[2]
        bxcenter = int((x2center - x1center) / 2 + x1center)
        bycenter = int((sycoor[2] - sycoor[1]) / 2 + sycoor[1])
        bx1 = int(bxcenter - bh / 2)
        bx2 = int(bxcenter + bh / 2)
        by1 = int(bycenter - bw / 2)
        by2 = int(bycenter + bw / 2)
    board = [bx1, by1, bx2, by2]  # 地面左上点坐标与右下点坐标
    carea, h2b = cal_diff(headcoor, board)  # 地面与头部差集

    return carea, h2b

def headdir(headlist, bodylist, temp_middle, hsvbody):
    '''

    :param headlist:头部中心式坐标
    :param bodylist:身体中心式坐标
    :param temp_middle: 上一帧身体中点（横或纵坐标）
    :param hsvbody 身体光流
    :return:middle, rbx_center, rby_center 身体中点（横或纵坐标），身体靠近头部1/3处点
    '''
    hx_center, hy_center, hwidth, hheight = headlist
    bx_center, by_center, bwidth, bheight = bodylist

    # 判断'x''y'模式（母猪大致朝向）
    if hx_center <= bx_center and hy_center <= by_center:
        roix_center = int(bwidth / 3)
        roiy_center = int(bheight / 3)
    elif hx_center <= bx_center and hy_center > by_center:
        roix_center = int(bwidth / 3)
        roiy_center = int(bheight / 3) * 2
    elif hx_center > bx_center and hy_center <= by_center:
        roix_center = int(bwidth / 3) * 2
        roiy_center = int(bheight / 3)
    else:
        roix_center = int(bwidth / 3) * 2
        roiy_center = int(bheight / 3) * 2

    list_center = []  # 存储中线上点移动的距离
    if bwidth >= bheight:
        for j in range(body.shape[0]):
            list_center.append(hsvbody[j, roix_center][1])
        mode = 'x'
    else:
        for j in range(body.shape[1]):
            list_center.append(hsvbody[roiy_center, j][1])
        mode = 'y'

    # 定位中线上猪的位置
    l = len(list_center)  # 中线的像素长度
    # r_sl = int(min(bwidth, bheight) / 3)  # 身体高或宽中最短一边的1/3长度
    first = 0
    last = l
    # 中线上第一个有移动的像素点的索引
    for j in range(l):
        if list_center[j] > 10:
            first = j
            break
        if j == len(list_center) - 1:  # 恰好中线上没有发生移动
            first = 0
    # 中线上最后一个有点动的像素点的索引
    list_center.reverse()
    for j in range(l):
        if list_center[j] > 10:
            last = l - j - 1
            break
        if j == len(list_center) - 1:  # 恰好中线上没有发生移动
            last = len(list_center) - 1

    middle = int((last - first) / 2 + first)
    if temp_middle == 0:  # 第一帧
        temp_middle = middle
    middlediff = abs(temp_middle - middle)
    if middlediff > 25:
        middle = temp_middle
    if mode == 'x':
        rbx_center = roix_center
        rby_center = middle
    else:
        rbx_center = middle
        rby_center = roiy_center

    return middle, rbx_center, rby_center


flowdir = "E:/yrt/D2/keyflow/D02_20210926051056/"  # E:/behavior flow/dataenhance/ E:/behavior flow/temp/
flowfile = os.listdir(flowdir)
detectdir = "E:/yrt/D2/keydetect/D02_20210926051056/"  # E:/behavior detect/dataenhance/ F:/arch/imgdect/dataenhance/
detectfile = os.listdir(detectdir)
labels = ['head', 'stand', 'lie', 'sit', 'trough', 'door']
# w = 1619
# h = 1329
k = 0
workbook = xlwt.Workbook(encoding='utf-8')  # 新建工作簿
sheet1 = workbook.add_sheet("频率计算", cell_overwrite_ok=True)  # 新建sheet
sheet1.write(0, 0, "关键时序片段序号")  # 第1行第1列数据
sheet1.write(0, 1, "4秒分割序号")  # 第1行第2列数据
sheet1.write(0, 2, "频率")  # 第1行第4列数据
sheet1.write(0, 3, "站姿占比")  # 第1行第4列数据
sheet1.write(0, 4, "头部在地面比例")  # 第1行第4列数据
# sheet1.write(0, 5, "置信度方差")  # 第1行第4列数据
workbook.save(r'E:/yrt/D2/locate_result/0926051056.xls')  # 保存

for p in range(len(flowfile)):  # len(flowfile)
    print(p + 1)
    flowpath = os.path.join(flowdir, flowfile[p])  # flow path
    print(flowpath)
    pathdetect = os.path.join(detectdir, detectfile[p])  # detect path
    print(pathdetect)
    img_total = []
    txt_total = []

    imgdir = os.listdir(flowpath)
    for imgname in imgdir:
        imgpath = os.path.join(flowpath, imgname)
        img_total.append(imgpath)

    txtdir = os.listdir(pathdetect)
    txti = 0
    for txtname in txtdir:
        txtpath = os.path.join(pathdetect, txtname)
        txt_total.append(txtpath)
        '''
        if txti != 0:    # 光流图少第一帧
            txt_total.append(txtpath)
        else:
            txti = 1
        '''

    forward = []
    back = []
    left = []
    right = []
    maindire = []
    # direlist = []
    ds = []
    sz = []
    cset = []  # 与地面差集
    n = 0
    m = 0
    parall_num = 0
    temp_middle = 0
    lslist = []  # posture
    pslist = []  # 姿态置信度变化值
    for i in range(len(img_total)):  # len(img_total)
        ls_sign = 0
        img = cv2.imread(img_total[i])
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgsat = hsvimg[..., 1]
        flowelem = np.sum(imgsat)  # 图像所有像素点运动幅度
        headsign = 0  # 头部置信度存储（头部多识别，选择置信度较高的一个）
        bodysign = 0
        tgsign = 0
        drsign = 0
        sline = 0  # 一个图像的标签数量（应该有4个：头部、身体姿态、食槽、门）
        xcoor = []  # 确定地面区域
        ycoor = []  # 确定地面区域
        if i == 0:
            h, w = img.shape[0:2]
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
                    head = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]  # 头
                    headcoor = [x1, y1, x2, y2]
                elif label == 'stand' or label == 'sit' or label == 'lie' and bodysign < lineinfo[5]:
                    bodysign = lineinfo[5]
                    bx_center, by_center, bwidth, bheight = convertcenter(bbox, [h, w])  # 身体中点和检测框高宽
                    body = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]  # 身体
                    blefttopx = x1
                    blefttopy = y1
                    hsvbody = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
                    bodysat = hsvbody[..., 1]
                    bodyelem = np.sum(bodysat)  # 身体检测框内像素点运动幅度
                    if label == 'lie' or label == 'sit':
                        ls_sign = -1
                    else:
                        ls_sign = 1
                elif label == 'trough' and tgsign < lineinfo[5]:
                    envisign = lineinfo[5]
                    xcoor.append(x1)
                    xcoor.append(x2)
                    ycoor.append(y1)
                    ycoor.append(y2)
                elif label == 'door' and drsign < lineinfo[5]:
                    envisign = lineinfo[5]
                    xcoor.append(x1)
                    xcoor.append(x2)
                    ycoor.append(y1)
                    ycoor.append(y2)
        pslist.append(bodysign)
        if i != 0 and (tgsign == 0 or drsign == 0):
            xcoor = xcoor1
            ycoor = ycoor1
        xcoor1 = xcoor  # 防止食槽和门漏识别
        ycoor1 = ycoor
        lslist.append(ls_sign)
        if headsign == 0:  # 防止头部未识别
            carea = 0
            cset.append(cset[-1])  # 复制上一帧结果
        else:
            # print(txt_total[i], xcoor, ycoor)
            carea, boardc = hbdiffset(xcoor, ycoor, headcoor)
            if carea >= 3000 or boardc < 0.75:
                cset.append(-1)  # 存储头部是否位于关键区域，与地面差集小于3000则在反之不在
            else:
                cset.append(1)

        flownum = (flowelem - bodyelem) / 1000000  # 背景像素运动幅度总和

        # 判断母猪身体检测框中点
        headlist = [hx_center, hy_center, hwidth, hheight]
        bodylist = [bx_center, by_center, bwidth, bheight]
        middle, rbx_center, rby_center = headdir(headlist, bodylist, temp_middle, hsvbody)

        temp_middle = middle
        if i != 0 and flownum > 5:  # 坐姿、卧姿或者背景运动幅度过大（母猪运动幅度小）
            forward.append(0)
            back.append(0)
            left.append(0)
            right.append(0)
        else:
            # 绝对位置（rbx_center、rby_center是身体区域的相对位置）
            rbx_center = rbx_center + blefttopx
            rby_center = rby_center + blefttopy

            # 计算两点之间的倾斜角
            Tilt_angle = math.atan2(rby_center - hy_center, rbx_center - hx_center)
            # 转换为角度
            Tilt_angle = int(Tilt_angle * 180 / math.pi)
            # print('Tilt_angle', Tilt_angle)
            if Tilt_angle <= 0:
                Tilt_angle = Tilt_angle + 360
                Tilt_angle = Tilt_angle / 2
            else:
                Tilt_angle = Tilt_angle / 2

            flowhead = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
            hhue = flowhead[..., 0]
            hsaturation = flowhead[..., 1]
            fw, bk, lf, rg = mdm(hsaturation, hhue, Tilt_angle)
            # print(fw, bk, lf, rg)
            # singledir = np.array([fw, bk, lf, rg])
            # ind = singledir.argmax()
            forward.append(fw)
            back.append(bk)
            left.append(lf)
            right.append(rg)

        n = n + 1

    archsign = [0 for i in range(len(img_total))]  # 初始化：每帧都属于非拱地行为
    direlist = []
    for i in range(len(forward)):
        if forward[i] > back[i]:
            dire = 1
            direlist.append(1)
        elif forward[i] == back[i]:  # forward[i] == 0 and back[i] == 0
            dire = 0
            direlist.append(0)
        else:
            dire = -1
            direlist.append(-1)

    ### 拱地次数序列
    for i in range(len(direlist) - 2):
        if direlist[i] == 1:
            if direlist[i + 1] == -1:
                archsign[i+1] = 1
        i = i + 5  # 波动误差（一秒25帧）

    ### 每四秒保存一次特征
    ## 确定每个片段的开始1、5、9、13、17、21……
    # 片段开头的处理
    temptn = re.split('\W', img_total[p])[-4].split('_')[2]
    print(temptn)
    timename = int(int(temptn) / 25) + 1  # \D数字字符
    print('timename', timename)
    moresec = (timename - 1) % 4
    piank = 1
    timename_ft = int((timename-1) / 4) * 4 + 1
    if moresec == 0:
        k += 1
        archnum = sum(archsign[0:99])  # 四秒内拱地次数
        standnum = sum(lslist[0:99])  # 四秒内站姿帧数
        headonfr = sum(cset[0:99])  # 四秒内头部在关键区域的帧数

        print("识别次数", archnum)
        ratios = archnum / 4
        print('重复运动频率', ratios)
        posratio = standnum / 99
        print('站姿占比', posratio)
        canratio = headonfr / 99
        print('头部在地面比例', canratio)
        time = '%05d' % timename_ft

        sheet1.write(k, 0, p + 1)  # 第2行第1列数据  第几片段
        sheet1.write(k, 1, time)  # 第2行第2列数据  第几秒
        sheet1.write(k, 2, ratios)  # 第2行第3列数据
        sheet1.write(k, 3, posratio)  # 第2行第3列数据
        sheet1.write(k, 4, canratio)  # 第2行第3列数据
        # sheet1.write(k, 5, arrconv)  # 第2行第3列数据
        workbook.save(r'E:/yrt/D2/locate_result/0926051056.xls')  # 保存

    elif moresec == 1:
        k += 1
        archnum = sum(archsign[0: 74])  # 四秒内拱地次数
        standnum = sum(lslist[0: 74])  # 四秒内站姿帧数
        headonfr = sum(cset[0: 74])  # 四秒内头部在关键区域的帧数

        print("识别次数", archnum)
        ratios = archnum / 3
        print('重复运动频率', ratios)
        posratio = standnum / 74
        print('站姿占比', posratio)
        canratio = headonfr / 74
        print('头部在地面比例', canratio)
        time = '%05d' % timename_ft

        sheet1.write(k, 0, p + 1)  # 第2行第1列数据  第几片段
        sheet1.write(k, 1, time)  # 第2行第2列数据  第几秒
        sheet1.write(k, 2, ratios)  # 第2行第3列数据
        sheet1.write(k, 3, posratio)  # 第2行第3列数据
        sheet1.write(k, 4, canratio)  # 第2行第3列数据
        # sheet1.write(k, 5, arrconv)  # 第2行第3列数据
        workbook.save(r'E:/yrt/D2/locate_result/0926051056.xls')  # 保存
    ## 片段第二段往后
    for i in range((4-moresec)*25, len(img_total) - 99, 100):  # 光流图少一帧
        k += 1
        piank += 1
        archnum = sum(archsign[i:i + 100])  # 四秒内拱地次数
        standnum = sum(lslist[i:i + 100])  # 四秒内站姿帧数
        headonfr = sum(cset[i:i + 100])  # 四秒内头部在关键区域的帧数

        print(k, p, i + 1)
        print("识别次数", archnum)
        ratios = archnum / 4
        print('重复运动频率', ratios)
        posratio = standnum / 100
        print('站姿占比', posratio)
        canratio = headonfr / 100
        print('头部在地面比例', canratio)
        timename = (piank-1)*4 + timename_ft
        time = '%05d' % timename

        sheet1.write(k, 0, p + 1)  # 第2行第1列数据  第几片段
        sheet1.write(k, 1, time)  # 第2行第2列数据  第几秒
        sheet1.write(k, 2, ratios)  # 第2行第3列数据
        sheet1.write(k, 3, posratio)  # 第2行第3列数据
        sheet1.write(k, 4, canratio)  # 第2行第3列数据
        # sheet1.write(k, 5, arrconv)  # 第2行第3列数据
        workbook.save(r'E:/yrt/D2/locate_result/0926051056.xls')  # 保存