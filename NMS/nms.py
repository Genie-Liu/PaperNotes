import numpy as np
 
 
def nms(boxes, threshold):
    # 得到每一个box的左上和右下的坐标值，以及每一个box的概率值
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
 
    # 计算每一个box的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对每一个box按照score降序排列
    order = scores.argsort()[::-1]
 
    # 最后保留box的集合
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类box中概率值最大的一个
 
        # 相交区域的左上和右下的坐标值
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
 
        # 相交区域的宽和高度，计算相交区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
 
        # 计算IoU值
        over = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        index = np.where(over <= threshold)[0]
        # np.where(condition)返回索引
        order = order[index + 1]
 
    return keep

if __name__ == "__main__":
    boxes = [(0, 0, 20, 50, 0.8), (0, 0, 15, 40, 0.9)]
    threshold = 0.5
    boxes = np.array(boxes)
    print(nms(boxes, threshold))