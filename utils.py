
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm



def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)        (..., 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)    (..., 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2     # (..., 1)
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2     # (..., 1)
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2     # (..., 1)
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2     # (..., 1)
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2   # (..., 1)
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2   # (..., 1)
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2   # (..., 1)
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2   # (..., 1)

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]    # (..., 1)
        box1_y1 = boxes_preds[..., 1:2]    # (..., 1)
        box1_x2 = boxes_preds[..., 2:3]    # (..., 1)
        box1_y2 = boxes_preds[..., 3:4]    # (..., 1)
        box2_x1 = boxes_labels[..., 0:1]   # (..., 1)
        box2_y1 = boxes_labels[..., 1:2]   # (..., 1)
        box2_x2 = boxes_labels[..., 2:3]   # (..., 1)
        box2_y2 = boxes_labels[..., 3:4]   # (..., 1)

    x1 = torch.max(box1_x1, box2_x1)   # (..., 1)   # torch.max()不改变形状
    y1 = torch.max(box1_y1, box2_y1)   # (..., 1)
    x2 = torch.min(box1_x2, box2_x2)   # (..., 1)
    y2 = torch.min(box1_y2, box2_y2)   # (..., 1)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)        # (..., 1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))    # (..., 1)
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))    # (..., 1)

    return intersection / (box1_area + box2_area - intersection + 1e-6)    # (..., 1)



def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors, device):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor_for_scale = torch.tensor(anchors[i]) * S
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor_for_scale, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def iou_width_height(boxes1, boxes2):
    # boxes1: (..., 2)  2->(w1, h1)
    # boxes2: (..., 2)  2->(w2, h2)
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])  # (...) * (...) = (...)
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection  # (...) * (...) + (...) * (...) - (...) = (...)
    return intersection / union




def plot_image(image, boxes):
    # image: tensor (H, W, C)
    # boxes: list of lists  [box1, box2,...,boxn]   boxi = [class_label, conf, x_cell, y_cell, w_cell, h_cell]
    cmap = plt.get_cmap("tab20b")    # 从matplotlib库中选取一种颜色映射(一种将数值映射到颜色的机制)，名为“tab20b”，“tab20b”是一种包含20种不同颜色的离散型颜色映射
    class_labels = config.COCO_LABELS if config.DATASET == "coco" else config.PASCAL_CLASSES   # 选择类别标签
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]   # [cmap(0/20), cmap(1.1/20), cmap(2.2/20), ..., cmap(20/20)]
    im = np.array(image)    # 将tensor图片转换为numpy数组图片
    height, width, _ = im.shape    # (H, W, C)
    fig, ax = plt.subplots(1)   # 只建造一个子图，返回图窗和坐标轴
    ax.imshow(im)   # 在坐标轴上将图片显示出来

    # 遍历boxes中的每个box将其在image上显示出来
    for box in boxes:    # box = [class_label, conf, x_cell, y_cell, w_cell, h_cell]
        assert len(box) == 6
        class_label = int(box[0])    # 0,1,2,3,4,..,19
        conf = round(box[1], 3)   # 四舍五入保留三位小数
        x_cell, y_cell, w_cell, h_cell = box[2:]
        # 画矩形框需要左上角坐标
        upper_left_x = (x_cell - w_cell / 2) * width   # 归一化左上角坐标 -> 左上角实际像素坐标
        upper_left_y = (y_cell - h_cell / 2) * height  # 归一化左上角坐标 -> 左上角实际像素坐标
        rect = patches.Rectangle((upper_left_x, upper_left_y), w_cell * width, h_cell * height, linewidth=2, edgecolor=colors[class_label], facecolor="none")   # facecolor="none"才是没有填充而不是facecolor=None
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            upper_left_y,
            s=class_labels[class_label]+str(conf),
            color="white",                        # color="white"  设置文本的颜色为白色
            verticalalignment="top",              # 垂直对齐方式为"top"，设置文本的顶部对齐upper_left_y
            bbox={"color": colors[class_label], "pad": 0}   # 用于在文本周围绘制一个背景矩形，背景矩形的颜色与矩形框一样，pad为0表示文本到矩形边缘之间的部分不填充，也就是只填充文本部分，超出文本的部分不填充
        )

    plt.show()




# 将来自prediction和target的目标框信息转换为列表形式，便于进行NMS和画图操作
def cells_to_bboxes(prediction, anchors_for_scale, S, is_preds=True):

    """
    将模型预测prediction和数据集中标签target中各个scale的输出转换为相对于图像归一化的列表(BS, 3*S*S, 6)  6:[class_label, conf, x_image, y_image, w_image, h_image]

    predictions: one scale tensor (BS, 3, S, S, 25) for prediction or (BS, 3, S, S, 6) for target
    anchors_for_scale: tensor (3, 2)   ((,),(,),(,))  必须是相对于cell大小的anchors,如果是相对于图像大小的anchors要乘以S
    S: could be None
    is_preds: prediction is predition or target

    return: lists [BS, 3*S*S, 6]
    """
    if is_preds:
        # prediction: (BS, 3, S, S, 25)   25: t_conf, t_x, t_y, t_w, t_h, class1_prob, class2_prob, ..., class20_prob
        anchors_for_scale = anchors_for_scale.reshape(1, 3, 1, 1, 2)     # 注意这里的anchors_for_scale对应的锚框必须是相对于cell的，如果是原本的anchors相对于整张图片的在放入这个函数前要乘以S
        conf = torch.sigmoid(prediction[..., 0:1])    # (BS, 3, S, S, 1)    conf = sigmoid(t_conf)
        x_cell = torch.sigmoid(prediction[..., 1:2])   # (BS, 3, S, S, 1)   x_cell = sigmoid(t_x)
        y_cell = torch.sigmoid(prediction[..., 2:3])   # (BS, 3, S, S, 1)   y_cell = sigmoid(t_y)
        w_cell = anchors_for_scale[..., 0:1] * torch.exp(prediction[..., 3:4])   # (1, 3, 1, 1, 1) * (BS, 3, S, S, 1) -> (BS, 3, S, S, 1)  w_cell = pw * exp(t_w)
        h_cell = anchors_for_scale[..., 1:2] * torch.exp(prediction[..., 4:5])   # (1, 3, 1, 1, 1) * (BS, 3, S, S, 1) -> (BS, 3, S, S, 1)  h_cell = ph * exp(t_h)
        class_label = torch.argmax(prediction[..., 5:], dim=-1).unsqueeze(-1)    # (BS, 3, S, S, 20) -> (BS, 3, S, S) -> (BS, 3, S, S, 1)

    else:
        # prediction=target: (BS, 3, S, S, 6)  6: conf, x_cell, y_cell, w_cell, y_cell, class_label
        conf = prediction[..., 0:1]
        x_cell = prediction[..., 1:2]
        y_cell = prediction[..., 2:3]
        w_cell = prediction[..., 3:4]
        h_cell = prediction[..., 4:5]
        class_label = prediction[..., 5:6]

    BS = prediction.shape[0]
    # 新创造的张量要与已有的张量放在同一个device上
    x_cell_indices = torch.arange(S).repeat(BS, 3, S, 1).unsqueeze(-1).to(prediction.device)        # (BS, 3, S, S)->(BS, 3, S, S, 1)  # don't forget to(prediction.device)
    y_cell_indices = x_cell_indices.permute(0, 1, 3, 2, 4)       # (BS, 3, S*, S#, 1)->(BS, 3, S#, S*, 1)
    x_image = (x_cell_indices + x_cell) / S    # (BS, 3, S, S, 1)
    y_image = (y_cell_indices + y_cell) / S    # (BS, 3, S, S, 1)
    w_image = w_cell / S                       # (BS, 3, S, S, 1)
    h_image = h_cell / S                       # (BS, 3, S, S, 1)
    converted_bboxes = torch.cat((class_label, conf, x_image, y_image, w_image, h_image), dim=-1).reshape(BS, 3*S*S, 6)   # (BS, 3, S, S, 6)->(BS, 3*S*S, 6)
    return converted_bboxes.tolist()     # tensor (BS, 3*S*S, 6) -> lists of list [batch0[ 0[class_label, conf, x_image, y_image, w_image, h_image], 1[],..., 3*S*S-1[]],batch1[],...batchN-1[]]




def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    """
    将loader中所有图片检测出的目标框和真实目标框转换成目标框列表all_pred_boxes和all_true_boxes

    anchors: 这里的anchors就是原始的没有任何变化的anchors list [3,3,2]
    iou_threshold和threshold会用来进行NMS操作，box_format也是NMS中指定求IOU的坐标格式

    return:
    all_pred_boxes = [box0, box1, box2, ...]   其中boxi = [train_idx, class, conf, x_image, y_image, w_image, h_image] train_idx表示这个box在第几张图片上
    all_true_boxes = [box0, box1, box2, ...]   其中boxi = [train_idx, class, conf, x_image, y_image, w_image, h_image]
    """
    model.eval()   # 将模型设置为验证模式
    train_idx = 0  # 表示第几张图片，从第0张图片开始
    all_pred_boxes = []
    all_true_boxes = []

    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        # x : (BS, 3, H, W)
        # labels: [(BS, 3, 13, 13, 6), (BS, 3, 26, 26, 6), (BS, 3, 52, 52, 6)]
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)   # predictions: [(BS, 3, 13, 13, 25), (BS, 3, 26, 26, 25), (BS, 3, 52, 52, 25)]

        batch_size = x.shape[0]
        true_boxes = cells_to_bboxes(labels[2], anchors_for_scale=torch.tensor(anchors[2]).to(device) * (labels[2].shape[2]), S=labels[2].shape[2], is_preds=False)
        # true_boxes: lists of list  [BS, 3*52*52, 6]    6:class, conf, x_image, y_image, w_image, h_image
        pred_boxes = [[] for _ in range(batch_size)]    # 存放每张图像对应的所有box [第0张图像[[],[],[],...,[]],第1张图像[],第2张图像[],...,第BS-1张图像[]]

        for scale_idx in range(3):    # scale_idx表示第几个scale   取值为0,1,2
            S = predictions[scale_idx].shape[2]
            anchors_for_scale = torch.tensor(anchors[scale_idx]).to(device) * S   # 将对应scale的anchors提取出来并缩放为相对于cell归一化的
            boxes_scale_i = cells_to_bboxes(predictions[scale_idx], anchors_for_scale=anchors_for_scale, S=S, is_preds=True)
            # boxes_scale_i: lists of list [BS, 3*S*S, 6]   6: [class_label, conf, x_image, y_image, w_image, h_image]
            for image_idx, boxes in enumerate(boxes_scale_i):    # image_idx 图片标号，表示第几张图片，boxes是这张图片上所有box，共3*S*S个box
                pred_boxes[image_idx] += boxes     # boxes:[[],[],...,[]共3*S*S个box]

        # pred_boxes: [BS, 3*13*13+3*26*26+3*52*52, 6]  6:class, conf, x_image, y_image, w_image, h_image


        for idx in range(batch_size):  # idx表示图像的序号，这里表示对这个batch中第idx张图像进行操作
            nms_boxes = non_max_suppression(pred_boxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)  # 对第idx张图片上的所有box进行NMS
            # pred_boxes[idx]  [3*13*13+3*26*26+3*52*52, 6]  [[],[],[],[],...,[]共有3*13*13+3*26*26+3*52*52个box，每一个box有6个元素]
            # nms_boxes [<3*13*13+3*26*26+3*52*52, 6]
            for box in nms_boxes:
                all_pred_boxes.append([train_idx]+box)   # [train_idx]+box   [train_idx, class, conf, x_image, y_image, w_image, h_image]
                # all_pred_boxes [[],[],...,[]] 每个box[]中有7个元素


            for box in true_boxes[idx]:    # true_boxes[idx]:  [3*52*52, 6]  [[],[],[],...,[]共3*52*52个box， 每个box有6个元素]
                if box[1] > threshold:
                    all_true_boxes.append([train_idx]+box)    # [train_idx, class, conf, x_image, y_image, w_image, h_image]
                    # all_true_boxes [[],[],...,[]] 每个box[]中有7个元素

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def non_max_suppression(bboxes, iou_threshold=0.3, threshold=0.5, box_format="midpoint"):
    """
    根据给定的iou_threshold和threshold进行NMS操作

    bboxes:包含一张图片上所有的box  lists of list [box0, box1, ...]  其中每个boxi=[class, conf, x_image, y_image, w_image, h_image]
    threshold: 置信度低于threshold的box直接丢弃，用于第一遍筛选
    iou_threshold: 如果是同一类别的box，如果它与那个置信度较高的chosen_box的IOU低于iou_threshold才会保留，否则也会被丢弃
    box_format:指定求解IOU的坐标格式

    return:返回经过NMS后的这张图片上保留的box列表[nms_box0, nms_box1, ...] 其中每个nms_boxi=[class, conf, x_image, y_image, w_image, h_image]
    """
    assert type(bboxes) == list
    bboxes_after_nms = []    # 用于保存经过NMS的所有box
    bboxes = [box for box in bboxes if box[1] > threshold]  # 利用置信度threshold对这张图片上的所有box进行第一遍筛选
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)   # 将所有box按照置信度conf由高到低排序


    while bboxes:
        chosen_box = bboxes.pop(0)

        # 如果box的类别与chosen_box不同，box保留
        # 如果box的类别与chosen_box相同，但是IOU < iou_threshold，box保留
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(box[2:]), torch.tensor(chosen_box[2:]), box_format=box_format) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def mean_average_precision(
    all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    计算所有类的mAP

    all_pred_boxes:数据集中所有图片预测得到的box [box0, box1, box2, ...]   其中boxi = [train_idx, class, conf, x_image, y_image, w_image, h_image] train_idx表示这个box在第几张图片上
    all_true_boxes:数据集中所有图片标签box [box0, box1, box2, ...]   其中boxi = [train_idx, class, conf, x_image, y_image, w_image, h_image]
    iou_threshold:用于判断一个检测是否是FP，如果一个检测与最匹配的GT的IOU小于这个阈值iou_threshold，那这个检测是FP
    box_format:用于计算IOU的坐标格式
    num_classes:类别总数

    return:所有类的mAP
    """
    average_precision = []   # 用于保存所有类别的AP  [AP1, AP2, AP3, ...,AP20]
    epsilon = 1e-6   # 为了保持数值稳定性，防止除0

    # 遍历每个类别，求每个类的AP
    for c in range(num_classes):
        detections = []        # 用于保存这个类别c的所有检测出的box
        ground_truths = []     # 用于保存这个类别c的所有真实box


        # 筛选出这个类别c的所有预测box和真实box
        for detection in all_pred_boxes:
            if detection[1] == c:
                detections.append(detection)


        for true_box in all_true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        amount_gt = Counter([gt[0] for gt in ground_truths])
        # {0:3, 1:5, ...}表示第0张图片上有这个类别的3个gt,第1张图片上有这个类别的5个gt...

        for key, value in amount_gt.items():
            amount_gt[key] = torch.zeros(value)

        # amount_gt:{0:tensor[0,0,0], 1:tensor[0,0,0,0,0], ...}用于后续标记这个gt是否已经与某个TP匹配


        detections.sort(key=lambda x: x[2], reverse=True)    # 将这个类别的所有检测的box按置信度从大到小进行排序
        TP = torch.zeros((len(detections)))  # 设置TP
        FP = torch.zeros((len(detections)))  # 设置FP
        total_gt_num = len(ground_truths)  # 这个类别的所有gt的总数，用于求recall=TP/all_GT_num
        if total_gt_num == 0:   # 如果没有这个类别的目标，则直接跳过这个类别
            continue

        # 遍历这个类别的每一个detection，区分是TP还是FP
        # 找到这个detection所在的image,找出这个image上的这个类别的所有gt,
        # 根据IOU找出这些gt中与detection最匹配的gt
        # 如果这个最好的IOU<iou_threshold，则直接把detection归为FP
        # 如果这个最好的IOU>iou_threshold
        # 并且这个gt没有与其他TP相匹配，则把这个detection归为TP，否则把detection归为FP
        for detection_idx, detection in enumerate(detections):
            ground_truths_img = [gt for gt in ground_truths if gt[0] == detection[0]]   # 取出与detection同张图片上的所有gt

            best_iou = 0.0
            best_gt_idx = 0
            # 遍历这张图片上这个类别的所有gt，找出与这个detection最匹配的gt
            for gt_idx, gt in enumerate(ground_truths_img):
                IOU = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if IOU > best_iou:
                    best_iou = IOU
                    best_gt_idx = gt_idx

            # 根据条件判断detection是否为TP还是FP
            if best_iou > iou_threshold:
                if amount_gt[detection[0]][best_gt_idx] == 0:
                    amount_gt[detection[0]][best_gt_idx] = 1
                    TP[detection_idx] = 1

                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)          # cumsum累计和
        FP_cumsum = torch.cumsum(FP, dim=0)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum+epsilon)
        recall = TP_cumsum / (total_gt_num + epsilon)
        precision = torch.cat([torch.tensor([1]), precision])
        recall = torch.cat([torch.tensor([0]), recall])
        AP = torch.trapz(precision, recall)    # 求precision(y)-recall(x)曲线下的面积
        average_precision.append(AP)   # 将此类的AP加入average_precision

    return sum(average_precision) / len(average_precision)    # 返回所有类别的mAP



def check_class_accuracy(model, loader, threshold):
    """
    计算并打印经过训练的model在测试数据集上的表现，如分类准确度class_accuracy,obj置信度准确度,noobj置信度准确度
    model:经过训练的模型
    loader:测试数据加载器
    threshold:区分obj/noobj置信度的阈值，如果模型输出的box的置信度大于threshold直接将置信度设置为1，小于threshold直接将置信度设置为0
    """
    model.eval()  # 将模型设为验证模式，batchnorm采用所有batch，drop取消改为采用整个模型，没有梯度计算
    total_class_preds, correct_class = 0, 0  # 总的分类box个数， 分类正确的box个数
    total_obj, correct_obj = 0, 0   # obj的box的总数，obj中预测置信度正确的box总数，obj的box希望预测置信度为1
    total_noobj, correct_noobj = 0, 0  # noobj的box总数，noobj中预测置信度正确的总数，noobj的box希望预测置信度为0
    for idx, (x, y) in enumerate(tqdm(loader)):
        # x(imgs): (BS, 3, H, W)
        # y(labels):[(BS, 3, 13, 13, 6),(BS, 3, 26, 26, 6),(BS, 3, 52, 52, 6)]  6->[conf, x_cell, y_cell, w_cell, h_cell, class]
        x = x.to(config.DEVICE)
        # y = y.to(config.DEVICE)  'list' object has no attribute 'to'

        with torch.no_grad():
            out = model(x)   # out:[(BS, 3, 13, 13, 25),(BS, 3, 26, 26, 25),(BS, 3, 52, 52, 25)]

        for i in range(3):   # 遍历每个scale
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1   #  (BS, 3, S, S)  哪些scale的哪些cell的哪些ancTruehor是要预测box的，将其设为True，放在obj中
            noobj = y[i][..., 0] == 0  #  (BS, 3, S, S)   那些不需要预测box的anchor设为True，放在noobj中

            correct_class += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj])
            # out[i][..., 5:]:(BS, 3, S, S, 20)   ->  out[i][..., 5:][obj]:(obj中True总数, 20)  ->  torch.argmax(out[i][..., 5:][obj], dim=-1):(obj中True总数)
            # y[i][..., 5]:(BS, 3, S, S)  -> y[i][..., 5][obj]:(obj中True总数)
            total_class_preds += torch.sum(obj)

            conf_pred = torch.sigmoid(out[i][..., 0]) > threshold     # sigmoid(t_conf)  (BS, 3, S, S)
            # > threshold操作相当于将大于threshold的置信度设置为1，小于threshold的置信度设置为0

            correct_obj += torch.sum(conf_pred[obj] == y[i][..., 0][obj])
            # conf_pred:(BS, 3, S, S)  -> conf_pred[obj]:(obj中True的总数)  取值True/False相当于1/0
            # y[i][..., 0][obj]:(obj中True的总数)  取值1/0
            total_obj += torch.sum(obj)

            correct_noobj += torch.sum(conf_pred[noobj] == y[i][..., 0][noobj])
            total_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(total_class_preds+1e-6))*100 :2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(total_noobj+1e-16))*100 :2f}%")
    print(f"Obj accuracy is: {(correct_obj/(total_obj+1e-6))*100 :2f}%")
    model.train()







