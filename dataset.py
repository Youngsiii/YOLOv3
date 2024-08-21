import torch
import os
import pandas as pd
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import iou_width_height
from utils import cells_to_bboxes
from utils import non_max_suppression as nms
from utils import plot_image

ImageFile.LOAD_TRUNCATED_IMAGES = True



class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        super(YOLODataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        # img    label
        # xx0.jpg  xx0.txt
        # xx1.jpg  xx1.txt
        # ...
        # self.annotations的第一行是从xx0.jpg开始的，pd.read_csv会自动把第一行当作head
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])   # (9,2) list -> (9,2) tensor
        self.image_size = image_size
        self.S = S
        self.C = C
        self.transform = transform
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3  # 3
        self.ignore_iou_thresh = 0.5   # 如果某个anchor与box的iou比较大超过这个阈值，但是box已经在这个scale上分配过，就将这个anchor置信度设置为-1


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist()   # 以000001.txt为例，(2,5)
        # np.loadtxt()从文本文件加载数据到numpy数组，读取fname指定的文本文件并根据参数将数据转换为数组
        # delimiter为分隔符，这里指定为空格，表示loadtxt将每一行按照空格分隔的值读取为数组中的元素
        # ndmin为返回数组的最小维度，ndmin=2表示返回的数组至少是2维的，即使只有一行数据，也会返回二维数组，第一个维度表示行，第二个维度表示列
        # np.roll 将数组进行滚动，沿axis=1正方向移动4个位置  (c,x,y,w,h)->(x,y,w,h,c)  方便后续transform
        # 最后通过tolist()将numpy数组转换为列表
        # bboxes中每个box=(x,y,w,h,c)中的x,y,w,h都是相对于整张图片归一化得到的

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")    # PIL图片格式
        image = np.array(image)  # 将PIL图片转换为numpy数组图片为后续的transform做准备

        if self.transform is not None:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.S]
        # targets = [[torch.zeros((3, 13, 13, 6))],   # -> scale 0  第一个S表示y轴，第二个S表示x轴
        #            [torch.zeros((3, 26, 26, 6))],   # -> scale 1
        #            [torch.zeros((3, 52, 52, 6))]]   # -> scale 2

        # # 将这张图片上的所有box信息赋值到targets，将每一个box分配到三个scale上，并且每个scale只有一个cell上的一个anchor保存(预测)box信息
        # for box in bboxes:
        #     x, y, width, height, class_label = box   # x,y,width,height都是相对于整张图像归一化的
        #     # 有x,y就能知道分配到哪个cell上(也就是哪个cell应该预测这个box)，但是分配给这个cell的哪个anchor呢(但是这个cell上的哪个anchor预测这个box呢)？
        #     # 通过计算box与每个anchor的iou,每个scale上iou最大的anchor就被分配预测这个box
        #     iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)   # (9)
        #     anchor_indices = iou_anchors.argsort(descending=True, dim=0)   # iou降序排序，然后返回排完序的索引例如tensor([0, 1, 5, 2, 4, 3, 8, 7, 6])
        #     use_scale = [False, False, False]   # 记录这个box已经在哪些scale上分配过了
        #
        #
        #     for anchor_idx in anchor_indices:
        #         scale_idx = anchor_idx // self.num_anchors_per_scale   # 这个anchor在哪个scale
        #         anchor_idx_on_scale = anchor_idx % self.num_anchors_per_scale   # 这个anchor是scale_idx的哪个anchor
        #         S = self.S[scale_idx]  # 获得这个scale的网格大小，即S×S个cells
        #         i, j = int(x * S), int(y * S)  # box由这个scale上的(i,j)cell来负责预测，因为box的中心落在这个cell内部
        #         anchor_taken = targets[scale_idx][anchor_idx_on_scale, j, i, 0]  # 获得这个scale这个cell这个anchor的confidence,显示这个anchor是否要预测其他box
        #         if use_scale[scale_idx] == False and anchor_taken == 0:  # 如果box还没有在这个scale上赋值过，并且这个最相关的anchor也不需要预测其他box，则可以让这个anchor来预测这个box
        #             use_scale[scale_idx] = True    # 设置这个box已在这个scale上赋值过
        #             targets[scale_idx][anchor_idx_on_scale, j, i, 0] = 1  # 设置这个anchor(这个scale上这个cell上这个anchor)预测这个box
        #             x_cell, y_cell = x * S - i, y * S - j   # 这个box相对于cell的x,y
        #             width_cell, height_cell = width * S, height * S  # 这个box相对于cell的w,h
        #             box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        #             # 下面是赋值操作，让这个anchor负责预测这个box
        #             targets[scale_idx][anchor_idx_on_scale, j, i, 1:5] = box_coordinates
        #             targets[scale_idx][anchor_idx_on_scale, j, i, 5] = int(class_label)
        #         elif anchor_taken == 0 and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
        #             targets[scale_idx][anchor_idx_on_scale, j, i, 0] = -1
        #             # 如果这个box已经在这个scale上赋值过，也就是已经有其他anchor负责预测这个box, 但是这个anchor与box的iou比较大并且这个anchor还没有用来预测box
        #             # 就将这个anchor的confidence置为-1，使其不干扰那个负责预测box的anchor

        # 将这张图片上的所有box信息赋值到targets，将每一个box分配到三个scale上，并且每个scale只有一个cell上的一个anchor保存(预测)box信息
        # 以下为自己写的box信息分配到targets的过程，与上面的差不多，不过似乎更容易理解，代码更加compact
        for box in bboxes:
            x, y, width, height, class_label = box   # 此处box的x,y,width,height都是相对于整张图片归一化得到的，所有的x,y,width,height都介于[0,1]
            for scale_idx, S in enumerate(self.S):    # 遍历三个scale   scale_idx:0,1,2
                ious_scale_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors[3*scale_idx:3*(scale_idx+1), :])   # (3)
                anchors_indices = ious_scale_anchors.argsort(descending=True, dim=0)   # 取box与三个anchor的iou的最大值的索引，也就是选择与box有最大iou的anchor来预测box(这样预测的难度更简单)
                i, j = int(S * x), int(S * y)
                x_cell, y_cell = S * x - i, S * y - j
                width_cell, height_cell = S * width, S * height
                box_cell_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                taken_anchor = targets[scale_idx][anchors_indices[0], j, i, 0]   # 查看与box最匹配的anchor是否已经用于预测其他box
                if taken_anchor == 0:    # 如果这个scale的这个cell的这个anchor还没有用来预测其他box，就让它来预测当前的box
                    targets[scale_idx][anchors_indices[0], j, i, 0] = 1
                    targets[scale_idx][anchors_indices[0], j, i, 1:5] = box_cell_coordinates
                    targets[scale_idx][anchors_indices[0], j, i, 5] = int(class_label)


        # return image, tuple(targets)
        return image, targets  # targets:[tensor_size(3, 13, 13, 6), tensor_size(3, 26, 26, 6), tensor_size(3, 52, 52, 6)]
        # 经过DataLoader封装后->targets:[tensor_size(BS, 3, 13, 13, 6), tensor_size(BS, 3, 26, 26, 6), tensor_size(BS, 3, 52, 52, 6)]


def test():
    anchors = config.ANCHORS   # list [3,3,2]
    # ANCHORS = [
    #     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    #     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    #     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    # ]  # Note these have been rescaled to be between [0, 1]
    transform = config.test_transforms
    dataset = YOLODataset("VOC/8examples.csv", "VOC/images", "VOC/labels", anchors=anchors, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # 遍历loader中的图片和目标边界框来进行画图
    for imgs, imgs_targets in loader:      # imgs:(BS, 3, 416, 416)  imgs_targets:[tensor_size(BS, 3, 13, 13, 6), tensor_size(BS, 3, 26, 26, 6), tensor_size(BS, 3, 52, 52, 6)]
        boxes = []  # 用于存放imgs_targets对应的所有box
        for i in range(3):   # 遍历3个scale的targets, 将3个scale的targets转换成box的列表
            targets = imgs_targets[i]  # 第i个scale的targets   (BS, 3, S, S, 6)
            anchors_for_scale = torch.tensor(anchors[i])   # 第i个scale对应的anchors  (3, 2)
            boxes += cells_to_bboxes(targets, anchors_for_scale, targets.shape[2], is_preds=False)[0]   # 取[0]表示只取这个batch中第0张图片上的所有box   取[0]前列表为list:(BS, 3*S*S, 6) 取[0]后列表为list:(3*S*S, 6)
            # 在3个scale上遍历后，boxes = [[1,2,3,4,5,6], [1,2,3,4,5,6],..., [1,2,3,4,5,6]]  #共有3*13*13+3*26*26+3*52*52=10647个box   [1,2,3,4,5,6]=[class_label, confidence, x_cell, y_cell, w_cell, h_cell]

        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")  # 对boxes中的box进行nms，也就是筛选出最接近真实目标的边界框
        plot_image(imgs[0].permute(1, 2, 0).to("cpu"), boxes)     # imgs[0] 只取这个batch中的第0张图片(3, 416, 416)->(416, 416, 3).to("cpu")


if __name__ == "__main__":
    test()




















# def test():
#     anchors = config.ANCHORS    # (3, 3, 2) 每个anchor的大小也都是相对于整张图像进行归一化的
#     transform = config.test_transforms
#     dataset = YOLODataset(csv_file="VOC/8examples.csv", img_dir="VOC/images", label_dir="VOC/labels", anchors=anchors, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
#     for x, y in loader:   # x(images):(BS, 3, H, W)=(1, 3, 416, 416)   y(images_targets):[tensor(BS, 3, 13, 13, 6), tensor(BS, 3, 26, 26, 6), tensor(BS, 3, 52, 52, 6)]
#         boxes = []
#         for i in range(3):   # 3 scale
#             anchors_for_scale = torch.tensor(anchors[i])  # (3, 2)
#             targets = y[i]    # 第i个scale上的targets (1, 3, S, S, 6)
#             boxes += cells_to_bboxes(targets, anchors_for_scale, targets.shape[2], is_preds=False)[0]   # 加上[0]是只想取y这个batch中第0张图片上的所有boxes
#             # boxes经过3个scale上的循环后获得了这个batch中第0张图片上的所有box boxes=[box1, box2, box3,..,box(num_all)]  # num_all = 13*13*3+26*26*3+52*52*3
#
#         boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
#         plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
#
#
#
#
#
#
# if __name__ == "__main__":
#     test()













# def test():
#     anchors = config.ANCHORS
#     transform = config.test_transforms
#     dataset = YOLODataset("VOC/8examples.csv", "VOC/images", "VOC/labels", anchors=anchors, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
#     S = [13, 26, 52]
#     # scaled_anchors = torch.tensor(anchors) * (torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))   # 不使用S=[13, 26, 52]对anchors进行缩放也行
#     scaled_anchors = torch.tensor(anchors)
#     for x, y in loader:
#         boxes = []
#         for i in range(3):   # 3个scale
#             anchor = scaled_anchors[i]  # 第i个scale上的scaled_anchor   (3,2)
#             print(anchor.shape)  # (3,2)
#             print(y[i].shape)   # (1, 3, S, S, 6)  # 第一个S表示y轴，第二个S表示x轴
#             boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors=anchor)[0]
#
#         boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
#         print(boxes)
#         plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
#
# if __name__ == "__main__":
#     test()






