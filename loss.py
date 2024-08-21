import torch
import torch.nn as nn
from utils import intersection_over_union



class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

        # 初始化几种需要用到的损失函数
        self.mse = nn.MSELoss()  # 均方误差损失函数
        self.bce = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
        self.entropy = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.sigmoid = nn.Sigmoid()

        # 初始化一些常数用于加权求和各部分的损失
        self.lambda_class = 1      # 加权分类损失
        self.lambda_box = 10       # 加权边界框坐标损失
        self.lambda_obj = 1        # 加权有目标边界框置信度损失
        self.lambda_noobj = 10     # 加权没有目标边界框置信度损失


    # 计算loss的时候对每个scale的prediction和target分开算，所以每次只计算一个scale的loss
    def forward(self, prediction, target, anchors):
        # prediction (BS, 3, S, S, num_classes+5)
        # num_classes+5 = 20+5 => (t_conf, t_x, t_y, t_w, t_h, class1_prob, class2_prob,..., class20_prob)

        # target (BS, 3, S, S, 6)  6 => (confidence=0/1, x_cell, y_cell, w_cell, h_cell, class_label)
        # x_cell,y_cell,w_cell,h_cell∈[0,1]  class_label: 0,1,2,...,19

        # anchors tensor(3, 2) for the scale

        # check obj and noobj
        obj = target[..., 0] == 1   # bool (BS, 3, S, S)  # 找出存在目标的batch中所有image上的所有cell上的anchor
        noobj = target[..., 0] == 0   # bool (BS, 3, S, S)   # 找出不存在目标的batch中所有image上的所有cell上的anchor




        ##########################
        # FOR NO OBJECT LOSS    希望没有目标的box的confidence -> 0
        ##########################
        noobj_loss = self.bce(self.sigmoid(prediction[..., 0:1][noobj]), target[..., 0:1][noobj])
        # prediction[..., 0:1][noobj]    (BS, 3, S, S, 1) (BS, 3, S, S) -> (noobj中True的总数, 1)  1:t_conf    sigmoid(t_conf)->conf
        # target[..., 0:1][noobj]    (BS, 3, S, S, 1) (BS, 3, S, S) -> (noobj中True的总数, 1)      1:   0




        ##########################
        # FOR OBJECT LOSS       希望有目标的box的confidence -> 1*IOU   # 后续会优化IOU使其->1,这样连带使得有目标的box的confidence->1
        ##########################
        # 对于prediction中的t_x,t_y,t_w,t_h，需要通过公式转换才能变成x_cell,y_cell,w_cell,h_cell
        # x_cell = sigmoid(t_x)   [0, 1]
        # y_cell = sigmoid(t_y)   [0, 1]
        # w_cell = pw * exp(t_w)
        # h_cell = ph * exp(t_h)
        # conf = sigmoid(t_conf)  [0, 1]
        anchors = anchors.reshape((1, 3, 1, 1, 2))   # anchors:(3, 2)->(1, 3, 1, 1, 2) 后续会在维度为1的地方进行广播操作
        box_xy_cell = self.sigmoid(prediction[..., 1:3])   # (BS, 3, S, S, 2)  2: (x_cell, y_cell)
        box_wh_cell = anchors * torch.exp(prediction[..., 3:5])      # anchors广播->(BS, 3, S, S, 2)  (BS, 3, S, S, 2) 2: (w_cell, h_cell)
        box_cell = torch.cat([box_xy_cell, box_wh_cell], dim=-1)     # (BS, 3, S, S, 2) (BS, 3, S, S, 2) -> (BS, 3, S, S, 4)
        ious = intersection_over_union(box_cell[obj], target[..., 1:5][obj]).detach()    # (BS, 3, S, S, 4) (BS, 3, S, S) -> (obj中True总数, 4) -> (obj中True总数, 1)  # .detach()
        obj_loss = self.mse(self.sigmoid(prediction[..., 0:1][obj]), target[..., 0:1][obj] * ious)
        # prediction[..., 0:1][obj]   (BS, 3, S, S, 1) (BS, 3, S, S) -> (obj中True总数, 1)
        # target[..., 0:1][obj] * ious    (obj中True总数, 1) * (obj中True总数, 1) -> (obj中True总数, 1)



        ##########################
        # FOR BOX COORDINATES       希望有目标的box的坐标预测更准确，也就是与真实目标框的IOU->1,这样也会间接引导有目标box的confidence->1
        ##########################
        # 希望达成的目标如下
        # sigmoid(t_x) -> x_cell
        # sigmoid(t_y) -> y_cell
        # pw * exp(t_w) -> w_cell  转化为 t_w -> log(w_cell / pw)
        # ph * exp(t_h) -> h_cell  转换为 t_h -> log(h_cell / ph)
        prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])  # (BS, 3, S, S, 2)  2:(x_cell, y_cell)
        target[..., 3:5] = torch.log(target[..., 3:5] / anchors + 1e-16)   # (BS, 3, S, S, 2) / (1, 3, 1, 1, 2)  广播机制     结果为(BS, 3, S, S, 2)
        box_loss = self.mse(prediction[..., 1:5][obj], target[..., 1:5][obj])   # (obj中True的总数, 4)  (obj中True的总数, 4)

        ##########################
        # FOR CLASS LOSS      希望有目标的box的类别是准确的
        ##########################
        class_loss = self.entropy(prediction[..., 5:][obj], (target[..., 5][obj].long()))
        # prediction[..., 5:][obj]  (BS, 3, S, S, 20) (BS, 3, S, S) -> (obj中True的总数, 20)
        # target[..., 5][obj]  (BS, 3, S, S) (BS, 3, S, S) -> (obj中True的总数)


        total_loss = (
            self.lambda_noobj * noobj_loss
            + self.lambda_obj * obj_loss
            + self.lambda_box * box_loss
            + self.lambda_class * class_loss
        )

        return total_loss




if __name__ == "__main__":
    loss_fn = YoloLoss()
    prediction = torch.randn(1, 3, 13, 13, 25)
    target = torch.randn(1, 3, 13, 13, 6)
    anchors = torch.randn(3, 2)
    loss = loss_fn(prediction, target, anchors)
    print(loss)
































































