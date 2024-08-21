
import torch
import config
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from loss import YoloLoss
from model import YOLOv3
from utils import get_loaders
from utils import check_class_accuracy
from utils import get_evaluation_bboxes
from utils import mean_average_precision
from utils import plot_couple_examples



# 构造训练函数   (apply mixes precision training)
def train_fn(train_loader, model, optimizer, scaler, loss_fn):
    loop = tqdm(train_loader, leave=True)  # 迭代结束后保留显示进度条
    losses = []   # 保存每一个epoch的所有loss，并利用它来求这个epoch的平均loss

    for batch_idx, (imgs, targets) in enumerate(loop):
        # imgs: (BS, 3, H, W)
        # targets: [tensor(BS, 3, 13, 13, 6),tensor(BS, 3, 26, 26, 6), tensor(BS, 3, 52, 52, 6)]
        imgs = imgs.to(config.DEVICE)

        # 混合精度训练
        # forward
        with torch.cuda.amp.autocast():
            outs = model(imgs)  # outs: [tensor(BS, 3, 13, 13, 25),tensor(BS, 3, 26, 26, 25), tensor(BS, 3, 52, 52, 25)]
            # loss是在每个scale上求得的
            loss = (
                loss_fn(outs[0], targets[0].to(config.DEVICE), torch.tensor(config.ANCHORS)[0].to(config.DEVICE))    # config.ANCHORS: list[3,3,2]
                + loss_fn(outs[1], targets[1].to(config.DEVICE), torch.tensor(config.ANCHORS)[1].to(config.DEVICE))  # 张量要放在相同的device上
                + loss_fn(outs[2], targets[2].to(config.DEVICE), torch.tensor(config.ANCHORS)[2].to(config.DEVICE))
            )

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()    # 优化器清零梯度，防止上一次的梯度累积到这次
        # loss.backward()
        scaler.scale(loss).backward()   # scaler缩放梯度避免混合精度训练时的梯度消失或梯度爆炸再反向传播
        # optimizer.step()
        scaler.step(optimizer)    # scaler将梯度变换回原来的大小再使用优化器对权重进行更新
        scaler.update()   # scaler根据梯度的大小情况进行更新，防止混合精度训练时候的梯度消失或梯度爆炸

        mean_loss = sum(losses) / len(losses)    # 求这个epoch的平均损失
        loop.set_postfix(loss=mean_loss)    # 给进度条加上后缀loss=mean_loss
    return mean_loss




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    checkpoint = torch.load(r"checkpoint/yolov3_pascal_78.1map.pth",
                            map_location=device)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = YoloLoss()
    train_loader, test_loader, train_eval_loader = get_loaders(r"VOC/train.csv", r"VOC/test.csv")


    for epoch in range(100):
        mean_loss = train_fn(train_loader=train_loader, model=model, optimizer=optimizer, scaler=scaler, loss_fn=loss_fn, scheduler=scheduler)
        scheduler.step(mean_loss)
        if epoch >= 0 and epoch % 5 == 0:
            check_class_accuracy(model, test_loader, threshold=0.6)
            print("check_class_accuracy")
            all_pred_boxes, all_true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=0.1, anchors=config.ANCHORS, threshold=0.6, box_format="midpoint", device=device)
            print("get_eval_box")
            mapval = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=config.NUM_CLASSES)
            print(f"epoch:{epoch}, map:{mapval.item()}")
            model.train()
            if mapval.item() > 0.8:
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, "checkpoint/yolov3_pascal.pth")
                break




def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    checkpoint = torch.load(r"checkpoint\yolov3_pascal_78.1map.pth", map_location=device)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    show_num = 0

    train_loader, test_loader, train_eval_loader = get_loaders(
        r"VOC\8examples.csv",
        r"VOC\8examples.csv")

    plot_couple_examples(model, test_loader, thresh=0.5, iou_thresh=0.3, anchors=config.ANCHORS, device=device)




if __name__ == "__main__":
    # main()
    test()


























