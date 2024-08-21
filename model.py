"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (filters/out_channels, kernel_size, stride)
Every conv is a same convolution.
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer

"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leakyrelu(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=1, use_residual=True):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.num_repeats = num_repeats
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
                    CNNBlock(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)

        return x




class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes=20):
        super(ScalePrediction, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels * 2, 3 * (num_classes + 5), bn_act=False, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x):
        return (self.pred(x)).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        # x:(BS, C, S, S)->(BS, 3*(num_classes+5), S, S)->(BS, 3, num_classes+5, S, S)->(BS, 3, S, S, num_classes+5)





class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLOv3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        in_channels = self.in_channels
        layers = nn.ModuleList()

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module   # (32, 3, 1)
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(1 if kernel_size == 3 else 0),
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]   # ["B", 1]
                layers.append(
                    ResidualBlock(in_channels, num_repeats=num_repeats)
                )

            elif isinstance(module, str):
                if module == "S":
                    layers.append(ResidualBlock(in_channels, num_repeats=1, use_residual=False))
                    layers.append(CNNBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0))
                    layers.append(ScalePrediction(in_channels//2, num_classes=self.num_classes))
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers



    def forward(self, x):
        outputs = []   # [out1, out2, out3]
        route_connections = []   # [rc_1, rc_2]
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)   # (BS, C, H, W)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)   # x:(BS, in_channels*3, H, W)
                route_connections.pop(-1)   # 默认是pop(-1)，-1可以不用写

        return outputs    # 3 * (BS, 3, S, S, num_classes+5)  5: x,y,w,h,confidence
        # num_classes+5: t_conf, t_x, t_y, t_w, t_h, class1_prob, class2_prob, ..., class20_prob





if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(in_channels=3, num_classes=num_classes)
    input = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    outputs = model(input)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    assert outputs[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert outputs[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert outputs[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    print("Success!")












