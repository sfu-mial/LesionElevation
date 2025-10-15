import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, num_classes, GAP=False):
        super(VGG16, self).__init__()
        self.GAP = GAP
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        if self.GAP:
            self.pool = nn.AvgPool2d(7, stride=1)
            self.cls = nn.Linear(
                in_features=512, out_features=num_classes, bias=True
            )
        else:
            self.dense = nn.Sequential(*list(net.classifier.children())[:-1])
            self.cls = nn.Linear(
                in_features=4096, out_features=num_classes, bias=True
            )
        # initialize
        nn.init.normal_(self.cls.weight, 0.0, 0.01)
        nn.init.constant_(self.cls.bias, 0.0)

    def forward(self, x):
        block1 = self.conv_block1(x)  # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)  # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)  # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)  # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)  # /16
        pool5 = F.max_pool2d(block5, 2, 2)  # /32
        N, __, __, __ = pool5.size()
        if self.GAP:
            g = self.pool(pool5).view(N, -1)
        else:
            g = self.dense(pool5.view(N, -1))
        out = self.cls(g)
        return out


class VGG16_GradCAM(nn.Module):
    #     https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    #     https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/utilities.py#L10

    def __init__(self, num_classes, GAP=False):
        super(VGG16_GradCAM, self).__init__()
        self.GAP = GAP
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        if self.GAP:
            self.pool = nn.AvgPool2d(7, stride=1)
            self.cls = nn.Linear(
                in_features=512, out_features=num_classes, bias=True
            )
        else:
            self.dense = nn.Sequential(*list(net.classifier.children())[:-1])
            self.cls = nn.Linear(
                in_features=4096, out_features=num_classes, bias=True
            )
        # initialize
        nn.init.normal_(self.cls.weight, 0.0, 0.01)
        nn.init.constant_(self.cls.bias, 0.0)

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        block1 = self.conv_block1(x)  # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)  # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)  # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)  # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)  # /16
        pool5 = F.max_pool2d(block5, 2, 2)  # /32
        N, __, __, __ = pool5.size()
        if self.GAP:
            g = self.pool(pool5).view(N, -1)
        else:
            g = self.dense(pool5.view(N, -1))
        out = self.cls(g)

        # register the hook
        _ = block5.register_hook(self.activations_hook)

        return out

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        block1 = self.conv_block1(x)  # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)  # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)  # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)  # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)  # /16

        return block5


class VGG16_Depth(nn.Module):
    def __init__(self, num_classes, n_depth_classes, GAP=False):
        super(VGG16_Depth, self).__init__()
        self.GAP = GAP
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        if self.GAP:
            self.pool = nn.AvgPool2d(7, stride=1)
            self.cls = nn.Linear(
                in_features=(512 + n_depth_classes),
                out_features=num_classes,
                bias=True,
            )
        else:
            self.dense = nn.Sequential(*list(net.classifier.children())[:-1])
            self.cls = nn.Linear(
                in_features=4096, out_features=num_classes, bias=True
            )
        # initialize
        nn.init.normal_(self.cls.weight, 0.0, 0.01)
        nn.init.constant_(self.cls.bias, 0.0)

    def forward(self, x, p_depth):
        block1 = self.conv_block1(x)  # /1
        pool1 = F.max_pool2d(block1, 2, 2)  # /2
        block2 = self.conv_block2(pool1)  # /2
        pool2 = F.max_pool2d(block2, 2, 2)  # /4
        block3 = self.conv_block3(pool2)  # /4
        pool3 = F.max_pool2d(block3, 2, 2)  # /8
        block4 = self.conv_block4(pool3)  # /8
        pool4 = F.max_pool2d(block4, 2, 2)  # /16
        block5 = self.conv_block5(pool4)  # /16
        pool5 = F.max_pool2d(block5, 2, 2)  # /32
        N, __, __, __ = pool5.size()
        if self.GAP:
            g = self.pool(pool5).view(N, -1)
            g_concat = torch.cat([g, p_depth], dim=1)
        else:
            g = self.dense(pool5.view(N, -1))
        out = self.cls(g_concat)
        return out


class OtherModels(nn.Module):
    def __init__(self, num_classes, base):
        super(OtherModels, self).__init__()
        if base == "resnet50":
            self.base = models.resnet50(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        elif base == "resnet18":
            self.base = models.resnet18(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        elif base == "densenet121":
            self.base = models.densenet121(pretrained=True)
            self.base.classifier = nn.Linear(
                in_features=self.base.classifier.in_features,
                out_features=num_classes,
            )
        elif base == "mobilenetv2":
            self.base = models.mobilenet_v2(pretrained=True)
            self.base.classifier[1] = nn.Linear(
                in_features=self.base.classifier[1].in_features,
                out_features=num_classes,
            )
        elif base == "mobilenetv3l":
            self.base = models.mobilenet_v3_large(pretrained=True)
            self.base.classifier[3] = nn.Linear(
                in_features=self.base.classifier[3].in_features,
                out_features=num_classes,
            )
        elif base == "efficientnetb0":
            self.base = models.efficientnet_b0(pretrained=True)
            self.base.classifier[1] = nn.Linear(
                in_features=1280, out_features=num_classes
            )
        elif base == "efficientnetb1":
            self.base = models.efficientnet_b1(pretrained=True)
            self.base.classifier[1] = nn.Linear(
                in_features=1280, out_features=num_classes
            )

    def forward(self, x):
        return self.base(x)
