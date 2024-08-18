<<<<<<< HEAD
import torch
import torchvision
import torch.nn as nn
from CAU_T import Cau
from UAC_T import Uac
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):     # resnet3d basicblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DoubleConv(nn.Module):   # unet encoder
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DbcmtNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes, num_blocks, image_size, patch_size, patch_size_v, num_layers,
                 num_heads, hidden_dim, mlp_dim, dropout, num_frames, block=BasicBlock):
        super(DbcmtNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.image_size = image_size
        self. patch_size = patch_size
        self.patch_size_v = patch_size_v
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.num_frames = num_frames
        self.block = block
        self.hidden_dim_v = hidden_dim * patch_size_v[0]

        # Unet
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.cr1 = nn.Conv2d(64, 3, kernel_size=1, stride=1)    # unet channel reduction
        self.cr2 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.cr3 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.cr4 = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        self.ci1 = nn.ConvTranspose2d(3, 64, kernel_size=2, stride=2)    # unet channel increase
        self.ci2 = nn.ConvTranspose2d(3, 128, kernel_size=2, stride=2)
        self.ci3 = nn.ConvTranspose2d(3, 256, kernel_size=2, stride=2)
        self.ci4 = nn.ConvTranspose2d(3, 512, kernel_size=2, stride=2)

        self.r_cr1 = nn.Conv3d(64, 3, kernel_size=1, stride=1)    # res3d channel reduction
        self.r_cr2 = nn.Conv3d(128, 3, kernel_size=1, stride=1)
        self.r_cr3 = nn.Conv3d(256, 3, kernel_size=1, stride=1)
        self.r_cr4 = nn.Conv3d(512, 3, kernel_size=1, stride=1)
        self.r_ci1 = nn.Conv3d(3, 64, kernel_size=1, stride=1)    # res3d channel increase
        self.r_ci2 = nn.Conv3d(3, 128, kernel_size=1, stride=1)
        self.r_ci3 = nn.Conv3d(3, 256, kernel_size=1, stride=1)
        self.r_ci4 = nn.Conv3d(3, 512, kernel_size=1, stride=1)


        # resnet3d
        self.in_planes = 64
        self.r_conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.r_bn1 = nn.BatchNorm3d(64)
        self.r_layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.r_layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.r_layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.r_layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.r_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.r_fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        # encoder1
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        out = F.relu(self.r_bn1(self.r_conv1(y)))
        out1 = self.r_layer1(out)
        p1_a = self.cr1(p1)
        out1_a = self.r_cr1(out1)
        cau_t1 = Cau(image_size=self.image_size, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t1.to(device)
        uac_t1 = Uac(image_size=self.image_size, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames)
        uac_t1.to(device)
        attn_c1 = cau_t1(p1_a, out1_a)
        attn_u1 = uac_t1(p1_a, out1_a)
        attn_c1 = self.r_ci1(attn_c1)
        attn_u1 = self.ci1(attn_u1)
        out1 = out1 + attn_c1
        # encoder2
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        out2 = self.r_layer2(out1)
        p2_a = self.cr2(p2)
        out2_a = self.r_cr2(out2)
        cau_t2 = Cau(image_size=self.image_size // 2, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t2.to(device)
        uac_t2 = Uac(image_size=self.image_size // 2, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 2)
        uac_t2.to(device)
        attn_c2 = cau_t2(p2_a, out2_a)
        attn_u2 = uac_t2(p2_a, out2_a)
        attn_c2 = self.r_ci2(attn_c2)
        attn_u2 = self.ci2(attn_u2)
        out2 = out2 + attn_c2
        # encoder3
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        out3 = self.r_layer3(out2)
        p3_a = self.cr3(p3)
        out3_a = self.r_cr3(out3)
        cau_t3 = Cau(image_size=self.image_size // 4, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t3.to(device)
        uac_t3 = Uac(image_size=self.image_size // 4, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 4)
        uac_t3.to(device)
        attn_c3 = cau_t3(p3_a, out3_a)
        attn_u3 = uac_t3(p3_a, out3_a)
        attn_c3 = self.r_ci3(attn_c3)
        attn_u3 = self.ci3(attn_u3)
        out3 = out3 + attn_c3
        # encoder4
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        out4 = self.r_layer4(out3)
        p4_a = self.cr4(p4)
        out4_a = self.r_cr4(out4)
        cau_t4 = Cau(image_size=self.image_size // 8, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t4.to(device)
        uac_t4 = Uac(image_size=self.image_size // 8, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 8)
        uac_t4.to(device)
        attn_c4 = cau_t4(p4_a, out4_a)
        attn_u4 = uac_t4(p4_a, out4_a)
        attn_c4 = self.r_ci4(attn_c4)
        attn_u4 = self.ci4(attn_u4)
        out4 = out4 + attn_c4

        # unet decoder1
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, attn_u4], dim=1)
        c6 = self.conv6(merge6)
        # decoder2
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, attn_u3], dim=1)
        c7 = self.conv7(merge7)
        # decoder3
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, attn_u2], dim=1)
        c8 = self.conv8(merge8)
        # decoder4
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, attn_u1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out_cls = self.r_avgpool(out4)
        out_cls = torch.flatten(out_cls, 1)
        out_cls = self.r_fc(out_cls)
        out_seg = nn.Sigmoid()(c10)
        return out_cls, out_seg


# # exemple
# x = torch.randn((16,3,448,448))
# y = torch.rand((16,3, 16, 224, 224))
# model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14, patch_size_v=[2, 14, 14], num_layers=1,
#                  num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
# z = model(x, y)
# print(z[0].shape)
# print(z[1].shape)



=======
import torch
import torchvision
import torch.nn as nn
from CAU_T import Cau
from UAC_T import Uac
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):     # resnet3d basicblock
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DoubleConv(nn.Module):   # unet encoder
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DbcmtNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes, num_blocks, image_size, patch_size, patch_size_v, num_layers,
                 num_heads, hidden_dim, mlp_dim, dropout, num_frames, block=BasicBlock):
        super(DbcmtNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.image_size = image_size
        self. patch_size = patch_size
        self.patch_size_v = patch_size_v
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.num_frames = num_frames
        self.block = block
        self.hidden_dim_v = hidden_dim * patch_size_v[0]

        # Unet
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.cr1 = nn.Conv2d(64, 3, kernel_size=1, stride=1)    # unet channel reduction
        self.cr2 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.cr3 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.cr4 = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        self.ci1 = nn.ConvTranspose2d(3, 64, kernel_size=2, stride=2)    # unet channel increase
        self.ci2 = nn.ConvTranspose2d(3, 128, kernel_size=2, stride=2)
        self.ci3 = nn.ConvTranspose2d(3, 256, kernel_size=2, stride=2)
        self.ci4 = nn.ConvTranspose2d(3, 512, kernel_size=2, stride=2)

        self.r_cr1 = nn.Conv3d(64, 3, kernel_size=1, stride=1)    # res3d channel reduction
        self.r_cr2 = nn.Conv3d(128, 3, kernel_size=1, stride=1)
        self.r_cr3 = nn.Conv3d(256, 3, kernel_size=1, stride=1)
        self.r_cr4 = nn.Conv3d(512, 3, kernel_size=1, stride=1)
        self.r_ci1 = nn.Conv3d(3, 64, kernel_size=1, stride=1)    # res3d channel increase
        self.r_ci2 = nn.Conv3d(3, 128, kernel_size=1, stride=1)
        self.r_ci3 = nn.Conv3d(3, 256, kernel_size=1, stride=1)
        self.r_ci4 = nn.Conv3d(3, 512, kernel_size=1, stride=1)


        # resnet3d
        self.in_planes = 64
        self.r_conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.r_bn1 = nn.BatchNorm3d(64)
        self.r_layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.r_layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.r_layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.r_layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.r_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.r_fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        # encoder1
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        out = F.relu(self.r_bn1(self.r_conv1(y)))
        out1 = self.r_layer1(out)
        p1_a = self.cr1(p1)
        out1_a = self.r_cr1(out1)
        cau_t1 = Cau(image_size=self.image_size, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t1.to(device)
        uac_t1 = Uac(image_size=self.image_size, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames)
        uac_t1.to(device)
        attn_c1 = cau_t1(p1_a, out1_a)
        attn_u1 = uac_t1(p1_a, out1_a)
        attn_c1 = self.r_ci1(attn_c1)
        attn_u1 = self.ci1(attn_u1)
        out1 = out1 + attn_c1
        # encoder2
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        out2 = self.r_layer2(out1)
        p2_a = self.cr2(p2)
        out2_a = self.r_cr2(out2)
        cau_t2 = Cau(image_size=self.image_size // 2, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t2.to(device)
        uac_t2 = Uac(image_size=self.image_size // 2, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 2)
        uac_t2.to(device)
        attn_c2 = cau_t2(p2_a, out2_a)
        attn_u2 = uac_t2(p2_a, out2_a)
        attn_c2 = self.r_ci2(attn_c2)
        attn_u2 = self.ci2(attn_u2)
        out2 = out2 + attn_c2
        # encoder3
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        out3 = self.r_layer3(out2)
        p3_a = self.cr3(p3)
        out3_a = self.r_cr3(out3)
        cau_t3 = Cau(image_size=self.image_size // 4, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t3.to(device)
        uac_t3 = Uac(image_size=self.image_size // 4, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 4)
        uac_t3.to(device)
        attn_c3 = cau_t3(p3_a, out3_a)
        attn_u3 = uac_t3(p3_a, out3_a)
        attn_c3 = self.r_ci3(attn_c3)
        attn_u3 = self.ci3(attn_u3)
        out3 = out3 + attn_c3
        # encoder4
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        out4 = self.r_layer4(out3)
        p4_a = self.cr4(p4)
        out4_a = self.r_cr4(out4)
        cau_t4 = Cau(image_size=self.image_size // 8, patch_size=self.patch_size, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim, mlp_dim=1024, dropout=0.1)
        cau_t4.to(device)
        uac_t4 = Uac(image_size=self.image_size // 8, patch_size_v=self.patch_size_v, num_layers=1, num_heads=2,
                     hidden_dim=self.hidden_dim_v, mlp_dim=1024, dropout=0.1, num_frames=self.num_frames // 8)
        uac_t4.to(device)
        attn_c4 = cau_t4(p4_a, out4_a)
        attn_u4 = uac_t4(p4_a, out4_a)
        attn_c4 = self.r_ci4(attn_c4)
        attn_u4 = self.ci4(attn_u4)
        out4 = out4 + attn_c4

        # unet decoder1
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, attn_u4], dim=1)
        c6 = self.conv6(merge6)
        # decoder2
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, attn_u3], dim=1)
        c7 = self.conv7(merge7)
        # decoder3
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, attn_u2], dim=1)
        c8 = self.conv8(merge8)
        # decoder4
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, attn_u1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out_cls = self.r_avgpool(out4)
        out_cls = torch.flatten(out_cls, 1)
        out_cls = self.r_fc(out_cls)
        out_seg = nn.Sigmoid()(c10)
        return out_cls, out_seg


# # exemple
# x = torch.randn((16,3,448,448))
# y = torch.rand((16,3, 16, 224, 224))
# model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14, patch_size_v=[2, 14, 14], num_layers=1,
#                  num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
# z = model(x, y)
# print(z[0].shape)
# print(z[1].shape)



>>>>>>> 661c694 ('init')
