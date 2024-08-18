<<<<<<< HEAD
from model import DbcmtNet
from thop import profile
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14,
                 patch_size_v=[2, 14, 14], num_layers=1,
                 num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
model = model.to(device)
model.eval()
flops, params = profile(model, input_size=(1, 3, 448,448))
print('FLOPs = ' + str(flops/1000**3) + 'G')
=======
from model import DbcmtNet
from thop import profile
import torch
import torchvision
from unet import Unet


# DBCMT = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14,
#                  patch_size_v=[2, 14, 14], num_layers=1,
#                  num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
# RESNET = torchvision.models.video.r3d_18(num_classes=2)
# UNET = Unet(in_ch=3, out_ch=1)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1176, nhead=2)
cau = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)


x = torch.randn(1, 2048, 1176)

flops, params = profile(cau, inputs=(x,))

print('FLOPs = ' + str(flops/1000**3) + 'G')
>>>>>>> 661c694 ('init')
print('Params = ' + str(params/1000**2) + 'M')