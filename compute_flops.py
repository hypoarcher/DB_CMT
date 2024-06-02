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
print('Params = ' + str(params/1000**2) + 'M')