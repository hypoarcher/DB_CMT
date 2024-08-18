import torch
from model import DbcmtNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14, patch_size_v=[2, 14, 14], num_layers=1,
                     num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
model.load_state_dict(torch.load('/data_chi/wubo/DB_CMT_Net/weight/DB_CMT/tnus/2/bestmodel.pth'))
model.to(device)
model.eval()
