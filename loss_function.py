<<<<<<< HEAD
import torch


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)
        - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# 多类分割dice损失
class Generalized_Dice_Loss(torch.nn.Module):
    def __init__(self):
        super(Generalized_Dice_Loss, self).__init__()

    def forward(self, pred, target):

        epsilon = 1e-6
        wei = torch.sum(target, axis=[0, 2, 3]) # (n_class,)
        wei = 1/(wei**2+epsilon)
        intersection = torch.sum(wei*torch.sum(pred * target, axis=[0, 2, 3]))
        union = torch.sum(wei*torch.sum(pred + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss
=======
import torch


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)
        - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# 多类分割dice损失
class Generalized_Dice_Loss(torch.nn.Module):
    def __init__(self):
        super(Generalized_Dice_Loss, self).__init__()

    def forward(self, pred, target):

        epsilon = 1e-6
        wei = torch.sum(target, axis=[0, 2, 3]) # (n_class,)
        wei = 1/(wei**2+epsilon)
        intersection = torch.sum(wei*torch.sum(pred * target, axis=[0, 2, 3]))
        union = torch.sum(wei*torch.sum(pred + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss
>>>>>>> 661c694 ('init')
