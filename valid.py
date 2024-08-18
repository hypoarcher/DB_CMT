<<<<<<< HEAD
import torch
from utils import confusion_matrix
from sklearn.metrics import roc_auc_score


def valid(config, net, val_loader, criterion_cls, criterion_seg):
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.class_num, config.class_num))
    for i, pack in enumerate(val_loader):
        frames = pack['videos'].to(device)
        mask = pack['masks'].to(device)
        image = pack['images'].to(device)
        if frames.shape[1] == 1:
            frames = frames.expand((-1, 3, -1, -1, -1))

        if image.shape[1] == 1:
            image = image.expand((-1, 3, -1, -1))
        names = pack['names']
        labels = pack['labels'].to(device)

        out_cls, out_seg = net(image, frames)
        # output = output.logits        #GOOGLENET需要

        loss_cls = criterion_cls(out_cls, labels)
        loss_seg = criterion_seg(out_seg, mask)
        loss = loss_cls + 0.5 * loss_seg

        pred = out_cls.argmax(dim=1)
        y_true.append(labels.detach().item())
        y_score.append(out_cls[0].softmax(0)[1].item())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()

    avg_epoch_loss = epoch_loss / len(val_loader)

    #   t  N  P
    # p
    # N    TN FN
    # P    FP TP
    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    rec = sen
    f1score = 2 * pre * rec / (pre + rec + 1e-6)
    auc = roc_auc_score(y_true, y_score)

=======
import torch
from utils import confusion_matrix
from sklearn.metrics import roc_auc_score


def valid(config, net, val_loader, criterion_cls, criterion_seg):
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.class_num, config.class_num))
    for i, pack in enumerate(val_loader):
        frames = pack['videos'].to(device)
        mask = pack['masks'].to(device)
        image = pack['images'].to(device)
        if frames.shape[1] == 1:
            frames = frames.expand((-1, 3, -1, -1, -1))

        if image.shape[1] == 1:
            image = image.expand((-1, 3, -1, -1))
        names = pack['names']
        labels = pack['labels'].to(device)

        out_cls, out_seg = net(image, frames)
        # output = output.logits        #GOOGLENET需要

        loss_cls = criterion_cls(out_cls, labels)
        loss_seg = criterion_seg(out_seg, mask)
        loss = loss_cls + 0.5 * loss_seg

        pred = out_cls.argmax(dim=1)
        y_true.append(labels.detach().item())
        y_score.append(out_cls[0].softmax(0)[1].item())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()

    avg_epoch_loss = epoch_loss / len(val_loader)

    #   t  N  P
    # p
    # N    TN FN
    # P    FP TP
    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    rec = sen
    f1score = 2 * pre * rec / (pre + rec + 1e-6)
    auc = roc_auc_score(y_true, y_score)

>>>>>>> 661c694 ('init')
    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]