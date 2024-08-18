<<<<<<< HEAD
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch
import os
from metrics import Metrics

def evaluate(mask_path,predict_path):
    image_mask = cv2.imread(mask_path,0)
    # print(mask_path)
    # print(predict_path)
    mask_binary = (image_mask >125).flatten().astype('float64')
    # predict不是256 *256的情况
    # predict = Image.open(predict_path).resize((256,256),Image.BICUBIC)
    # predict = np.array(predict)
    # pred_binary  = (predict >= 125).flatten().astype('float64')
    #predict是256 * 256的情况
    predict = cv2.imread(predict_path,0)
    pred_binary = (predict >= 125).flatten().astype('float64')
    mask_binary = torch.from_numpy(mask_binary).float()
    mask_binary_inverse = (mask_binary == 0).float()

    pred_binary = torch.from_numpy(pred_binary).float()
    pred_binary_inverse = (pred_binary == 0).float()
    TP = pred_binary.mul(mask_binary).sum()
    FP = pred_binary.mul(mask_binary_inverse).sum()
    TN = pred_binary_inverse.mul(mask_binary_inverse).sum()
    FN = pred_binary_inverse.mul(mask_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1])
    # 在numpy中计算hd
    pred = torch.from_numpy((predict >0.5).astype('float64'))
    mask = (image_mask >125).astype('float64')
    if torch.sum(pred) == 0:
        pred[100][100] = 1
    pred_to_mask = np.percentile(edt(np.logical_not(mask))[np.nonzero(pred.numpy())], 95)
    mask_to_pred = np.percentile(edt(np.logical_not(pred.numpy()))[np.nonzero(mask)], 95)
    hd = max(pred_to_mask, mask_to_pred)
    Recall = TP / (TP + FN)
    Specificity = TN / (TN +FP)
    Precision = TP / (TP + FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU = (TP / (TP + FP + FN))
    MAE = torch.abs(pred_binary - mask_binary).mean()#错误率或差异率
    DICE = 2 * IoU / (IoU +1)
    fpr, tpr, threshold = roc_curve(mask_binary.cpu().numpy().flatten(), pred_binary.cpu().numpy().flatten())
    auc_roc = auc(fpr, tpr)

    return Precision, Recall, Specificity, F1, auc_roc, accuracy, IoU, DICE, MAE, hd
mask_dir = r'/data_chi/wubo/data/ROI/'
pre_dir = r'/data_chi/wubo/data/US/predict/'
files = os.listdir(mask_dir)
metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
for file in files:
    print(file)
    mask_path = mask_dir + file
    predict_path = pre_dir + file
    _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(mask_path, predict_path)
    metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                   F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, auc=_auc, hd=_hd)
metrics_result = metrics.mean(len(files))
values_txt = 'TN3K_de_de_sch1(SAM)' + '\t'
for k, v in metrics_result.items():
    if k != 'hd':
        v = 100 * v
    # keys_txt += k + '\t'
    values_txt += '%.2f' % v + '\t'
text = values_txt + '\n'
with open('iou_compare/iou1.txt', 'a+') as f:
=======
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch
import os
from metrics import Metrics

def evaluate(mask_path,predict_path):
    image_mask = cv2.imread(mask_path,0)
    # print(mask_path)
    # print(predict_path)
    mask_binary = (image_mask >125).flatten().astype('float64')
    # predict不是256 *256的情况
    # predict = Image.open(predict_path).resize((256,256),Image.BICUBIC)
    # predict = np.array(predict)
    # pred_binary  = (predict >= 125).flatten().astype('float64')
    #predict是256 * 256的情况
    predict = cv2.imread(predict_path,0)
    pred_binary = (predict >= 125).flatten().astype('float64')
    mask_binary = torch.from_numpy(mask_binary).float()
    mask_binary_inverse = (mask_binary == 0).float()

    pred_binary = torch.from_numpy(pred_binary).float()
    pred_binary_inverse = (pred_binary == 0).float()
    TP = pred_binary.mul(mask_binary).sum()
    FP = pred_binary.mul(mask_binary_inverse).sum()
    TN = pred_binary_inverse.mul(mask_binary_inverse).sum()
    FN = pred_binary_inverse.mul(mask_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1])
    # 在numpy中计算hd
    pred = torch.from_numpy((predict >0.5).astype('float64'))
    mask = (image_mask >125).astype('float64')
    if torch.sum(pred) == 0:
        pred[100][100] = 1
    pred_to_mask = np.percentile(edt(np.logical_not(mask))[np.nonzero(pred.numpy())], 95)
    mask_to_pred = np.percentile(edt(np.logical_not(pred.numpy()))[np.nonzero(mask)], 95)
    hd = max(pred_to_mask, mask_to_pred)
    Recall = TP / (TP + FN)
    Specificity = TN / (TN +FP)
    Precision = TP / (TP + FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU = (TP / (TP + FP + FN))
    MAE = torch.abs(pred_binary - mask_binary).mean()#错误率或差异率
    DICE = 2 * IoU / (IoU +1)
    fpr, tpr, threshold = roc_curve(mask_binary.cpu().numpy().flatten(), pred_binary.cpu().numpy().flatten())
    auc_roc = auc(fpr, tpr)

    return Precision, Recall, Specificity, F1, auc_roc, accuracy, IoU, DICE, MAE, hd
mask_dir = r'/data_chi/wubo/data/ROI/'
pre_dir = r'/data_chi/wubo/data/US/predict/'
files = os.listdir(mask_dir)
metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
for file in files:
    print(file)
    mask_path = mask_dir + file
    predict_path = pre_dir + file
    _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(mask_path, predict_path)
    metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                   F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, auc=_auc, hd=_hd)
metrics_result = metrics.mean(len(files))
values_txt = 'TN3K_de_de_sch1(SAM)' + '\t'
for k, v in metrics_result.items():
    if k != 'hd':
        v = 100 * v
    # keys_txt += k + '\t'
    values_txt += '%.2f' % v + '\t'
text = values_txt + '\n'
with open('iou_compare/iou1.txt', 'a+') as f:
>>>>>>> 661c694 ('init')
    f.write(text)