<<<<<<< HEAD
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
from PIL import Image
from sklearn.metrics import jaccard_score
import SimpleITK as sitk
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch

f = torch.cuda.is_available()
device = torch.device("cuda" if f else "cpu")
def get_iou(mask_name,predict):#预测的mask是一个概率图，需要二值化0，1

    image_mask = cv2.imread(mask_name,0)
    mask_org = Image.fromarray(image_mask)
    image_mask = np.array(mask_org.resize((256,256), Image.BICUBIC))#image_mask和predict都是2维数组
    image_mask = (image_mask > 125).flatten().astype(np.int16)#要把元素转换为整形否则无法用jaccard_score计算
    # image_mask = np.array(Image.open(mask_name)).astype('float32') / 255
    # image_mask = image_mask.flatten().astype(np.int16)
    # predict = np.around(predict.flatten())#flatten把2维数组展成一维数组,around把0.6以下都变成0
    predict = (predict>0.6).flatten().astype(np.int16)
    iou_tem = jaccard_score(y_pred=predict, y_true=image_mask)#只能算numpy,计算二分类metrics的均值，为每个类给出相同权重的分值

    return iou_tem

def xy_iou(mask,predict):#预测的mask是一个概率图，需要二值化0，1

    mask = mask.flatten().astype(np.int16)
    predict = np.around(predict.flatten())#flatten把2维数组展成一维数组,around把0.6以下都变成0
    iou = jaccard_score(y_pred=predict, y_true=mask)#计算二分类metrics的均值，为每个类给出相同权重的分值

    return iou

class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics

def evaluate(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    mask_org = Image.fromarray(image_mask)
    image_mask = np.array(mask_org.resize((256, 256), Image.BICUBIC))  # image_mask和predict都是2维数组
    #将预测值和mask二值化并flatten()至一维
    mask_binary = (image_mask >125).flatten().astype('float64')  # 要把元素转换为整形否则无法用jaccard_score计算
    pred_binary = (predict >= 0.5).flatten().astype('float64')
    mask_binary = torch.from_numpy(mask_binary).float()
    # 不要用if else，用这种除法比较快
    # mask_binary = (mask >= 0.5).float()
    mask_binary_inverse = (mask_binary == 0).float()
    # pred = np.around(predict.flatten())

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
    # print(hd)
    Recall = TP / (TP + FN)#召回率，找回的正样本占mask的正样本的比例
    #可以找到更多的正样本，但同时容易误判负样本
    Specificity = TN / (TN +FP)#特异率，找回的负样本占mask的负样本的比例
    #可以找到更多的负样本，但同时容易误判正样本
    Precision = TP / (TP + FP)#准确率，预测正样本的正确比例
    #保证找到的正样本有较高的置信度
    F1 = 2 * Precision * Recall / (Precision + Recall)#与Dice相同

    accuracy = (TP + TN) / (TP + FP + FN + TN)#精度（正确率），预测样本中的正确比例

    IoU = (TP / (TP + FP + FN))

    MAE = torch.abs(pred_binary - mask_binary).mean()#错误率或差异率

    DICE = 2 * IoU / (IoU +1)

    fpr, tpr, threshold = roc_curve(mask_binary.cpu().numpy().flatten(), pred_binary.cpu().numpy().flatten())
    auc_roc = auc(fpr, tpr)

    return Precision, Recall, Specificity, F1, auc_roc, accuracy, IoU, DICE, MAE, hd


# if __name__ == '__main__':
#     mask = torch.sigmoid(torch.randn(65536))
#     pred = torch.sigmoid(torch.randn(65536))
#     print(evaluate(mask,pred))
=======
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
from PIL import Image
from sklearn.metrics import jaccard_score
import SimpleITK as sitk
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch

f = torch.cuda.is_available()
device = torch.device("cuda" if f else "cpu")
def get_iou(mask_name,predict):#预测的mask是一个概率图，需要二值化0，1

    image_mask = cv2.imread(mask_name,0)
    mask_org = Image.fromarray(image_mask)
    image_mask = np.array(mask_org.resize((256,256), Image.BICUBIC))#image_mask和predict都是2维数组
    image_mask = (image_mask > 125).flatten().astype(np.int16)#要把元素转换为整形否则无法用jaccard_score计算
    # image_mask = np.array(Image.open(mask_name)).astype('float32') / 255
    # image_mask = image_mask.flatten().astype(np.int16)
    # predict = np.around(predict.flatten())#flatten把2维数组展成一维数组,around把0.6以下都变成0
    predict = (predict>0.6).flatten().astype(np.int16)
    iou_tem = jaccard_score(y_pred=predict, y_true=image_mask)#只能算numpy,计算二分类metrics的均值，为每个类给出相同权重的分值

    return iou_tem

def xy_iou(mask,predict):#预测的mask是一个概率图，需要二值化0，1

    mask = mask.flatten().astype(np.int16)
    predict = np.around(predict.flatten())#flatten把2维数组展成一维数组,around把0.6以下都变成0
    iou = jaccard_score(y_pred=predict, y_true=mask)#计算二分类metrics的均值，为每个类给出相同权重的分值

    return iou

class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics

def evaluate(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    mask_org = Image.fromarray(image_mask)
    image_mask = np.array(mask_org.resize((256, 256), Image.BICUBIC))  # image_mask和predict都是2维数组
    #将预测值和mask二值化并flatten()至一维
    mask_binary = (image_mask >125).flatten().astype('float64')  # 要把元素转换为整形否则无法用jaccard_score计算
    pred_binary = (predict >= 0.5).flatten().astype('float64')
    mask_binary = torch.from_numpy(mask_binary).float()
    # 不要用if else，用这种除法比较快
    # mask_binary = (mask >= 0.5).float()
    mask_binary_inverse = (mask_binary == 0).float()
    # pred = np.around(predict.flatten())

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
    # print(hd)
    Recall = TP / (TP + FN)#召回率，找回的正样本占mask的正样本的比例
    #可以找到更多的正样本，但同时容易误判负样本
    Specificity = TN / (TN +FP)#特异率，找回的负样本占mask的负样本的比例
    #可以找到更多的负样本，但同时容易误判正样本
    Precision = TP / (TP + FP)#准确率，预测正样本的正确比例
    #保证找到的正样本有较高的置信度
    F1 = 2 * Precision * Recall / (Precision + Recall)#与Dice相同

    accuracy = (TP + TN) / (TP + FP + FN + TN)#精度（正确率），预测样本中的正确比例

    IoU = (TP / (TP + FP + FN))

    MAE = torch.abs(pred_binary - mask_binary).mean()#错误率或差异率

    DICE = 2 * IoU / (IoU +1)

    fpr, tpr, threshold = roc_curve(mask_binary.cpu().numpy().flatten(), pred_binary.cpu().numpy().flatten())
    auc_roc = auc(fpr, tpr)

    return Precision, Recall, Specificity, F1, auc_roc, accuracy, IoU, DICE, MAE, hd


# if __name__ == '__main__':
#     mask = torch.sigmoid(torch.randn(65536))
#     pred = torch.sigmoid(torch.randn(65536))
#     print(evaluate(mask,pred))
>>>>>>> 661c694 ('init')
