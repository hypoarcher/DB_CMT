<<<<<<< HEAD
import torch
import os
import torch.nn as nn
import torchvision.models
import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from valid import valid
from utils import confusion_matrix
import math
import tqdm
from model import DbcmtNet
from loss_function import BCEFocalLoss, Generalized_Dice_Loss


def train(config, train_loader, test_loader, fold, test_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL
    # model = create_model(img_size=config.img_size, num_classes=config.class_num, drop_rate=0.1, attn_drop_rate=0.1,
    #         patch_size=config.patch_size, dim=config.dim, depth=config.depth, num_heads=config.num_heads,
    #         num_inner_head=config.num_inner_head)

    model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14, patch_size_v=[2, 14, 14], num_layers=1,
                     num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
    model = model.to(device)
    model.train()

    if config.loss_function == 'Focal':
        criterion_cls = BCEFocalLoss().to(device)
        criterion_seg = Generalized_Dice_Loss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (
                        config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_' + config.model_name + '_' + config.writer_comment + '_' + str(fold))

    print("START TRAINING")
    best_acc = 0
    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path, str(fold))
    cm = None
    for epoch in range(config.epochs):
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0

        for i, pack in enumerate(train_loader):
            frames = pack['videos'].to(device)
            image = pack['images'].to(device)
            mask = pack['masks'].to(device)
            if frames.shape[1] == 1:
                frames = frames.expand((-1, 3, -1, -1, -1))

            if image.shape[1] == 1:
                image = image.expand((-1, 3, -1, -1))
            label = pack['labels'].to(device)
            names = pack['names']

            out_cls, out_seg = model(image, frames)
            # if epoch <= 4:
            #     output = output.logits  # GOOGLENET需要

            loss_cls = criterion_cls(out_cls, label)
            loss_seg = criterion_seg(out_seg, mask)
            loss = loss_cls + 0.5 * loss_seg

            pred = out_cls.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), label.detach(), cm)
        lr_scheduler.step()
        if (epoch + 1) % config.log_step == 0:
            print('[epoch %d]' % epoch)
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion_cls, criterion_seg)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)

            if epoch > config.epochs // 4:
                if val_acc > best_acc:
                    best_acc = val_acc
                    print("=> saved best model")
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'bestmodel.pth'))
                    with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc, spe, sen, auc, pre, f1score))
        if epoch + 1 == config.epochs:
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion_cls, criterion_seg)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write('\nLast Result:\n')
                f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                        % (val_acc, spe, sen, auc, pre, f1score))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Avg Epoch Loss: %.4f' % (epoch + 1, config.epochs, avg_epoch_loss))
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)


    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch(42)
    args = config()
    cv=KFold(n_splits=args.fold, random_state=42, shuffle=True)
    fold=0
    train_set = utils.get_dataset(args.imgpath, args.maskpath, args.videopath, args.csvpath, args.img_size, mode='train')
    test_set = utils.get_dataset(args.imgpath, args.maskpath, args.videopath, args.csvpath, args.img_size, mode='train')
    print(args)
    argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
    if not os.path.exists(argspath):
        os.makedirs(argspath)
    with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
        f.write(str(args))

    for train_idx,test_idx in cv.split(train_set):
        print("\nCross validation fold %d" %fold)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler= SubsetRandomSampler(test_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,sampler=train_sampler, num_workers=6)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)
        train(args, train_loader, test_loader, fold, test_idx)
        fold+=1
    # (随机五折交叉验证)0
    #
    # seed_torch(42)
    # args = config()
    # cv = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    # fold = 0
    # train_set = utils.get_dataset(args.img_path, args.video_path, args.csv_path, args.img_size, mode='train')
    # test_set = utils.get_dataset(args.img_path, args.video_path, args.csv_path, args.img_size, mode='test')
    # print(args)
    # argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
    # if not os.path.exists(argspath):
    #     os.makedirs(argspath)
    # with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
    #     f.write(str(args))
    #
    # for train_idx, test_idx in cv.split(train_set.info, train_set.get_labels()):
    #     print("\nCross validation fold %d" % fold)
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     test_sampler = SubsetRandomSampler(test_idx)
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
    #                               num_workers=6)
    #     test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)
    #     train(args, train_loader, test_loader, fold, test_idx)
    #     fold += 1
















=======
import torch
import os
import torch.nn as nn
import torchvision.models
import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from valid import valid
from utils import confusion_matrix
import math
import tqdm
from model import DbcmtNet
from loss_function import BCEFocalLoss, Generalized_Dice_Loss


def train(config, train_loader, test_loader, fold, test_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL
    # model = create_model(img_size=config.img_size, num_classes=config.class_num, drop_rate=0.1, attn_drop_rate=0.1,
    #         patch_size=config.patch_size, dim=config.dim, depth=config.depth, num_heads=config.num_heads,
    #         num_inner_head=config.num_inner_head)

    model = DbcmtNet(in_ch=3, out_ch=1, num_classes=2, num_blocks=[2, 2, 2, 2], image_size=224, patch_size=14, patch_size_v=[2, 14, 14], num_layers=1,
                     num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1, num_frames=16)
    model = model.to(device)
    model.train()

    if config.loss_function == 'Focal':
        criterion_cls = BCEFocalLoss().to(device)
        criterion_seg = Generalized_Dice_Loss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (
                        config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_' + config.model_name + '_' + config.writer_comment + '_' + str(fold))

    print("START TRAINING")
    best_acc = 0
    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path, str(fold))
    cm = None
    for epoch in range(config.epochs):
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0

        for i, pack in enumerate(train_loader):
            frames = pack['videos'].to(device)
            image = pack['images'].to(device)
            mask = pack['masks'].to(device)
            if frames.shape[1] == 1:
                frames = frames.expand((-1, 3, -1, -1, -1))

            if image.shape[1] == 1:
                image = image.expand((-1, 3, -1, -1))
            label = pack['labels'].to(device)
            names = pack['names']

            out_cls, out_seg = model(image, frames)
            # if epoch <= 4:
            #     output = output.logits  # GOOGLENET需要

            loss_cls = criterion_cls(out_cls, label)
            loss_seg = criterion_seg(out_seg, mask)
            loss = loss_cls + 0.5 * loss_seg

            pred = out_cls.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), label.detach(), cm)
        lr_scheduler.step()
        if (epoch + 1) % config.log_step == 0:
            print('[epoch %d]' % epoch)
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion_cls, criterion_seg)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)

            if epoch > config.epochs // 4:
                if val_acc > best_acc:
                    best_acc = val_acc
                    print("=> saved best model")
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'bestmodel.pth'))
                    with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc, spe, sen, auc, pre, f1score))
        if epoch + 1 == config.epochs:
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion_cls, criterion_seg)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write('\nLast Result:\n')
                f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                        % (val_acc, spe, sen, auc, pre, f1score))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Avg Epoch Loss: %.4f' % (epoch + 1, config.epochs, avg_epoch_loss))
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)


    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch(42)
    args = config()
    cv=KFold(n_splits=args.fold, random_state=42, shuffle=True)
    fold=0
    train_set = utils.get_dataset(args.imgpath, args.maskpath, args.videopath, args.csvpath, args.img_size, mode='train')
    test_set = utils.get_dataset(args.imgpath, args.maskpath, args.videopath, args.csvpath, args.img_size, mode='train')
    print(args)
    argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
    if not os.path.exists(argspath):
        os.makedirs(argspath)
    with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
        f.write(str(args))

    for train_idx,test_idx in cv.split(train_set):
        print("\nCross validation fold %d" %fold)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler= SubsetRandomSampler(test_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,sampler=train_sampler, num_workers=6)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)
        train(args, train_loader, test_loader, fold, test_idx)
        fold+=1
    # (随机五折交叉验证)0
    #
    # seed_torch(42)
    # args = config()
    # cv = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    # fold = 0
    # train_set = utils.get_dataset(args.img_path, args.video_path, args.csv_path, args.img_size, mode='train')
    # test_set = utils.get_dataset(args.img_path, args.video_path, args.csv_path, args.img_size, mode='test')
    # print(args)
    # argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
    # if not os.path.exists(argspath):
    #     os.makedirs(argspath)
    # with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
    #     f.write(str(args))
    #
    # for train_idx, test_idx in cv.split(train_set.info, train_set.get_labels()):
    #     print("\nCross validation fold %d" % fold)
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     test_sampler = SubsetRandomSampler(test_idx)
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
    #                               num_workers=6)
    #     test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)
    #     train(args, train_loader, test_loader, fold, test_idx)
    #     fold += 1
















>>>>>>> 661c694 ('init')
