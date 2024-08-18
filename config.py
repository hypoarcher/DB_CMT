<<<<<<< HEAD
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='/data_chi/wubo/data/US')
    parser.add_argument('--maskpath', type=str, default='/data_chi/wubo/data/ROI')
    parser.add_argument('--videopath', type=str, default='/data_chi/wubo/data/CEUS/video')
    parser.add_argument('--csvpath', type=str, default='/data_chi/wubo/data/CEUS/TNUS.csv')
    parser.add_argument('--model_name', type=str, default='DB_CMT')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--writer_comment', type=str, default='tnus')
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--patch_size_v', type=list, default=[2, 14, 14])
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_blocks', type=list, default=[2, 2, 2, 2])

    parser.add_argument('--loss_function', type=str, default='Focal')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)


    config = parser.parse_args()
    return config
=======
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='/data_chi/wubo/data/US/img')
    parser.add_argument('--maskpath', type=str, default='/data_chi/wubo/data/ROI')
    parser.add_argument('--videopath', type=str, default='/data_chi/wubo/data/CEUS/video')
    parser.add_argument('--csvpath', type=str, default='/data_chi/wubo/data/CEUS/TNUS.csv')
    parser.add_argument('--model_name', type=str, default='DB_CMT')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--writer_comment', type=str, default='tnus')
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--patch_size_v', type=list, default=[2, 14, 14])
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_blocks', type=list, default=[2, 2, 2, 2])

    parser.add_argument('--loss_function', type=str, default='Focal')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)


    config = parser.parse_args()
    return config
>>>>>>> 661c694 ('init')
