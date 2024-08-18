### DB_CMT
A Dual-Branch Cross-Modality-Attention Transformer Network for Thyroid Nodule Diagnosis based on Contrast-Enhanced Ultrasound Videos
![image](https://github.com/hypoarcher/DB_CMT/blob/main/DB_CMT.jpg)
## Introduction
The implementation of:   
A Dual-Branch Cross-Modality-Attention Transformer Network for Thyroid Nodule Diagnosis based on Contrast-Enhanced Ultrasound Video
## Requirements
* python 3.9  
* pytorch 2.1.0 + CUDA11.8  
* torchvision 0.16.0  
* pandas 2.0.3  
* scipy 1.10.1  
* sklearn 1.2.2  
## Setup
# Installation
Clone the repo and install required packages:  
```git clone https://github.com/hypoarcher/DB_CMT.git```

# Dataset
* The format of my dataset is as follows:  
```./data  
├── CEUS  
│   ├── video  
│   │   ├── benign(0).mp4  
│   │   ├── benign(1).mp4  
│   │   ├── ...  
│   └── label.csv  
├── ROI  
│   ├── benign(0).png  
│   ├── benign(1).png  
│   ├── ...  
└── US  
    ├── benign(0).png  
    ├── benign(1).png  
    ├── ...
```

* The format of the TNUS.csv is as follows:
```
|name            |label|
|----------------|-----|
|benign(0).mp4   |  0  |
|benign(1).mp4   |  0  |
|malignant(0).mp4|  1  |
...
```
## Training
```
python train.py --imgpath /data_chi/wubo/data/US/img --maskpath /data_chi/wubo/data/ROI --videopath /data_chi/wubo/data/CEUS/video --csvpath /data_chi/wubo/data/CEUS/label.csv --batch_size 16 --class_num 2 --epochs 250 --lr 0.0001  
```



  
    


  
