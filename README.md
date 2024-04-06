# [CVPR-2024] _AllSpark_: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-21)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-21?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-4?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-9)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-9?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-44)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-44?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-27)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-27?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-28)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-28?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-29)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-29?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-30)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-30?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-coco)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-coco-1)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco-1?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-coco-2)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco-2?p=allspark-reborn-labeled-features-from)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/allspark-reborn-labeled-features-from/semi-supervised-semantic-segmentation-on-coco-3)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco-3?p=allspark-reborn-labeled-features-from)


This repo is the official implementation of [_AllSpark_: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2403.01818) which is accepted at CVPR-2024.



<p align="center">
<img src="./docs/allspark.jpg" width=39% height=65% class="center">
<img src="./docs/framework.png" width=60% height=65% class="center">
</p>

The _**AllSpark**_ is a powerful Cybertronian artifact in the film series of _Transformers_. It was used to reborn Optimus Prime in _Transformers: Revenge of the Fallen_, which aligns well with our core idea.

-------

## 💥 Motivation
In this work, we discovered that simply converting existing semi-segmentation methods into a pure-transformer framework is ineffective. 
<p align="center">
<img src="./docs/backbone.png" width=50% height=80% class="center">
<img src="./docs/issue.jpg" width=35% height=65% class="center">
</p>

- The first reason is that transformers inherently possess weaker inductive bias compared to CNNs, so transformers heavily rely on a large volume of training data to perform well. 

- The more critical issue lies in the existing semi-supervised segmentation frameworks. These frameworks separate the training flows for labeled and unlabeled data, which aggravates the overfitting issue of transformers on the limited labeled data.

Thus, we propose to _intervene and diversify_ the labeled data flow with unlabeled data in the feature domain, leading to improvements in generalizability.

-------

## 🛠️ Usage

**‼️ IMPORTANT: This version is not the final version. We made some mistakes when re-organized the code. We will release the correct version soon. Sorry for any inconvience may caused.**

### 1. Environment

First, clone this repo:

```shell
git clone https://github.com/xmed-lab/AllSpark.git
cd AllSpark/
```

Then, create a new environment and install the requirements:
```shell
conda create -n allspark python=3.7
conda activate allspark
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard
pip install six
pip install pyyaml
pip install -U openmim
mim install mmcv==1.6.2
pip install einops
pip install timm
```

### 2. Data Preparation & Pre-trained Weights

#### 2.1 Pascal VOC 2012 Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O pascal.zip
unzip pascal.zip
```

#### 2.2 Cityscapes Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EWoa_9YSu6RHlDpRw_eZiPUBjcY0ZU6ZpRCEG0Xp03WFxg\?e\=LtHLyB\&download\=1 -O cityscapes.zip
unzip cityscapes.zip
```

#### 2.3 COCO Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EXCErskA_WFLgGTqOMgHcAABiwH_ncy7IBg7jMYn963BpA\?e\=SQTCWg\&download\=1 -O coco.zip
unzip coco.zip
```


Then your file structure will be like:

```
├── VOC2012
    ├── JPEGImages
    └── SegmentationClass
    
├── cityscapes
    ├── leftImg8bit
    └── gtFine
    
├── coco
    ├── train2017
    ├── val2017
    └── masks
```

Next, download the following [pretrained weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/Eobv9tk6a6RJqGXEDm2D_TcB2mEn4r2-BLDkotZHkd2l6w?e=fJBy7v).
```
├── ./pretrained_weights
    ├── mit_b2.pth
    ├── mit_b3.pth
    ├── mit_b4.pth
    └── mit_b5.pth
```

For example, mit-B5:
```shell
mkdir pretrained_weights
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ET0iubvDmcBGnE43-nPQopMBw9oVLsrynjISyFeGwqXQpw?e=9wXgso\&download\=1 -O ./pretrained_weights/mit_b5.pth
```


### 3. Training & Evaluating

```bash
# use torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 4 on all three datasets
# otherwise, you need to adjust the learning rate accordingly

# or use slurm
# sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/xmed-lab/AllSpark/blob/main/scripts/train.sh).


### 4. Results

Model weights and training logs will be released soon.

#### 4.1 PASCAL VOC 2012 _original_
<p align="left">
<img src="./docs/pascal_org.png" width=60% class="center">
</p>


| Splits | 1/16 | 1/8  | 1/4 | 1/2 | Full |
| :- | - | - | - | - | - |
| Weights of _**AllSpark**_ | 76.07 | 78.41 | 79.77 | 80.75 | 82.12 |
| Reproduced | [76.06](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EZY5whWCIuNOnaAYpPk9cyoBmspntpHoa75z2DAnD1Tg2g?e=AkHect\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ERqwnOSTFyZBlMmLfqA1xegBJDJEW5pqM-WHTG3JMWdwKg?e=2tL8Fs) | [78.41](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ESsfJbP0ipxAmhzzdESOIdgBKv3OLceKhpJscDaxTo9Grg?e=UDxRmb\&download\=1) | [79.93](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ETHrHSQotHVOvKPjZKl3DUEByst-7FHBZFm7lSLt0L71HA?e=qb9sv5\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EeBcskpwXhpJr1839qVe-f4Bjq68paMI0GSxaWJQjnrlQQ?e=V9VPkx) | [80.70](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EasoxicajudMmMmBzDCwNjIBnzg5kXXVga4lsYFIrq587A?e=njs6W7\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EXI_9n636B9Hsql6JXSzT6oBzkweKDqxtLbPf5zrV0NFyA?e=iZTKWy) | [82.56](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/Ec-K5rthA2xLheg0blwnPbkBI2zvYs7xcPdZ7Hr2RBHTag?e=kaiBEE\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcIWs3l1faBKh1HOM29QgmgBPjALel1IG9FPLJaA9xBPZg?e=of3NlB) |


#### 4.2 PASCAL VOC 2012 _augmented_

<p align="left">
<img src="./docs/pascal_aug.png" width=60% class="center">
</p>

| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**AllSpark**_ | 78.32 | 79.98 | 80.42 | 81.14 |




#### 4.3 Cityscapes

<p align="left">
<img src="./docs/cityscapes.png" width=60% class="center">
</p>


| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**AllSpark**_ | 78.33 | 79.24 | 80.56 | 81.39 |



#### 4.4 COCO

<p align="left">
<img src="./docs/coco.png" width=60% class="center">
</p>


| Splits | 1/512 | 1/256  | 1/128 | 1/64 |
| :- | - | - | - | - |
| Weights of _**AllSpark**_ | [34.10](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EaabBYut1PNEtPeQRCIlMtEBxpmkvbZ_ERmBGwTObS0H_g?e=69ToFl\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EbOcn45vPcJGvERaP2tnJdQBLY-xF6GExOP2BsTFQta1xg?e=lSodC0) | [41.65](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EfIyzut1SwBMha25yKpeIWIBwPfhc3NzdGLjdlyuKdr0ig?e=H58uKd\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EU9w0R77DpxOqE8pW62BSh0BC7OZ9Kqml2eluvGftx-fJw?e=9Ugtwc) | [45.48](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EUHmlDEXNPZPuq5qRfhTChgBs9GZ2n9qVRYdPWHGwgkYBQ?e=yRNTcg\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EYIjEixClLVKtMaYJz8KZssB56s5nduIR31dAxUSALWF7A?e=IGRUSs) | [49.56](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ETeZ7agRCkRIjJeONaL8BYEBKIe4rDI3ZgRkEDdBcVPPOA?e=56diA2\&download\=1) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EckkrL4B2HFHtNHY0IBwizYBsU7iSdh5zO36hzPhDUIlDg?e=tA2Eov) |




## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{allspark,
  title={AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation},
  author={Wang, Haonan and Zhang, Qixiang and Li, Yi and Li, Xiaomeng},
  booktitle={CVPR},
  year={2024}
}
```


## Acknowlegement
_**AllSpark**_ is built upon [UniMatch](https://github.com/LiheYoung/UniMatch) and [SegFormer](https://github.com/NVlabs/SegFormer). We thank their authors for making the source code publicly available.


