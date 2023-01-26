# CV05 Semantic Segmentation

## 재활용 품목 분류를 위한 Semantic Segmentation

기간 : 2022.12.19 ~ 2023.01.05

주관 : 네이버 커넥트 재단

### 대회 소개

![7645ad37-9853-4a85-b0a8-f0f151ef05be](https://user-images.githubusercontent.com/62612606/214768120-945997c1-5195-4570-929b-3cf1f83087a5.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

**Data** 

11 class : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

이미지 크기 : (512, 512)

Train : 3272장, Test : 819장

**Evaluation Metric**

mIoU(Mean Intersection over Union)

## Members


김도윤: Augmentation/Model/Loss 실험, Ensemble 진행

김형석: Augmentation 및 Loss 실험, SMP 기반 Baseline code 작성

박근태: EDA, CV strategy, Pseudo Update K-fold, Augmentation 실험

양윤석 : Augmentation 및 Model 실험

정선규 : Augmentation 및 Model 실험

## Project process & Experiment

**Process**

EDA - class 분포, 이미지 당 포함하고 있는 class 개수, 이미지 크기 대비 객체 크기 분포, class 별 객체 크기 분포, 이미지 내 객체 영역 비율 분포, 객체 중심 좌표 분포 분석

CV strategy - test set을 잘 대변하는 validation set 구축

EDA 결과, class 분포 뿐 아니라 이미지 별 객체 영역 크기, 객체 개수, 객체 평균 크기가 매우 상이하였기에,  Multilabel Stratified k fold를 이용해 class, 객체 영역 크기, 객체 개수, 객체 평균 크기 모두 고려한 val set 추출

Pseudo Update K-fold - train set split의 학습과정에 Pseudo labeling을 추가

![화면 캡처 2023-01-26 145659](https://user-images.githubusercontent.com/62612606/214768201-545ca5ab-e0dc-4c9e-a203-0d38037e7c02.png)


**Experiment**

| Model | Validation | Test |
| --- | --- | --- |
| Unet efficient b0 | 0.5759 | 0.5381 |
| Upernet Swin base | 0.6935 | 0.7053 |
| Upernet Swin large(pseudo update k fold) | - | 0.7412 |
| Upernet convnext xl | 0.7329 | 0.748 |
| PAN Swin_large | 0.7211 | 0.7138 |

## Models & Schedulers


| Model | Schedulers | Optimizers | Loss |
| --- | --- | --- | --- |
| PAN Swin_large | CosineAnnealingWarmUpRestarts | AdamP | DiceFocalLoss |
| Upernet Swin large(pseudo update k fold) | CosineAnnealingWarmUpRestarts | AdamW | CrossEntropyLoss |
| UpernetConvnext_xl | Poly | AdamW | CrossEntropyLoss |

## TTA & Ensemble


TTA

- Multiscale : img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
- Horizontal Flip , Vertical Flip

Ensemble -4가지의 model을 활용하여 ensemble(soft voting, hard voting)

- PAN Swin_large
- Upernet Swin large(pseudo update k fold)
- Upernet Convnext_xl K-fold
- Upernet Convnext_xl Pseudo labeling

## Rusult


![화면 캡처 2023-01-26 145728](https://user-images.githubusercontent.com/62612606/214768166-b3babf45-af74-4837-a1f4-a7ff4f4abc7c.png)


Public : 0.7676 (19팀 중 8등), Private : 0.7638 (19팀 중 5등)

## Citation

**MMsegmentation**

```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
