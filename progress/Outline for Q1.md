# Outline for Q1

## 1. No-patch

- Data preprocessing method


- - 0-255 grey-scale normalize/non-normalize


- Feature extraction method


- - Tight frame — 54 dimensions each sample


- Feature preprocessing method


- - (21 x 54) matrix standardize/non-standardize

*So totally we have four groups of different features to train in each model*

## 2. 16-patch

*The same as No-patch frame where the only difference is that the input images are 16 times larger in number*

## 3. Best results for different single models

| Models             | Predict accuracy | Parameters                         | Features          |
| ------------------ | ---------------- | ---------------------------------- | ----------------- |
| Forward stage-wise | **0.86**         | {32, 33, 34, 24, 28, 30}           | No_patch_std      |
| Forward stage-wise | **0.90**         | {0, 2, 36, 39, 49, 18, 19, 20, 21} | No_patch_norm_std |
| SVM                | 0.67             | (c=0.1, d=1, k='linear')           | Patch_std         |
| KNN                | **0.90**         | (n=[1-10], p=[1-2] )               | Patch_std         |
| DT                 | **0.86** / 0.76  | () / (l=2, n=4)                    | Patch_std         |
| SVM                | 0.62             | (c=0.1, d=7, k='poly')             | No_patch_std      |
| KNN                | 0.71             | (n=3, p=1)                         | No_patch_std      |
| DT                 | 0.71             | (l=2, n=2)                         | No_patch_std      |
| SVM                | 0.62             | (c=0.1, d=1, k='linear')           | No_patch_norm_std |
| KNN                | 0.71             | (n=3, p=1)                         | No_patch_norm_std |
| DT                 | 0.81             | (l=2, n=2)                         | No_patch_norm_std |

## 4. Other potential ideas

### 4.1 Using ResNet + fine-tuning



### 4.2 Using Combined models or Voting model

