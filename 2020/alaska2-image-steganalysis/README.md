# kaggle ALASKA2 Image Steganalysis

2020/4/28~2020/7/21<br>
https://www.kaggle.com/c/alaska2-image-steganalysis

## Explanatory Data Analysis

## Data Augmetation

拡張の候補は以下の通りである。
- transforms.CenterCrop()
- transforms.RandomCrop()
- transforms.RandomHorizontalFlip()

CropはResize以外の方法で画像サイズを調整するために使用する。
Flipは純粋にデータを水増しするために使用する。

'''python
transform = transforms.Compose([
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
'''

## Model

3つの異なるアルゴリズムで作られたステゴ画像がある。
- JMiPOD
- JUNIWARD
- UERD

ここでモデルの作り方には大きく分けて2つの選択肢がある。
- 2値分類モデルを3つ作る。
- マルチクラス分類モデルを1つ作る。

今回は後者を選ぶ。前者と比べ、モデルの交換や追加のチューニングが容易になる。

'''python
import sys
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
from efficientnet_pytorch import EfficientNet
'''

## Training



## Local Score



## Public Leaderboard



## Private Leaderboard



## Ranking


## License

https://github.com/pudae/kaggle-hpa

