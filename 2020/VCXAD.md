# VCXAD

## Overview

開始日	2020/12/31
終了日	2021/03/31
主催者	VinLab
タイトル	VinBigData Chest X-ray Abnormalities Detection
サブタイトル	Automatically localize and classify thoracic abnormalities from chest radiographs
URL	https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
動機	"現在、画像ごとに分類することはできるが、
場所の特定ができておらず、説明ができない。
そのため、物体検出タスクで研究をしたい。"
データセット	
評価指標	
"提案されている
アプローチ"	Object Detection

## memo

訓練データセットは	
3人の医師が別々にアノテーションしたデータをもつ	
	
テストデータセットは	
5人の医師の合意によってアノテーションされたデータとされている	
	
つまり、	
3人の医師のアノテーションデータをそのまま使って、3人の医師の予測を得るのではなく、	
事前にアノテーションデータを統合し、統合された予測を得る必要がある。	
（コンペの成果として期待されていることの1つ）	
最終的に3つのモデルを作成し、アンサンブルするという手もある。	
	
医師ごとに別々の画像とみなすと、	
損失計算の仕様で上手く学習されない恐れがある。	
	
方針	
	所見0~13をクラス0に、所見14をクラス1に変換した上で、
	1クラスの異常を検出するモデルで、
	まずはbounding boxが上手く検出できることを目指す。
	
精度向上案	
	骨格が白の画像と黒の画像があるので、片方に統一する
	DICOMの属性で確認できないか？

## Class

| No | Name | 初見 | 初見説明 |
|:--|:--|:--|:--|
| 0 | Aortic | enlargement | 大動脈拡大 | "大動脈の径が正常範囲よりも太い場合をいいます。状態によって精密検査を指示することがあります。" | https://www.jpm1960.org/exam/exam10.html |
| 1 | Atelectasis | 無気肺 | "気管支が肺腫瘍や炎症、異物などにより閉塞し、空気の出入りがなくなったために肺胞から肺胞気が抜けて部分的に肺が縮んだ状態です(閉塞性無気肺)。有効な化学療法のなかった時代に罹って治った肺結核には、広範に肺が線維化を起こして縮んでいることがあります（瘢痕性無気肺）。" | https://www.ningen-dock.jp/public/inspection/chest-x |
| 2 | Calcification | 石灰化 | "石灰化とは軟部組織にカルシウム塩が沈着する現象あるいは沈着した状態。様々な生物で見られ、結果として硬化した組織などが形成される。 | "	https://ja.wikipedia.org/wiki/%E7%9F%B3%E7%81%B0%E5%8C%96 |
| 3 | Cardiomegaly | 心臓肥大 | 心臓肥大とは心臓の筋肉が普通より厚くなった状態をいう。 | https://ja.wikipedia.org/wiki/%E5%BF%83%E8%87%93%E8%82%A5%E5%A4%A77 |
| 4 | Consolidation | コンソリデーション | コンソリデーションは、「べたっと均質な、真っ白の陰影」を表す用語です。 | http://tnagao.sblo.jp/article/179695032html#:~:text=%E3%82%B3%E3%83%B3%E3%82%BD%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AF%E3%80%81%E3%80%8C%E3%81%B9%E3%81%9F%E3%81%A3,%E9%99%B0%E5%BD%B1%E3%80%8D%E3%82%92%E8%A1%A8%E3%81%99%E7%94%A8%E8%AA%9E%E3%81%A7%E3%81%99%E3%80%82 |
| 5 | ILD | 間質性肺疾患 | 両肺の広範囲に陰影が出現する疾患 | https://himeji.hosp.go.jp/dep/kona/ip.html |
| 6 | Infiltration | 浸潤影 | "肺胞内への細胞成分や液体成分が入り込んで生じる境界の不明確な陰影をいいます。肺炎、肺結核など肺感染症に見られます。" | https://www.ningen-dock.jp/public/inspection/chest-x |
| 7 | Lung Opacity | 肺の不透明度 |  |  |
| 8 | Nodule/Mass | 限局性肺野病変 | "通常、限局した円形の肺野病変を限局性肺野病変というが、境界の不鮮明な斑状または索状の陰影を呈することも多々ある。したがって、限局性陰影で、輪郭や境界の特徴、均一性に関わらず、径が3cm以下の陰影を結節影(nodule)、それ以上の大きさの陰影を腫？影(mass)とよぶ。 | "	http://www.iryokagaku.co.jp/frame/03-honwosagasu/361/168-173.pdf |
| 9 | Other lesion | その他の病変 |  | |
| 10 | Pleural effusion | 胸水 | "胸部に通常存在しない水がたまった状態です。心不全、腎不全、胸膜炎などの場合に見られます。" | https://www.ningen-dock.jp/public/inspection/chest-x |
| 11 | Pleural thickening | 胸膜肥厚 | "胸膜とは肺を包む膜で、その厚みが異常に増した状態が胸膜肥厚です。細菌やウィルス等による炎症が治癒した跡で、治癒像のひとつです。大半は心配のない所見（有所見健康）ですが、治癒像と言い切れないときに要経過観察または要精密検査とすることがあります。<br>肺を包む胸膜が厚くなった状態です。過去の胸膜炎、肺感染症などが考えられます。" | https://www.jpm1960.org/exam/exam10.html<br>https://www.ningen-dock.jp/public/inspection/chest-x |
| 12 | Pneumothorax | 気胸 | "胸膜の一部が破れ、空気が胸腔内に漏れ出て、肺が圧迫された状態です。治療が必要です。呼吸器科専門医を受診してください"<br>"肺胞という袋状の組織が融合した大きな袋が破れる病気です。ブラという空気の袋の破裂などが原因で起こります。その結果、肺から空気が抜けて萎んだ状態（肺虚脱）となり、胸部エックス線検査では虚脱した肺と胸腔内に空気の溜まりとして認められます。胸腔内圧が上昇する緊張性気胸では、縦隔部が圧排されて反対側に偏位し横隔膜が押し下げられます。" | https://www.jpm1960.org/exam/exam10.html<br>https://www.ningen-dock.jp/public/inspection/chest-x |
| 13 | Pulmonary fibrosis | 肺線維症 | "間質性肺炎は特発性びまん性と、原因のある２次性に分けられます。治療法の選択には肺生検による病理的な診断が重要です。その中で肺線維症は広範囲に進行したもので臨床的には不可逆性です。不整形陰影、網状影、多発輪状影、蜂巣、蜂窩肺が見られます。" | https://www.ningen-dock.jp/public/inspection/chest-x |
| 14 | No finding | 初見なし |

## Balance

0	Aortic enlargement	7,162	11%
1	Atelectasis	279	0%
2	Calcification	960	1%
3	Cardiomegaly	5,427	8%
4	Consolidation	556	1%
5	ILD	1,000	1%
6	Infiltration	1,247	2%
7	Lung Opacity	2,483	4%
8	Nodule/Mass	2,580	4%
9	Other lesion	2,203	3%
10	Pleural effusion	2,476	4%
11	Pleural thickening	4,842	7%
12	Pneumothorax	226	0%
13	Pulmonary fibrosis	4,655	7%
14	No finding	31,818	47%
		67,914	100%

## Columns

Name	Detail
image_id	unique image identifier
class_name	the name of the class of detected object (or "No finding")
class_id	the ID of the class of detected object
rad_id	the ID of the radiologist that made the observation
x_min	minimum X coordinate of the object's bounding box
y_min	minimum Y coordinate of the object's bounding box
x_max	maximum X coordinate of the object's bounding box
y_max	maximum Y coordinate of the object's bounding box

## Evaluation

The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.4.
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview/evaluation
http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf
