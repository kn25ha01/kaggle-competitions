# RANZCR

## memo

開始日	2020/12/15
終了日	2021/03/16
主催者	
タイトル	RANZCR CLiP - Catheter and Line Position Challenge
サブタイトル	Classify the presence and correct placement of tubes on chest x-rays to save lives
URL	https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification
動機	"カテーテルのような管の挿入位置が間違っている場合、
患者に合併症や死を招く恐れがある。
しかし、医師や看護師のリソースは限られているし、
ヒューマンエラーも起こり得る。
早期発見のため、RANZCRという組織はシステム化しようとしている。
そのため、今回のコンペでは管の位置を検出するシステムを作ることを目的としている。
配置が間違っている管の分類もする。"
データセット	"train	40,000
test public	140000 * 25% (35,000)
test private	140000 * 75% (105,000)"
評価指標	
"提案されている
アプローチ"	Segmentationではなく画像分類で解こうとしている？
メモ	RANZCR = The Royal Australian and New Zealand College of Radiologists

## Class

StudyInstanceUID		unique ID for each image
ETT	Abnormal	endotracheal tube placement abnormal
ETT	Borderline	endotracheal tube placement borderline abnormal
ETT	Normal	endotracheal tube placement normal
NGT	Abnormal	nasogastric tube placement abnormal
NGT	Borderline	nasogastric tube placement borderline abnormal
NGT	Incompletely Imaged	nasogastric tube placement inconclusive due to imaging
NGT	Normal	nasogastric tube placement borderline normal
CVC	Abnormal	central venous catheter placement abnormal
CVC	Borderline	central venous catheter placement borderline abnormal
CVC	Normal	central venous catheter placement normal
Swan Ganz Catheter Present		
PatientID		unique ID for each patient in the dataset
		
		
ETT, NGT, CVCって何？		
endotracheal tube	気管チューブ・・・口あるいは喉から挿入するタイプ、気道確保に使われる	
nasogastric tube	NGチューブ（経鼻胃管）・・・鼻から胃にかけて挿入するタイプ、胃に直接水分や栄養を送るために使われる	
central venous catheter	中心静脈カテーテル・・・首や鎖骨周りから血中に直接栄養を送るために使われる？	
Swan Ganz Catheter Present	肺動脈カテーテル・・・心臓に向けて挿入するタイプ？新機能の測定に使用される	
