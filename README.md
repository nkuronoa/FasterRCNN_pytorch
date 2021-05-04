# FasterRCNN_pytorch

本プロジェクトは、pytorchを用いた物体検出のサンプルプロジェクトです。

ベースのスクリプトは下記のpytorchチュートリアルのものになり、  
物体検出にFasterRCNNを使い、データセットは別の公開データセットを用いています。

TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL  
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


# 使い方
1. データセット作成  
下記コマンドを実行して、dataフォルダ以下にデータセットを保存します  
$ python make_dataset/src/main.py -model train  
$ python make_dataset/src/main.py -model test  

2. 学習  
下記コマンドを実行して、モデルを学習します  
$ python main/src/main.py  

3. テスト  
下記コマンドを実行して、テストデータでモデルの評価を行います  
$ python main/src/eval.py  
