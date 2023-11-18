# NICE MNIST

手書き数字文字のMNISTを使った、NNを使った生成手法の一つであるNICEの勉強用。   
コード[github repository](https://github.com/fmu2/NICE)を自分の環境で動くように改造したもの。  

## 追加した引数

coupling層の数を指定する。  
```
python --coupling 2 
```

## 内容  
  
coupling層が2のとき、1000回から15000回まで学習したとき、1000回毎の乱数から生成されたsamplesと手書き文字を分析して生成したreconstructionを収納。  

## ライセンス  
  
オリジナルコードのライセンスはLICENSE-NICE-master.txtを参照のこと。  





