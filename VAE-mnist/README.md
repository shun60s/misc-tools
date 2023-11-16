# VAE MNIST

手書き数字文字のMNISTを使ったVAEの勉強用。   
[【徹底解説】 VAEをはじめからていねいに](https://academ-aid.com/ml/vae)で説明されていたコード[github repository](https://github.com/beginaid/VAE)を自分の環境で動くように改造したもの。  

## 追加した引数

潜在変数latent variableの次元を指定する。  
```
python3 main.py --z-dim 2  
```

前回の学習したDICをロードする。学習をスキップして、imagesを描く。  
```
python3 main.py --load-model  --skip-train  
```

##　ライセンス  
  
オリジナルコードのライセンスはLICENSE-VAE-main.txtを参照のこと。  





