# Riemann Zeta 関数の偏角の原理の計算  

## 概要  
Riemann Zeta 関数の、□領域を区分求積法で、偏角の原理を計算する。  
積分の結果は、□領域に含まれる　「零点の数　－　極の数」　になる。  
実際の計算には誤差があるため、ぴったり整数にはならない。  
Riemann Zeta 関数とその導関数の計算は、mpmathを利用した。  

## 使い方  
### 最初の非自明な零点を５個含む□領域の計算  
```
python test1.py  
tate yoko position [0.1, 33.0] [0.0, 1.0]  
sekibun step 0.01  
tate yoko bunkatu number  3290 100  
sekibun (4.99999386888865 + 0.00026736587192154j)  
```

### 積分のステップを小さく(0.01から0.001)して計算精度を上げたもの　計算時間が掛かる
``` 
python test1.py -d 0.001  
tate yoko position [0.1, 33.0] [0.0, 1.0]  
sekibun step 0.001  
tate yoko bunkatu number  32900 1000  
sekibun (4.9999999386888 + 2.67364609117596e-6j)  
```

### 1+j0の極を含む□領域の計算　1個減って4になる  
```
python test1.py -y 1.1 -a -0.1
tate yoko position [-0.1, 33.0] [0.0, 1.1]
sekibun step 0.01
tate yoko bunkatu number  3310 110
sekibun (4.00012873370994 + 5.0342983213897e-6j)
```

### j100までの□領域の零点の計算　零点の数は29個となった  
```
python test1.py -b 100
tate yoko position [0.1, 100.0] [0.0, 1.0]
sekibun step 0.01
tate yoko bunkatu number  9990 100
sekibun (28.9999988534685 + 0.000267366957181842j)
```


## ノートブック  
[google colabで計算させるためのノートブック](https://colab.research.google.com/github/shun60s/misc-tools/blob/master/Riemann_Zeta_piecewise_quadrature/Riemann_Zeta_piecewise_quadrature_1.ipynb)  


## 参考リンク  
[Riemann Zeta 関数の零点のリスト](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/)  
