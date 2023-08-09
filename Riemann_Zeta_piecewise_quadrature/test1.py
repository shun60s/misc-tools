#coding:utf-8

#
# Riemann Zeta 関数の、□領域を区分求積法で、偏角の原理を計算する。
# 積分の結果は、□領域に含まれる　「零点の数　－　極の数」　になる。
# 実際の計算には誤差があるため、ぴったり整数にはならない。
#
# 非自明な零点の位置　最初の部分
#  14.134j + 1/2
#  21.022j + 1/2
#  25.010j + 1/2
#  30.424j + 1/2
#  32.935j + 1/2
#  
#  また、1+j0は極である。


"""
    例１：最初の非自明な零点を５個含む□領域の計算
    python test1.py
    tate yoko position [0.1, 33.0] [0.0, 1.0]
    sekibun step 0.01
    tate yoko bunkatu number  3290 100
    sekibun (4.99999386888865 + 0.00026736587192154j)
    
    例２：積分のステップを小さく(0.01から0.001)して 積分の計算精度を上げたもの　計算時間が掛かる
    python test1.py -d 0.001
    tate yoko position [0.1, 33.0] [0.0, 1.0]
    sekibun step 0.001
    tate yoko bunkatu number  32900 1000
    sekibun (4.9999999386888 + 2.67364609117596e-6j)
    
    例３：1+j0の極を含む□領域の計算 1個減って4になる  
    python test1.py -y 1.1 -a -0.1
    tate yoko position [-0.1, 33.0] [0.0, 1.1]
    sekibun step 0.01
    tate yoko bunkatu number  3310 110
    sekibun (4.00012873370994 + 5.0342983213897e-6j)
    
    例４： j100までの□領域の零点の計算　零点の数は29個となった  
    python test1.py -b 100
    tate yoko position [0.1, 100.0] [0.0, 1.0]
    sekibun step 0.01
    tate yoko bunkatu number  9990 100
    sekibun (28.9999988534685 + 0.000267366957181842j)
    
"""

import math
import mpmath
import numpy as np

# Python 3.6.4 on windows 10
# mpmath 1.1.0
# numpy  1.19.5


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='kubunn sekibun')
    parser.add_argument('--delta','-d', type=float, default=0.01, help='sekibun step')
    parser.add_argument('--x0','-x', type=float, default=0.0, help='yoko position')  # 臨界領域
    parser.add_argument('--x1','-y', type=float, default=1.0, help='yoko position')  # 臨界領域
    parser.add_argument('--y0','-a', type=float, default=0.1, help='tate position')  # 1+j0の極を避けるため0.1にした
    parser.add_argument('--y1','-b', type=float, default=33.0, help='tate position')
    args = parser.parse_args()
    
    # (1)□領域と分割ステップを指定する
    # □領域
    #w0=[0.0, 1.0]  # □の横幅　臨界領域の実数の幅
    #h0=[1.0, 33.0] # □の高さ　最初の非自明な零点を５個含む虚数の高さ
    w0=[ args.x0, args.x1 ]
    h0=[ args.y0, args.y1 ]
    
    
    print("tate yoko position",h0,w0)
    #　区分求積法の分割ステップ
    #delta0=0.01
    delta0= args.delta
    
    print("sekibun step", delta0)
    #
    hd=int((h0[1]-h0[0])/delta0)  # □の縦の分割数
    wd=int((w0[1]-w0[0])/delta0)  # □の横の分割数
    
    print("tate yoko bunkatu number ", hd, wd)
    
    #(2)以下は計算
    yp=np.arange(hd+1, dtype='float64') * delta0
    xp=np.arange(wd+1, dtype='float64') * delta0
    
    
    #　□の中で計算するポイント位置を求める
    c1=(w0[1]+ 1j * h0[0]) +  1j * yp
    c2= c1[-1] - xp
    c3= c2[-1] - 1j * yp
    c4= c3[-1] + xp
    
    c_all_plus_last=np.concatenate([c1[:-1],c2[:-1],c3[:-1],c4])  # ラストに始点も追加してある
    
    # 分割幅を求める
    b1= np.ones(hd) * delta0 * 1j
    b2= np.ones(wd) * delta0
    
    b_all_half=np.concatenate([b1,b2 * -1 ,b1 * -1 ,b2]) / 2.0  # 区分求積で2点の平均値を計算するためあらかじめ1/2を掛けてある
    
    # ポイントを表示する
    if 0:
        for i in range(len(c_all_plus_last)-1):
            print(i,c_all_plus_last[i],b_all_half[i] * 2.0)
        print(c_all_plus_last[-1])
    
    
    # 区分求積法で計算する
    rtn=mpmath.mpf(0.0)
    stack0=mpmath.fdiv(mpmath.zeta(c_all_plus_last[0], derivative=1),mpmath.zeta(c_all_plus_last[0]))
    for i in range(len(c_all_plus_last)-1):
        stack1=mpmath.fdiv(mpmath.zeta(c_all_plus_last[i+1], derivative=1),mpmath.zeta(c_all_plus_last[i+1]))
        rtn=mpmath.fadd(mpmath.fmul(mpmath.fadd(stack0,stack1), mpmath.mpmathify(b_all_half[i])),rtn)
        stack0=mpmath.mpmathify(stack1)
        
    sekibun= rtn / (2.0 * math.pi * 1j)
    
    # 結果を出力する
    print('sekibun', sekibun)
