#coding:utf-8

###########################################################
# Levinson-DurbinのアルゴリズムにてLPC係数を求める
#
# autocorr and LevinsonDurbin is baed on
# <http://aidiary.hatenablog.com/entry/20120415/1334458954>
#
###########################################################
#
# 追加　係数k (kc)
###########################################################
import numpy as np

#Check version
# Python 3.6.4, 64bit on Win32 (Windows 10)
# numpy (1.14.0)


def autocorr(x, nlags=None):
    """
    自己相関関数を求める
    x:     信号
    nlags: 自己相関関数のサイズ（lag=0からnlags-1まで）
           引数がなければ（lag=0からlen(x)-1まですべて）
    """
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r


def LevinsonDurbin(r, lpcOrder):
    """
    Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める
    """
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)
    kc = np.zeros(lpcOrder ) # 追加

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]
    kc[0] = lam # 追加

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]
        kc[k] = lam # 追加

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1], kc


#   入力：信号
#         LPCの次数
#
#   出力：LPC係数
#         差

def lpc(s, lpcOrder=32):
    r = autocorr(s, lpcOrder + 1)
    a, e, k  = LevinsonDurbin(r, lpcOrder)
    return a,e


# LPC係数を求める
#
#   入力：信号
#         LPCの次数
#
#   出力：LPC係数
#         差
#         係数k # 追加
def lpc_k(s, lpcOrder=32):  # 追加
    r = autocorr(s, lpcOrder + 1)
    a, e, k  = LevinsonDurbin(r, lpcOrder)
    return a,e,k 


# LPC予測残差を計算する
#
#   入力：LPC係数
#         信号
#
#   出力：LPC予測残差

def residual_error(a, s):
    lpcOrder=len(a)
    r_error=s.copy()
    
    for i in range(lpcOrder, len(s)):
        for j in range (0,lpcOrder):
            r_error[i] += (a[j] * s[i-j-1])
    r_error[0:lpcOrder-1]=0.0
    
    return r_error

# 面積を求める　
# 時間ステップが１サンプリング時間に相当するフィルターの係数を面積と考えた場合で、実際の声道の長さと違うのかな？
#
#   入力：係数k 
#
#   出力：面積
#        
def get_A(k, A0=1.0):  # 追加
    A = np.zeros(len(k)+1)
    A[0]=A0 # set initial value
    for l in range(len(k)):  # 出口から順番に求める
        if abs(k[l]) >= 1.0:
            print ('error: abs(k) > 1.0',l)
        A[l+1]= ((1.0 - k[l]) / ( 1 + k[l])) * A[l]
    return A[::-1]  # 入口からの順番に変換する


