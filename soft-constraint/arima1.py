#coding:utf-8

"""
This code refers to https://www.salesanalytics.co.jp/datascience/datascience087/.
振り子の角度のARIMAモデルを求める。

# 引数　--n_pendulum 1(signle pendulum) または 2(double pendulum)を追加
# usage:  python arima1.py
"""

import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import utils
from pmdarima import arima
from pmdarima import model_selection
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import pyplot as plt


"""
パラメータの説明：
非季節性パラメータ（p,d,q）は、ARIMAモデルのパラメータと同じです。

p：ARIMA の AR componentの次数（自己回帰パラメータ）と同じ
d：ARIMA の I componentの次数（差分の階数）と同じ
q：ARIMA の MA componentの次数（移動平均パラメータ）と同じ

m (Seasonal Period)：季節性の周期
P (Seasonal AR component)：季節性の AR componentの次数
D (Seasonal I component)：季節性の I componentの次数
Q (Seasonal MA Component)：季節性のMA componentの次数
"""

if __name__ == '__main__':
    parser = ArgumentParser(description='arima1')
    parser.add_argument('--n_pendulum', type=int, default=1, help='Single-pendulum 1, Double-pendulum 2' )
    args = parser.parse_args()
    n_pendulum= args.n_pendulum
    
    # データセットの読み込み
    if n_pendulum == 1:
        FileInputCSV = 'sampling1_t_y.csv'
        number_of_used_data = 500
        title1='signle pendulum'
    elif n_pendulum == 2:
        FileInputCSV = 'sampling2_t_y.csv'
        number_of_used_data = 1000
        title1='double pendulum'
    
    df=pd.read_csv( FileInputCSV,                         #読み込むデータ
                    index_col='time',           #変数「time」をインデックスに設定
                   parse_dates=True)            
    print (df.head()) #確認
    df=df.iloc[0:number_of_used_data] # 先頭のnumber_of_used_data個のデータを使う。
    print (df)
    
    # グラフのスタイルとサイズ
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [12, 9]
    # プロット
    df.plot()
    plt.title( title1)              #グラフタイトル
    plt.ylabel('angular, velocity') #タテ軸のラベル
    plt.xlabel('time')              #ヨコ軸のラベル
    plt.show()
    
    # 学習データとテストデータに分割
    if n_pendulum == 1:
        df_train, df_test = model_selection.train_test_split(df.drop('velocity', axis=1), test_size=50) # use only angular
    elif n_pendulum == 2:
        df_train, df_test = model_selection.train_test_split(df.drop(['angular0','velocity0','velocity'], axis=1), test_size=50) # use only angular

    
    ####################################
    # モデル構築（Auto ARIMA）
    # 学習
    arima_model = pm.auto_arima(df_train, 
                            seasonal=False,
                            trace=True,
                            n_jobs=-1,
                            maxiter=10)
    # 予測
    ##学習データの期間の予測値
    train_pred = arima_model.predict_in_sample()
    ##テストデータの期間の予測値
    test_pred, test_pred_ci = arima_model.predict(
        n_periods=df_test.shape[0], 
        return_conf_int=True
    )
    # テストデータで精度検証
    print('RMSE:')
    print(np.sqrt(mean_squared_error(df_test, test_pred)))
    print('MAE:')
    print(mean_absolute_error(df_test, test_pred)) 
    print('MAPE:')
    print(mean_absolute_percentage_error(df_test, test_pred))
    
    # グラフ化
    fig, ax = plt.subplots()
    ax.plot(df_train.index, df_train.angular, label="actual(train dataset)")
    ax.plot(df_test.index, df_test.angular, label="actual(test dataset)", color="gray")
    
    ax.plot(df_train.index, train_pred, color="c")
    ax.plot(df_test.index, test_pred, label="SARIMA", color="c") 
    
    ax.legend()
    plt.show()