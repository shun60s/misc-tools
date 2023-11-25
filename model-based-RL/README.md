# Model Based RL

車をパーキングに駐車させる(Parking)、簡単なモデルベースの強化学習の事例。 
[HighwayEnv Scripts](https://github.com/Farama-Foundation/HighwayEnv/tree/master/scripts)の中のscriptをgoogle colabで動かしている。   

## Parking envの説明  

Parkingの構成はドキュメント[highway-env Documentation](https://highway-env.farama.org/environments/parking/)に説明がある。   


### 状態 spaces of observation  

位置x, 位置y, 速度vx, 速度vy, 方向cos_h, 方向sin_h　の6個のパラメータ    
![](docs/observations.png)  


### 行動 actions  

アクセルとハンドルの２個パラメーター  
 ![](docs/actions.png)   
    
### リワードreward  

ゴール地点の状態との差を使って評価している  
![](docs/rewards.png)


## 力学的モデル dynamics model  

![](docs/dynamic_model.png)  

A = self.A2(F.relu(self.A1(xu)))  reluを使ってA1の出力は０以上の数にしている。  
B = self.B2(F.relu(self.B1(xu)))  
policy_frequency 5  
 
## 計画 planner 

![](docs/planner.png)  

  
  
state = state.expand(population, -1) 現在のstate を100個　複製して、  
NORMAL分布のアクションを生成して、predict_trajectoryでaction_repeat回後のstateを予想する。  
  
REWARDの上位10個からの新しい平均値と分散を計算する。  
best_actions = actions[:, best, :]  
action_mean = best_actions.mean(dim=1, keepdim=True)  
action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)  
  
上記をiterations回す。  
  
最後の平均値をactionとして、次のstateに進む。  
  
結果は上手く行っていない。各パラメータを変更してみても、上手くならい。  
理由は不明であるが、このplannerは上手く動かない。  
  
  
## 問題点 limitations  
  

・力学的モデルを完全に再現できる訳ではない。場合によってはバイアス(偏見)をかけてしまうこともある。  
・今回の方式では次のステップを決めるために多くの試行をしなくてはならず、次のアクションを直接計算するmodel freeに比べて計算量が多くなる。  
・RLの行動推定をモデル推定に置き換えているが、リワードrewardの評価関数と必ずしも相性がよいは限らない。  
  
  
## ライセンス  
  
オリジナルコードのライセンスはLICENSE-HighwayEnv.txtを参照のこと。  





