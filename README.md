# Axell_AI_2024
SIGNATEで開催されたAxell AI Contest 2024(https://signate.jp/competitions/1374)の29位獲得解法（グループに参加において最高スコアを獲得した自身の提出コード）

# コンペ概要
このコンペは4倍超解像モデルの精度(PSNRで評価)を競うものだったが、推論時間の制限(Tesla T4で1枚あたり0.035sec以内)があった。

# 採用モデル
公式から配布されていた元モデル(ESPCN)を2層だけ深くし、特段の変化は加えていない。(モデルの構造についても吟味する余裕があれば、取り組みたかった。)

※ESPCNについて
ESPCNの最大の特徴は、アップサンプリングの際に従来の補間やデコンボリューションを用いず、サブピクセルシフティングという手法を使う点にある。通常のアップサンプリングでは、解像度を拡大してから処理を行うのに対し、ESPCNでは低解像度のまま処理を行い、最終的なステップで高解像度の画像に変換する。これによって、計算コストが低減され、高精度な超解像が可能となる。

詳しくは論文を参照(https://arxiv.org/abs/1609.05158)

# 解法概要
## 1.Data Augmentation
画像のDAについては、水平変換、垂直変換、回転(45度を限度)、明るさ調整、コントラスト調整をランダムで行うように採用。cutmixやcutout等も使用したが、特段精度に対しての影響はなかった。(もしかしたら実装を間違ったかも)

## 2.AWPの採用
他の超解像コンペの入賞解法を参考にしている際に、AWPを採用し過学習を防いで、精度を上げているものがあったので、今回の私の解法にも取り入れさせていただいた。
参照解法(https://hack.nikkei.com/blog/solafune202304/)

また実際にコードとして書く際には、以下を参考にさせていただいた。
(https://speakerdeck.com/masakiaota/kaggledeshi-yong-sarerudi-dui-xue-xi-fang-fa-awpnolun-wen-jie-shuo-toshi-zhuang-jie-shuo-adversarial-weight-perturbation-helps-robust-generalization)

## 3.訓練の方法について
epochは100（余裕があれば、よりepochを増やすことで精度向上の余地あり）
optimizerとしてAdamWを採用（learning rateは1e-3、weight decayは1e-3）
schedulerとしてCosineAnnealingを採用（100epochにかけて学習率が0まで減衰するようにスケジューリング）

学習についてはepochごとに内容を分けて行った。
学習計画は以下の通り

1~60epoch：DA
61~80epoch：DAなし
81~100epoch：DAなし+AWP

前半からAWPを入れてしまうと学習が十分に進まなかったり、すべての学習にたいしてDAの画像だと元画像に対する学習を行えなかったりするので、以上のように設定した。

※筆者はGoogle Colab上で動かしていたのだが、頻繁に学習が中断されてしまい、困っていたので、学習再開機能を実装した。（何ならこの学習中断のせいで中盤くらいまで無駄に時間を食われていた）

