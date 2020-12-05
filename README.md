# cookpad-lstm-classification
LSTMを用いてcookpadの料理名から料理のカテゴリを予測する。

# 使い方

1. 下のコマンドを打つ。
```bash
$:pipenv sync
$:cp .env.example .env
```
2. Cookpadデータベースと接続。.envの環境変数を書き換える。(cookpadのデータベースはcookpadに申請したらもらえます。)
3. mysql上のcookpadデータベースからデータをもってくる。
```
python cookpad_lstm_classification/get_data.py
```
4. 前処理をする。
```
python cookpad_lstm_classification/preprocessing.py
```
5. モデルを学習させる。
```
python cookpad_lstm_classification/train.py
```
