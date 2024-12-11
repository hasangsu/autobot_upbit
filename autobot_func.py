import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import math
import requests
import os
import json
import time
from joblib import dump, load
from datetime import datetime, timedelta

# 모델과 스케일러 저장 및 로드
def save_model(ticker, type, extension, model, scaler, save_path = "."):
    # 모델과 스케일러 파일 경로 생성
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{ticker}_{type}_model.{extension}")
    scaler_path = os.path.join(save_path, f"{ticker}_{type}_scaler.{extension}")

    dump(model, model_path)
    dump(scaler, scaler_path)

    print(f"[notify] save model {ticker}")

def load_model(ticker, type, extension, load_path="."):
    # 모델과 스케일러 파일 경로 생성
    model_path = os.path.join(load_path, f"{ticker}_{type}_model.{extension}")
    scaler_path = os.path.join(load_path, f"{ticker}_{type}_scaler.{extension}")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"[notify] success load {ticker} model")
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    else:
        print(f"[notify] error load {ticker} model -> new model")
        return None, None
    
# 특정기간보다 오래된 모델 파일 제거
def remove_old_model(ticker, type, folder_path, days_threshold=7):
    now = time.time()
    cutoff = now - days_threshold * 86400  # 7일을 초로 변환

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # save_model에서 생성한 파일만 삭제 (패턴 기반 필터링)
        if (
            os.path.isfile(file_path)
            and filename.startswith(f"{ticker}_{type}_")
        ):
            file_mtime = os.path.getmtime(file_path)  # 파일의 마지막 수정 시간 가져오기
            if file_mtime < cutoff:
                os.remove(file_path)
                print(f"[notify] Removed old model file: {file_path}")

# 트레일링 스탑 저장
def save_trailing_stop(ticker, buy_price, trail_stop_price, save_path="."):
    # 트레일링 스탑 파일 경로 생성
    os.makedirs(save_path, exist_ok=True)
    trailing_stop_path = os.path.join(save_path, f"{ticker}_trailingstop.json")
    
    # 상태 저장
    trail_stop_data = {
        'buy_price': buy_price,
        'trail_stop_price': trail_stop_price
    }
    
    # JSON 파일로 저장
    with open(trailing_stop_path, 'w') as f:
        json.dump(trail_stop_data, f)

    print(f"[notify] save trailing stop {ticker}")

# 트레일링 스탑 로드
def load_trailing_stop(ticker, load_path="."):
    # 트레일링 스탑 파일 경로 생성
    trailing_stop_path = os.path.join(load_path, f"{ticker}_trailingstop.json")
    
    if os.path.exists(trailing_stop_path):
        with open(trailing_stop_path, 'r') as f:
            trail_stop_data = json.load(f)

        print(f"{ticker} trailing load: {trailing_stop_path}")
        return trail_stop_data['buy_price'], trail_stop_data['trail_stop_price']
    else:
        print(f"{ticker} fail trailing load: {trailing_stop_path}")
        return None, None

# rsi series
def calculate_rsi_series(data, period=14, target=-1):
    delta = data['close'].diff(1).dropna()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    '''
    # sma
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    '''
    
    # ema
    _gain = pd.Series(gain).ewm(com=(period - 1), min_periods=period).mean()
    _loss = pd.Series(loss).ewm(com=(period - 1), min_periods=period).mean()

    rs = _gain / _loss
    rsi = 100 - (100 / (1 + rs))
    return float(pd.Series(rsi, name="RSI").iloc[target])

# rsi
def calculate_rsi(data, period=14, target=-1):
    delta = data['close'].diff(1).dropna()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    '''
    # sma
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    '''
    
    # ema
    _gain = pd.Series(gain).ewm(com=(period - 1), min_periods=period).mean()
    _loss = pd.Series(loss).ewm(com=(period - 1), min_periods=period).mean()

    rs = _gain / _loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    # return float(pd.Series(rsi, name="RSI").iloc[target])

# ma
def calculate_ma(data, period=20, target=-1):
    return float(data['close'].rolling(window=period).mean().iloc[target])

# macd
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data['close'].ewm(span=short_period).mean()
    ema_long = data['close'].ewm(span=long_period).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period).mean()
    return macd, signal

# bollinger_bands
def calculate_bollinger_bands(data, window=20, std_dev=2, target=-1):
    ma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    return float(upper_band.iloc[target]), float(lower_band.iloc[target]), float(data['close'].iloc[target])
    

# bollinger_bands
def calculate_bollinger_bands_series(data, window=20, std_dev=2):
    ma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    return upper_band, lower_band
    
# buy conis
def should_buy(trade_data):
    return (
        (float(trade_data["rsi"].iloc[-1]) < 30 and float(trade_data["macd"].iloc[-1]) > float(trade_data["signal"].iloc[-1]) and trade_data["close_price"] <= trade_data["lower_band"] and trade_data["current_price"] < trade_data["ma"])
                or (trade_data["close_price"] <= trade_data["lower_band"] and trade_data["close_price"] > trade_data["lower_band"] * 1.02)  # 볼린저밴드 하단 반등
                or (trade_data["close_price"] <= trade_data["ma_100"] and trade_data["close_price"] > trade_data["ma_100"] * 0.98)  # 주요 지지선 근처에서 반등
                or (trade_data["close_price"] >= trade_data["recent_close_prices"] * 1.05)  # 최근 하락폭 대비 가격 회복
    )

# sell conis
def should_sell(trade_data):
    return (
        (float(trade_data["rsi"].iloc[-1]) > 70 and float(trade_data["rsi"].iloc[-3]) > float(trade_data["rsi"].iloc[-2]) > float(trade_data["rsi"].iloc[-1]))  # RSI가 연속 하락
                or (float(trade_data["macd"].iloc[-1]) < float(trade_data["signal"].iloc[-1]) and float(trade_data["macd"].iloc[-2]) > float(trade_data["signal"].iloc[-2]))  # MACD와 Signal 하향 교차
                or (trade_data["close_price"] >= trade_data["upper_band"] and trade_data["close_price"] < trade_data["upper_band"] * 0.98)  # 볼린저 밴드 상단 돌파 후 2% 하락
                or (trade_data["current_price"] < trade_data["recent_close_prices"] * 0.95)  # 최근 5개 캔들 평균보다 5% 하락
                or (trade_data["data"]['volume'].iloc[-1] < trade_data["recent_close_prices"] * 0.8)  # 거래량 감소:
    )

# slack notify
def notify_slack(url, msg, title):
    try:
        # 메시지 전송
        requests.post(
            url,
            headers={
                'content-type': 'application/json'
            },
            json={
                'text': title,
                'blocks': [
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': msg
                        }
                    }
                ]
            }
        )
    except Exception as ex:
        print(ex)

def create_notification(trade, status, ticker, details):
    return f"[notify] {trade}, {status}, {ticker}, {details}"

# 이상치 확인
def boxplot_vis(data, target_name):
    # 현재 스크립트 실행 경로에 figure 디렉토리 생성
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'figure')
    os.makedirs(save_dir, exist_ok=True)
    
    num_features = len(data.columns)  # 데이터의 열 개수
    ncols = 2  # 한 행에 서브플롯 2개
    nrows = math.ceil(num_features / ncols)  # 필요한 행 수 계산

    plt.figure(figsize=(15, 5 * nrows))  # 그래프 크기 동적 설정
    for col_idx in range(num_features):
        plt.subplot(nrows, ncols, col_idx + 1)  # 유효한 서브플롯 범위 설정
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops=dict(markerfacecolor='r', marker='D'))
        plt.title("Feature" + "(" + target_name + "): " + data.columns[col_idx], fontsize=10)

    # 그래프 저장
    plt.savefig(os.path.join(save_dir, 'boxplot_' + target_name + '.png'))
    plt.show()

# 이상치 제거
def remove_outlier(data):
    # 각 열에 대해 이상치를 제거하도록 수정
    for column in data.columns:
        q1 = data[column].quantile(0.25)  # 제 1사분위수
        q3 = data[column].quantile(0.75)  # 제 3사분위수
        iqr = q3 - q1  # IQR(Interquartile range) 계산
        minimum = q1 - (iqr * 1.5)  # IQR 최솟값
        maximum = q3 + (iqr * 1.5)  # IQR 최댓값
        data = data[(data[column] >= minimum) & (data[column] <= maximum)]  # 해당 열에서 이상치 제거
    return data

# 히스토그램
def histogram_vis(data):
    # 현재 스크립트 실행 경로에 figure 디렉토리 생성
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'figure')
    os.makedirs(save_dir, exist_ok=True)
    
    # 설명변수 선정
    x = data[data.columns.difference(['target'])]

    # 설명변수명 리스트
    feature_name = x.columns

    num_features = len(feature_name)  # 데이터의 열 개수
    ncols = 2  # 한 행에 서브플롯 2개
    nrows = math.ceil(num_features / ncols)  # 필요한 행 수 계산

    plt.figure(figsize=(30, 30))

    for col_idx in range(num_features):
        plt.subplot(nrows, ncols, col_idx + 1)  # 유효한 서브플롯 범위 설정
        feature = feature_name[col_idx]

        # data histogram 시각화
        plt.hist(data[data["target"] == 0][feature], alpha = 0.5)
        plt.legend()

        # 그래프 타이틀: feature name
        plt.title(f"Feature: {feature}", fontsize=20)
        
    # 그래프 저장
    plt.savefig(os.path.join(save_dir, 'relationshi.png'))
    plt.show()

def modeling_uncustomized(algorithm, x_train, y_train, x_test, y_test):
    # 하이퍼파라미터 조정 없이 모델 학습
    uncustomized = algorithm(random_state=1234)
    uncustomized.fit(x_train, y_train)

    # Train Data 설명력
    train_score_before = uncustomized.score(x_train, y_train)
    print(f"학습 데이터셋 정확도: {train_score_before}")

    # Test Data 설명력
    test_score_before = uncustomized.score(x_test, y_test)
    print(f"테스트 데이터셋 정확도: {test_score_before}")
    
    return train_score_before, test_score_before

def optimi_visualization(algorithm_name, x_values, train_score, test_score, xlabel, filename):
    # 현재 스크립트 실행 경로에 figure 디렉토리 생성
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'figure')
    os.makedirs(save_dir, exist_ok=True)

    # 하이퍼파라미터 조정에 따른 학습 데이터셋 기반 모델 성능 추이 시각화
    plt.plot(x_values, train_score, linestyle = '-', label = 'train score')

    # 하이퍼파라미터 조정에 따른 테스트 데이터셋 기반 모델 성능 추이 시각화
    plt.plot(x_values, test_score, linestyle = '--', label = 'test score')
    plt.ylabel('Accuracy(%)') # y축 라벨
    plt.xlabel(xlabel) # x축 라벨
    plt.legend() # 범례표시
    plt.savefig(os.path.join(save_dir, f"{algorithm_name}_{filename}.png"))

# 학습할 트리 모델 개수 선정
def optimi_estimator(algorithm, algorithm_name, x_train, y_train, x_test, y_test, n_estimator_min, n_estimator_max):
    train_score = []
    test_score =[]
    para_n_tree = [n_tree*5 for n_tree in range(n_estimator_min, n_estimator_max)]

    for v_n_estimators in para_n_tree:
        model = algorithm(n_estimators = v_n_estimators, random_state=1234)
        model.fit(x_train, y_train)
        
        train_accuracy = model.score(x_train, y_train)  # 훈련 세트 정확도
        test_accuracy = model.score(x_test, y_test)    # 테스트 세트 정확도

        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    # 트리 개수에 따른 모델 성능 저장
    df_score_n = pd.DataFrame({
        'n_estimators': para_n_tree, 
        'TrainScore': train_score, 
        'TestScore': test_score,
        'diff': [train - test for train, test in zip(train_score, test_score)]  # 학습-테스트 정확도 차이
        })
    
    # diff가 일정 범위 이내로 작은 값들만 필터링 (여기서는 0.51 이하로 필터링)
    filtered_df = df_score_n[df_score_n['diff'] <= 0.51]

    # filtered_df가 빈 데이터프레임인 경우
    if filtered_df.empty:
        # 빈 데이터프레임일 경우 TestScore가 가장 높은 값을 선택
        optimal_row = df_score_n.loc[df_score_n['TestScore'].idxmax()]
    else:
        # accuracy_diff가 작은 값들 중에서 TestScore가 가장 높은 값의 n_estimators 선택
        optimal_row = filtered_df.loc[filtered_df['TestScore'].idxmax()]

    # 최적의 n_estimators 선택 (가능하면 트리 개수가 많고 정확도 차이가 작은 모델을 선택)
    # 여기서 추가적으로 트리 개수가 많은 순으로 우선 고려
    optimal_n_estimators = int(optimal_row['n_estimators'])

    # 트리 개수에 따른 모델 성능 추이 시각화 함수 호출
    optimi_visualization(algorithm_name, para_n_tree, train_score, test_score, "The number of estimator", "n_estimator")

    print(round(df_score_n, 4))
    print(f"Optimal n_estimators: {optimal_n_estimators}")
    
    return optimal_n_estimators

# 최대 깊이 선정
def optimi_maxdepth (algorithm, algorithm_name, x_train, y_train, x_test, y_test, depth_min, depth_max, n_estimator):
    train_score = []
    test_score = []
    para_depth = [depth for depth in range(depth_min, depth_max)]

    for v_max_depth in para_depth:
        # 의사결정나무 모델의 경우 트리 개수를 따로 설정하지 않기 때문에 RFC, GBC와 분리하여 모델링
        if algorithm == RandomForestClassifier:
            model = algorithm(max_depth = v_max_depth,
                              random_state=1234)
        else:
            model = algorithm(max_depth = v_max_depth,
                              n_estimators = n_estimator,
                              random_state=1234)
        
        model.fit(x_train, y_train)

        train_accuracy = model.score(x_train, y_train)  # 훈련 세트 정확도
        test_accuracy = model.score(x_test, y_test)    # 테스트 세트 정확도

        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    # 최대 깊이에 따른 모델 성능 저장
    df_score_n = pd.DataFrame({
        'depth': para_depth, 
        'TrainScore': train_score, 
        'TestScore': test_score,
        'diff': [train - test for train, test in zip(train_score, test_score)]  # 학습-테스트 정확도 차이
        })
    
    # diff가 0에 가장 가까운 값을 찾아 선택
    optimal_row = df_score_n.loc[df_score_n['diff'].abs().idxmin()]

    # 최적의 max_depth 반환
    optimal_max_depth = int(optimal_row['depth'])

    # 최대 깊이에 따른 모델 성능 추이 시각화 함수 호출
    optimi_visualization(algorithm_name, para_depth, train_score, test_score, "The number of depth", "n_depth")

    print(round(df_score_n, 4))
    print(f"Optimal max_depth: {optimal_max_depth}")

    return optimal_max_depth

# 분리 노드의 최소 자료 수 선정
def optimi_minsplit (algorithm, algorithm_name, x_train, y_train, x_test, y_test, n_split_min, n_split_max, n_estimator, n_depth):
    train_score = []
    test_score = []
    para_split = [n_split*2 for n_split in range(n_split_min, n_split_max)]

    for v_min_samples_split in para_split:
        # 의사결정나무 모델의 경우 트리 개수를 따로 설정하지 않기 때문에 RFC, GBC와 분리하여 모델링
        if algorithm == RandomForestClassifier:
            model = algorithm(min_samples_split = v_min_samples_split,
                              max_depth = n_depth,
                              random_state = 1234)
        else:
            model = algorithm(min_samples_split = v_min_samples_split,
                              n_estimators = n_estimator,
                              max_depth = n_depth,
                              random_state = 1234)
        model.fit(x_train, y_train)

        train_accuracy = model.score(x_train, y_train)  # 훈련 세트 정확도
        test_accuracy = model.score(x_test, y_test)    # 테스트 세트 정확도

        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    # 분리 노드의 최소 자료 수에 따른 모델 성능 저장
    df_score_n = pd.DataFrame({
        'min_samples_split': para_split, 
        'TrainScore': train_score, 
        'TestScore': test_score, 
        'diff': [train - test for train, test in zip(train_score, test_score)]  # 학습-테스트 정확도 차이
        })
    
    # diff가 가장 작은 값이 아닌, diff가 작은 값들 중에서 TestScore가 가장 높은 값을 선택
    min_diff = df_score_n['diff'].min()  # 가장 작은 diff 찾기
    filtered_df = df_score_n[df_score_n['diff'] == min_diff]  # 가장 작은 diff 값만 남기기

    # 가장 작은 diff 값들 중에서 TestScore가 가장 높은 값을 선택
    optimal_row = filtered_df.loc[filtered_df['TestScore'].idxmax()]

    # 최적의 min_samples_split 반환
    optimal_min_samples_split = int(optimal_row['min_samples_split'])

    # 분리 노드의 최소 자료 수에 따른 모델 성능 추이 시각화 함수 호출
    optimi_visualization(algorithm_name, para_split, train_score, test_score, "The minimum number of samples required to split an internal node", "min_samples_split")

    print(round(df_score_n, 4))
    print(f"Optimal min_samples_split: {optimal_min_samples_split}")
    
    return optimal_min_samples_split

# 잎사귀 노드의 최소 자료 수 선정
def optimi_minleaf(algorithm, algorithm_name, x_train, y_train, x_test, y_test, n_leaf_min, n_leaf_max, n_estimator, n_depth, n_split):
    train_score = []
    test_score = []
    para_leaf = [n_leaf*2 for n_leaf in range(n_leaf_min, n_leaf_max)]

    for v_min_samples_leaf in para_leaf:
        # 의사결정나무 모델의 경우 트리 개수를 따로 설정하지 않기 때문에 RFC, GBC와 분리하여 모델링
        if algorithm == RandomForestClassifier:
            model = algorithm(min_samples_leaf = v_min_samples_leaf,
                                        max_depth = n_depth,
                                        min_samples_split = n_split,
                                        random_state=1234)
        else:
            model = algorithm(min_samples_leaf = v_min_samples_leaf,
                                n_estimators = n_estimator,
                                max_depth = n_depth,
                                min_samples_split = n_split,
                                random_state=1234)
            
        model.fit(x_train, y_train)

        train_accuracy = model.score(x_train, y_train)  # 훈련 세트 정확도
        test_accuracy = model.score(x_test, y_test)    # 테스트 세트 정확도

        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    # 잎사귀 노드의 최소 자료 수에 따른 모델 성능 저장
    df_score_n = pd.DataFrame({
        'min_samples_leaf': para_leaf, 
        'TrainScore': train_score, 
        'TestScore': test_score,
        'diff': [train - test for train, test in zip(train_score, test_score)]  # 학습-테스트 정확도 차이
        })
    
    # diff가 일정 범위 이내로 작은 값들만 필터링 (여기서는 0.5 이하로 필터링)
    filtered_df = df_score_n[df_score_n['diff'] <= 0.5]

    # filtered_df가 빈 데이터프레임인 경우
    if filtered_df.empty:
        # 빈 데이터프레임일 경우 TestScore가 가장 높은 값을 선택
        optimal_row = df_score_n.loc[df_score_n['TestScore'].idxmax()]
    else:
        # diff가 작은 값들 중에서 TestScore가 가장 높은 값의 min_samples_leaf 선택
        optimal_row = filtered_df.loc[filtered_df['TestScore'].idxmax()]

    # 최적의 min_samples_leaf 반환
    optimal_min_samples_leaf = int(optimal_row['min_samples_leaf'])

    # 잎사귀 노드의 최소 자료 수에 따른 모델 성능 추이 시각화 함수 호출
    optimi_visualization(algorithm_name, para_leaf, train_score, test_score, "The minimum number of samples required to be at a leaf node", "min_samples_leaf")

    print(round(df_score_n, 4))
    print(f"Optimal min_samples_leaf: {optimal_min_samples_leaf}")

    return optimal_min_samples_leaf

# 최종 모델 학습
def model_final(ticker, algorithm, algorithm_name, feature_name, x_train, y_train, x_test, y_test, n_estimator, n_depth, n_split, n_leaf):
    # 현재 스크립트 실행 경로에 figure 디렉토리 생성
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'figure')
    os.makedirs(save_dir, exist_ok=True)

    # 의사결정나무 모델의 경우 트리 개수를 따로 설정하지 않기 때문에 RFC, GBC와 분리하여 모델링
    if algorithm == RandomForestClassifier:
        model = algorithm(random_state=1234, 
                          min_samples_leaf = n_leaf,
                          min_samples_split = n_split, 
                          max_depth = n_depth)
    else:
        model = algorithm(random_state = 1234, 
                          n_estimators = n_estimator, 
                          min_samples_leaf = n_leaf,
                          min_samples_split = n_split, 
                          max_depth = n_depth)
    # 모델 학습
    model.fit(x_train, y_train)
    
    # 최종 모델의 성능 평가
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    
    # 정확도, 정밀도, 재현율, F1 점수 계산 시 average 파라미터 수정
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")  # 정확도
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.3f}")  # 정밀도 (다중 클래스에서는 macro, micro, weighted 중 선택)
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.3f}")  # 재현율
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.3f}")  # F1 스코어
    
    # 혼동행렬 시각화
    plt.figure(figsize =(30, 30))
    disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    disp.plot()
    
    # 변수 중요도 산출
    dt_importance = pd.DataFrame()
    dt_importance['Feature'] = feature_name # 설명변수 이름
    dt_importance['Importance'] = model.feature_importances_ # 설명변수 중요도 산출

    # 변수 중요도 내림차순 정렬
    dt_importance.sort_values("Importance", ascending = False, inplace = True)
    print(dt_importance.round(3))

    # 변수 중요도 오름차순 정렬
    dt_importance.sort_values("Importance", ascending = True, inplace = True)

    # 변수 중요도 시각화
    coordinates = range(len(dt_importance)) # 설명변수 개수만큼 bar 시각화

    plt.barh(y = coordinates, width = dt_importance["Importance"])
    plt.yticks(coordinates, dt_importance["Feature"]) # y축 눈금별 설명변수 이름 기입
    plt.xlabel("Feature Importance") # x축 이름
    plt.ylabel("Features") # y축 이름
    plt.savefig(os.path.join(save_dir, '_feature_importance.png'))

    return model