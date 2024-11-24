import pyupbit
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
from autobot_apikey import *
from autobot_func import *

# coin & env
#COIN = "KRW-XRP"
COINS = ["KRW-BTC", "KRW-XRP", "KRW-ETH", "KRW-DOGE"]
INTERVAL = "minute15"
BUY_AMOUNT = 5000
FEE_RATE = 0.0005 * 2  # 수수료 비율 (한번의 거래 수수료 0.05% * 매수와 매도)

# upbit instance
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 미체결 주문 확인 및 취소하는 함수
def cancel_open_orders(coin):
    # 미체결 주문 조회
    open_orders = upbit.get_order(coin, state='wait')
    if open_orders:
        for order in open_orders:
            order_uuid = order['uuid']
            cancel_result = upbit.cancel_order(order_uuid)
            print(f"[cancle order] {cancel_result}")
    else:
        print(f"[cancle order] {coin} no order")

# OHLCV 데이터 가져오기
def get_ohlcv_data(ticker, interval=INTERVAL, count=200, period=0.1):
    return pyupbit.get_ohlcv(ticker, interval, count, period=period)

# 학습 데이터 생성
def create_training_data(data):
    data['ma15'] = calculate_ma(data, period=15)
    data['ma50'] = calculate_ma(data, period=50)
    data['rsi'] = calculate_rsi_series(data)
    data['volatility'] = (data['high'] - data['low']) / data['low']  # 변동성
    data['volume_change_rate'] = data['volume'].pct_change()  # 거래량 변화율
    data['upper_band'], data['lower_band'] = calculate_bollinger_bands_series(data)
    data['macd'], data['signal'] = calculate_macd(data)

    target_return = (data['close'].shift(-1) - data['close']) / data['close'] - FEE_RATE
    data['target'] = np.where(target_return > 0.002, 1,   # 상승(수익 > 0.2%) -> 매수
                              np.where(target_return < -0.002, -1, 0))  # 하락(손실 > 0.2%) -> 매도

    data = data.dropna()
    features = data[['ma15', 'ma50', 'rsi', 'volatility', 'volume_change_rate', 'upper_band', 'lower_band', 'macd', 'signal']].values
    labels = data['target'].values
    return features, labels

# 학습 및 모델 생성
def train_model(ticker):
    print(f"[notify] start learning {ticker} model")
    data = get_ohlcv_data(ticker, count=1000, period=1.5)
    features, labels = create_training_data(data)
    if features is None or len(features) == 0:
        print(f"[error] {ticker} 특징 데이터 부족")
        return None, None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 하이퍼파라미터 그리드 정의 (RandomForestClassifier에 적합한 하이퍼파라미터)
    param_grid = {
        'n_estimators': [200, 300, 500],  # 트리의 수
        'max_depth': [10, 20, 30, None],  # 트리의 최대 깊이
        'min_samples_split': [2, 5, 10],  # 분할을 위한 최소 샘플 수
        'min_samples_leaf': [1, 2, 4],    # 리프 노드의 최소 샘플 수
        'max_features': ['auto', 'sqrt', 'log2'],  # 각 트리에서 사용할 특성의 수
        'bootstrap': [True, False],  # 부트스트랩 샘플링 여부
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)

    # 모델 학습 및 하이퍼파라미터 튜닝
    grid_search.fit(features_scaled, labels)

    # 최적 하이퍼파라미터와 성능 출력
    print("최적 하이퍼파라미터: ", grid_search.best_params_)
    print("최고 성능: ", grid_search.best_score_)

    # 최적 모델 반환
    best_model = grid_search.best_estimator_

    save_model(ticker, f"rf", f"pkl", best_model, scaler)
    return best_model, scaler

# 매매 신호 생성 및 실거래 실행
def generate_signals(model, scaler, ticker):
    data = get_ohlcv_data(ticker, count=100, period=0.1)
    features, _ = create_training_data(data)
    features = scaler.transform(features)

    predictions = model.predict(features)
    signal = predictions[-1]
    recent_signals = predictions[-5:]

    current_price = pyupbit.get_current_price(ticker)
    
    trade_message = f"trade_bot({signal}) has been executed\n"
    if signal == 1:
        print(f"{ticker}: 매수 신호 발생 - 현재가 {current_price}")
        # 매수
        krw_balance = upbit.get_balance("KRW")
        if krw_balance > BUY_AMOUNT:
            buy_result = upbit.buy_market_order(ticker, BUY_AMOUNT)
            trade_message += create_notification("buy", "success", ticker, BUY_AMOUNT)
        else:
            trade_message += create_notification("buy", "fail", ticker, f"don't have the cash to buy") 

    # elif signal == -1:
    elif (recent_signals == -1).sum() >= 5:
        print(f"{ticker}: 매도 신호 발생 - 현재가 {current_price}")
        # 매도
        coin_balance = upbit.get_balance(ticker.split('-')[1])
        if coin_balance > 0:
            sell_amount = coin_balance * 0.25
            sell_value_in_krw = sell_amount * current_price
            if sell_value_in_krw < 5000:
                sell_result = upbit.sell_market_order(ticker, coin_balance)
                trade_message += create_notification("sell", "success", ticker, f"remain {coin_balance}") 
            else:
                sell_result = upbit.sell_market_order(ticker, sell_amount)
                trade_message += create_notification("sell", "success", ticker, f"remain {coin_balance}") 
        else:
            trade_message += create_notification("sell", "fail", ticker, f"don't have {ticker}") 
    else:
        # 홀드
        print(f"{ticker}: 홀드 신호 발생 - 현재가 {current_price}")
        trade_message += create_notification("hold", "hold", ticker, f"hold") 

    notify_slack(SLACK_HOOKS_URL, trade_message, "notify")

# 코인별 처리 함수
def process_ticker(ticker):
    model, scaler = load_model(ticker, f"rf", f"pkl")
    if model is None or scaler is None:
        model, scaler = train_model(ticker)  # 모델 학습
    
    try:
        generate_signals(model, scaler, ticker)
    except Exception as e:
        print(f"[error] 신호 생성 중 오류 발생 ({ticker}): {e}")

# 메인 함수
def main():
    for COIN in COINS:
        cancel_open_orders(COIN)
        process_ticker(COIN)

if __name__ == "__main__":
    main()
