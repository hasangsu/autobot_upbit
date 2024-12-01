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
TRAILING_STOP_PERCENT = 0.02

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
    # 설명변수 추가
    data['ma15'] = calculate_ma(data, period=15)
    data['ma50'] = calculate_ma(data, period=50)
    data['rsi'] = calculate_rsi_series(data)
    data['volatility'] = (data['high'] - data['low']) / data['low']  # 변동성
    data['volume_change_rate'] = data['volume'].pct_change()  # 거래량 변화율
    data['upper_band'], data['lower_band'] = calculate_bollinger_bands_series(data)
    data['macd'], data['signal'] = calculate_macd(data)

    # 불필요 설명변수 제거
    data = data.dropna()
    data_prep = remove_outlier(data)    
    # boxplot_vis(data, "ssha")

    # 목표변수 할당
    target_return = (data_prep['close'].shift(-1) - data_prep['close']) / data_prep['close'] - FEE_RATE

    # 목표 변수 계산 (길이 일치)
    data_prep = data_prep.iloc[:-1]  # 마지막 행 제거 (shift(-1)로 NaN이 생긴 마지막 값 제외)
    target_return = target_return.iloc[:-1]  # 동일하게 길이 조정

    data_prep['target'] = np.where(target_return > 0.002, 1,   # 상승(수익 > 0.2%) -> 매수
                                   np.where(target_return < -0.002, -1, 0))

    # 결측치 제거
    data_prep.dropna(axis=0, how='any', inplace=True)

    x = data_prep[data_prep.columns.difference(['target'])]
    y = data_prep['target']
    return x, y

# 학습 및 모델 생성
def train_model(ticker):
    print(f"[notify] start learning {ticker} model")
    data = get_ohlcv_data(ticker, count=1000, period=1.5)
    x, y = create_training_data(data)
    if x is None or len(x) == 0:
        print(f"[error] {ticker} 특징 데이터 부족")
        return None, None

    # 표준 스케일러(평균 0, 분산 1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 123)
    y_train.value_counts(normalize=True)

    algorithm = RandomForestClassifier
    algorithm_name = 'rfc'

    # 기본 보델 학습
    modeling_uncustomized(algorithm, x_train, y_train, x_test, y_test)
    
    n_estimator = 30
    n_depth = 6
    n_split = 20
    n_leaf = 2

    model = model_final(ticker, algorithm, algorithm_name, x.columns,
            x_train, y_train, x_test, y_test,
            n_estimator, n_depth, n_split, n_leaf)

    return model, scaler

# 매매 신호 생성 및 실거래 실행
def generate_signals(model, scaler, ticker):
    data = get_ohlcv_data(ticker, count=100, period=0.1)
    features, _ = create_training_data(data)
    features = scaler.transform(features)

    predictions = model.predict(features)
    signal = predictions[-1]

    current_price = pyupbit.get_current_price(ticker)

    # 트레일링 스탑 로드
    buy_price, trail_stop_price = load_trailing_stop(ticker, ".")

    # 트레일링 스탑 갱신 (현재가 상승 시 상향 조정)
    if buy_price is not None and trail_stop_price is not None:
        new_trail_stop_price = current_price * (1 - TRAILING_STOP_PERCENT)
        if current_price > trail_stop_price:  # trail_stop_price 기준 상승 확인
            trail_stop_price = max(trail_stop_price, new_trail_stop_price)
            save_trailing_stop(ticker, buy_price, trail_stop_price, ".")
    
    trade_message = f"trade_bot({ticker}, {signal}) has been executed\n"
    if signal == 1:
        print(f"{ticker}: 매수 신호 발생 - 현재가 {current_price}")

        # 매수
        krw_balance = upbit.get_balance("KRW")
        if krw_balance > BUY_AMOUNT:
            if buy_price is None or trail_stop_price is None:
                buy_price = current_price
                trail_stop_price = current_price * (1 - TRAILING_STOP_PERCENT)
                save_trailing_stop(ticker, buy_price, trail_stop_price, ".")

            upbit.buy_market_order(ticker, BUY_AMOUNT)
            trade_message += create_notification("buy", "success", ticker, BUY_AMOUNT)

        else:
            trade_message += create_notification("buy", "fail", ticker, f"don't have the cash to buy") 

    elif signal == -1:
    # elif (recent_signals == -1).sum() >= 5:
        print(f"{ticker}: 매도 신호 발생 - 현재가 {current_price}")

        # 매도신호이면서, 트레일링 스탑 진행
        if buy_price is not None and trail_stop_price is not None and current_price < trail_stop_price:
            save_trailing_stop(ticker, None, None, ".")

            # 매도
            coin_balance = upbit.get_balance(ticker.split('-')[1])
            if coin_balance > 0:
                sell_amount = coin_balance * 0.25
                sell_value_in_krw = sell_amount * current_price

                if sell_value_in_krw < 5000:
                    upbit.sell_market_order(ticker, coin_balance)
                    trade_message += create_notification("sell", "success", ticker, f"remain {coin_balance}") 
                else:
                    upbit.sell_market_order(ticker, sell_amount)
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
    remove_old_model(ticker, r"rf", ".", 5)
    model, scaler = load_model(ticker, f"rf", f"pkl")
    if model is None or scaler is None:
        model, scaler = train_model(ticker)  # 모델 학습
        save_model(ticker, f"rf", f"pkl", model, scaler)
    
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
