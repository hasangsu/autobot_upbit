import pyupbit
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from autobot_apikey import *
from autobot_func import *

# coin & env
#COIN = "KRW-XRP"
COINS = ["KRW-BTC", "KRW-XRP", "KRW-ETH", "KRW-DOGE"]
INTERVAL = "minute15"
BUY_AMOUNT = 5000

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
    # data['upper_band'], data['lower_band'] = calculate_bollinger_bands_series(data)

    data['target'] = np.where(data['close'].shift(-1) > data['close'], 1,   # 상승 -> 매수
                              np.where(data['close'].shift(-1) < data['close'], -1, 0))  # 하락 -> 매도, 나머지 -> 홀드
    
    data = data.dropna()
    features = data[['ma15', 'ma50', 'rsi']].values
    labels = data['target'].values
    return features, labels

# 학습 및 모델 생성
def train_or_update_model(ticker, model=None, scaler=None):
    print(f"[notify] start learning/updating {ticker} model")
    data = get_ohlcv_data(ticker, count=1000, period=1.5)
    features, labels = create_training_data(data)
    if features is None or len(features) == 0:
        print(f"[error] {ticker} 특징 데이터 부족")
        return None, None

    # 기존 모델 및 스케일러 로드
    if model is None or scaler is None:
        print(f"[notify] 모델이나 스케일러가 없어서 새로 학습합니다. {ticker}")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)  # scaler에 데이터를 학습시킨 후 변환
        model = SGDClassifier(max_iter=1000, random_state=42)
        model.fit(features, labels)
        print(f"[success] {ticker} 모델 학습 완료")
    else:
        print(f"[notify] 기존 모델을 업데이트합니다. {ticker}")
        features = scaler.transform(features)  # 이미 학습된 scaler를 사용하여 변환
        model.partial_fit(features, labels)  # SGDClassifier의 incremental 학습 방식
        print(f"[success] {ticker} 모델 업데이트 완료")

    # 모델 및 스케일러 저장
    save_model(ticker, f"sgd", model, scaler)
    return model, scaler

# 매매 신호 생성 및 실거래 실행
def generate_signals(model, scaler, ticker):
    data = get_ohlcv_data(ticker, count=200, period=0.1)
    features, _ = create_training_data(data)
    features = scaler.transform(features)

    predictions = model.predict(features)
    signal = predictions[-1]
    current_price = pyupbit.get_current_price(ticker)
    
    trade_message = f"trade_bot({signal}) has been executed\n"
    if signal == 1:
        print(f"{ticker}: 매수 신호 발생 - 현재가 {current_price}")
        # # 매수
        # krw_balance = upbit.get_balance("KRW")
        # if krw_balance > BUY_AMOUNT:
        #     buy_result = upbit.buy_market_order(ticker, BUY_AMOUNT)
        #     trade_message += create_notification("buy", "success", ticker, BUY_AMOUNT)
        # else:
        #     trade_message += create_notification("buy", "fail", ticker, f"don't have the cash to buy") 

    elif signal == -1:
        print(f"{ticker}: 매도 신호 발생 - 현재가 {current_price}")
        # 매도
        # coin_balance = upbit.get_balance(ticker.split('-')[1])
        # if coin_balance > 0:
        #     sell_amount = coin_balance * 0.5
        #     sell_value_in_krw = sell_amount * current_price
        #     if sell_value_in_krw < 5000:
        #         sell_result = upbit.sell_market_order(ticker, coin_balance)
        #         trade_message += create_notification("sell", "success", ticker, f"remain {coin_balance}") 
        #     else:
        #         sell_result = upbit.sell_market_order(ticker, sell_amount)
        #         trade_message += create_notification("sell", "success", ticker, f"remain {coin_balance}") 
        # else:
        #     trade_message += create_notification("sell", "fail", ticker, f"don't have {ticker}") 
    else:
        # 홀드
        trade_message += create_notification("hold", "hold", ticker, f"hold") 

    # notify_slack(SLACK_HOOKS_URL, trade_message, "notify")

# 코인별 처리 함수
def process_ticker(ticker):
    model, scaler = load_model(ticker, f"sgd")
    if model is None or scaler is None:
        model, scaler = train_or_update_model(ticker)  # 모델 학습
    else:
        model, scaler = train_or_update_model(ticker, model, scaler)  # 모델 업데이트
    
    try:
        generate_signals(model, scaler, ticker)
    except Exception as e:
        print(f"[error] 신호 생성 중 오류 발생 ({ticker}): {e}")

    save_model(ticker, f"sgd", model, scaler)

# 메인 함수
def main():
    for COIN in COINS:
        cancel_open_orders(COIN)
        process_ticker(COIN)

if __name__ == "__main__":
    main()
