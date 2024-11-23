import numpy as np
import pandas as pd
import requests
import os
from joblib import dump, load

# 모델과 스케일러 저장 및 로드
def save_model(ticker, extension, model, scaler):
    print(f"[notify] save model {ticker}")
    dump(model, f"{ticker}_{extension}_model.pkl")
    dump(scaler, f"{ticker}_{extension}_scaler.pkl")

def load_model(ticker, extension):
    model_path = f"{ticker}_{extension}_model.pkl"
    scaler_path = f"{ticker}_{extension}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"[notify] success load {ticker} model")
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    else:
        print(f"[notify] error load {ticker} model -> new model")
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