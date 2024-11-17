import numpy as np
import pandas as pd
import requests

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