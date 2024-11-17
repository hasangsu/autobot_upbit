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