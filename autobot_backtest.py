import pyupbit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autobot_apikey import *
from autobot_func import *

# 초기 자본 설정
COIN = "KRW-XRP"
INTERVAL = "minute15"
INITIAL_BALANCE = 1_000_000
TRADE_FEE = 0.0005
TRADE_AMOUNT = 5000

upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 전략 지표 계산 함수들
def calculate_indicators(data):
    rsi = calculate_rsi(data)
    ma = calculate_ma(data)
    ma_100 = calculate_ma(data, period=100)
    macd, signal = calculate_macd(data)
    upper_band, lower_band, close_price = calculate_bollinger_bands(data)
    return rsi, ma, ma_100, macd, signal, upper_band, lower_band, close_price

# 백테스트 함수
def backtest(data):
    balance = INITIAL_BALANCE
    holdings = 0  # 보유 코인 수
    portfolio_values = []  # 포트폴리오 가치 추적
    dates = []  # 날짜 추적
    trade_log = []  # 거래 내역 저장

    for i in range(30, len(data)):
        # 슬라이스로 지표 계산에 필요한 데이터 확보
        sliced_data = data.iloc[:i + 1]
        current_price = sliced_data['close'].iloc[-1]

        # 전략에 필요한 지표 계산
        rsi, ma, ma_100, macd, signal, upper_band, lower_band, close_price = calculate_indicators(sliced_data)
        recent_close_prices = np.mean(sliced_data['close'][-5:])

        # 날짜 및 포트폴리오 가치 업데이트
        dates.append(sliced_data.index[-1])
        portfolio_value = balance + (holdings * current_price)
        portfolio_values.append(portfolio_value)

        trade_data = {
                "data": data,
                "current_price": current_price,
                "rsi": rsi,
                "ma": ma,
                "ma_100": ma_100,
                "macd": macd,
                "signal": signal,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "close_price": close_price,
                "recent_close_prices": recent_close_prices,
            }

        # 매수 조건
        if should_buy(trade_data):
            if balance >= TRADE_AMOUNT:
                # 거래 발생
                amount_to_buy = TRADE_AMOUNT / current_price
                holdings += amount_to_buy
                balance -= TRADE_AMOUNT * (1 + TRADE_FEE)
                trade_log.append(f"BUY: {TRADE_AMOUNT} KRW at {current_price:.2f}")

        # 매도 조건
        elif should_sell(trade_data):
            if holdings > 0:
                # 거래 발생
                sell_amount = holdings
                holdings = 0
                balance += sell_amount * current_price * (1 - TRADE_FEE)
                trade_log.append(f"SELL: {sell_amount} units at {current_price:.2f}")

    # 최종 자본 계산
    final_balance = balance + (holdings * data['close'].iloc[-1] * (1 - TRADE_FEE))
    portfolio_values.append(final_balance)
    dates.append(data.index[-1])
    return final_balance, trade_log, dates, portfolio_values

# 데이터 로드 및 백테스팅 실행
data = pyupbit.get_ohlcv(COIN, INTERVAL, count=200)
final_balance, trade_log, dates, portfolio_values = backtest(data)

# 결과 출력
print(f"Initial Balance: {INITIAL_BALANCE}")
print(f"Final Balance: {final_balance}")
print(f"Net Profit: {final_balance - INITIAL_BALANCE}")
print("\nTrade Log:")
for log in trade_log:
    print(log)

# 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(dates, portfolio_values, label="Portfolio Value", color="blue", linewidth=2)
plt.axhline(y=INITIAL_BALANCE, color="red", linestyle="--", label="Initial Balance")
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (KRW)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()