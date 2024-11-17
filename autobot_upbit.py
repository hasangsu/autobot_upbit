import pyupbit
import numpy as np
from autobot_apikey import *
from autobot_func import *

# coin & env
#COIN = "KRW-XRP"
COINS = ["KRW-BTC", "KRW-XRP", "KRW-ETH"]
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

# trade bot func
def trade_bot():
    try:
        trade_message = f"trade_bot() has been executed\n"
        for COIN in COINS:
            print(f"process {COIN}")
            
            cancel_open_orders(COIN)

            data = pyupbit.get_ohlcv(COIN, interval=INTERVAL, count=30)
            current_price = pyupbit.get_current_price(COIN)
            recent_close_prices = float(np.mean(data['close'][-5:]))

            rsi = calculate_rsi(data)
            ma = calculate_ma(data)
            ma_100 = calculate_ma(data, period=100)
            macd, signal = calculate_macd(data)
            upper_band, lower_band, close_price = calculate_bollinger_bands(data)

            print(
                f"현재 가격: {current_price:.2f}, RSI: {float(rsi.iloc[-1]):.2f}, MA: {ma:.2f}, "
                f"MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}, "
                f"BB 상단: {upper_band:.2f}, 하단: {lower_band:.2f}"
            )

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
                krw_balance = upbit.get_balance("KRW")
                if krw_balance > BUY_AMOUNT:
                    buy_result = upbit.buy_market_order(COIN, BUY_AMOUNT)
                    trade_message += f"[notify] buy {COIN} units for {BUY_AMOUNT} KRW: {buy_result}\n"
                else:
                    trade_message+= f"[error] don't have the cash to buy {COIN}\n"

            # 매도 조건
            elif should_sell(trade_data):
                coin_balance = upbit.get_balance(COIN.split('-')[1])
                if coin_balance > 0:
                    sell_amount = coin_balance * 0.5
                    sell_value_in_krw = sell_amount * current_price
                    if sell_value_in_krw < 5000:
                        trade_message += f"[error] No sales below 5000won"
                    else:
                        sell_result = upbit.sell_market_order(COIN, sell_amount)
                        trade_message += f"[notify] sell {COIN} {sell_amount}: {sell_result}\n"
                else:
                    trade_message += f"[error] sell fail: dont have {COIN}\n"

            else:
                trade_message += f"[notify] {COIN} condition mismatch: not eligible to buy or sell\n"

        notify_slack(SLACK_HOOKS_URL, trade_message, "notify")

    except Exception as e:
        print(f"[error] {e}")

# 메인 함수 선언
def main():
    print("[start] upbit tradebot")
    trade_bot()
    print("[finish] upbit tradebot")

# 메인 함수 호출
if __name__ == "__main__":
    main()

'''
macd와 signal의 교차라인 비교
    - 하향 교차 (Bearish Crossover)
        - 매도의 신호 (시장이 과열된 후 하락 전환)
        - (macd[-1] < signal[-1] and macd[-2] > signal[-2])
    - 상향 교차 (Bullish Crossover)
        - 매수의 신호 (시장이 하락 후 상승으로 전환)
        - or (macd[-1] > signal[-1] and macd[-2] < signal[-2])
'''

