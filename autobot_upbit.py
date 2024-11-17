import pyupbit
import numpy as np
from autobot_func import *

# slack api webhooks
SLACK_HOOKS_URL = "https://hooks.slack.com/services/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"

# upbit api keys
ACCESS_KEY = "P4oH9bjAqXHQ2JbyTkogKkbrGR0BNPQXvWpUI8Ml"
SECRET_KEY = "craU0SgR68iZBFdwZ1bKy9UxwWY7x2phu12kAOtf"

# coin & env
#COIN = "KRW-XRP"
COINS = ["KRW-BTC", "KRW-XRP"]
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
        notify_slack(SLACK_HOOKS_URL, "trade_bot() has been executed", "notify")

        buy_message = ""
        sell_message = ""
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
        
            # 매수 조건
            if (
                (float(rsi.iloc[-1]) < 30 and float(macd.iloc[-1]) > float(signal.iloc[-1]) and close_price <= lower_band and current_price < ma)
                or (close_price <= lower_band and close_price > lower_band * 1.02)  # 볼린저밴드 하단 반등
                or (close_price <= ma_100 and close_price > ma_100 * 0.98)  # 주요 지지선 근처에서 반등
                or (close_price >= recent_close_prices * 1.05)  # 최근 하락폭 대비 가격 회복
            ):
                krw_balance = upbit.get_balance("KRW")
                if krw_balance > BUY_AMOUNT:
                    buy_result = upbit.buy_market_order(COIN, BUY_AMOUNT)
                    buy_message += f"[notify] buy {COIN} units for {BUY_AMOUNT} KRW: {buy_result}"
                    # print(f"[buy] {COIN} units for {BUY_AMOUNT} KRW: {buy_result}")
                else:
                    print(f"[error buy] don't have the cash to buy {COIN}")

            else:
                print(f"[condition mismatch] not eligible to buy")

            # 매도 조건
            if (
                (float(rsi.iloc[-1]) > 70 and float(rsi.iloc[-3]) > float(rsi.iloc[-2]) > float(rsi.iloc[-1]))  # RSI가 연속 하락
                or (float(macd.iloc[-1]) < float(signal.iloc[-1]) and float(macd.iloc[-2]) > float(signal.iloc[-2]))  # MACD와 Signal 하향 교차
                or (close_price >= upper_band and close_price < upper_band * 0.98)  # 볼린저 밴드 상단 돌파 후 2% 하락
                or (current_price < recent_close_prices * 0.95)  # 최근 5개 캔들 평균보다 5% 하락
                or (data['volume'].iloc[-1] < recent_close_prices * 0.8)  # 거래량 감소:
            ):
                coin_balance = upbit.get_balance(COIN.split('-')[1])
                if coin_balance > 0:
                    sell_amount = coin_balance * 0.5
                    sell_value_in_krw = sell_amount * current_price
                    if sell_value_in_krw < 5000:
                        print(f"[error sell] No sales below 5000won")
                    else:
                        sell_result = upbit.sell_market_order(COIN, sell_amount)
                        sell_message += f"[notify] sell {COIN} {sell_amount}: {sell_result}"
                        # print(f"[sell] {sell_amount} units {COIN}: {sell_result}")
                else:
                    print(f"[error sell] dont have {COIN}" )

            else:
                print(f"[condition mismatch] not eligible to sell")

        notify_slack(SLACK_HOOKS_URL, buy_message, "notify")
        notify_slack(SLACK_HOOKS_URL, sell_message, "notify")

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

