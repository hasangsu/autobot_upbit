import os
from dotenv import load_dotenv
load_dotenv()
import pyupbit
import pandas as pd
import pandas_ta as ta
from joblib import dump, load
import json
import schedule
import time
from datetime import datetime
import requests
import base64

# Setup
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
COINS = ["KRW-XRP", "KRW-XLM", "KRW-ETH", "KRW-DOGE"]
TRAILING_STOP_PERCENT = 0.05
SELL_PERCENT = 50

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

def get_current_price(ticker):
    current_price = pyupbit.get_current_price(ticker)
    return current_price

# 트레일링 스탑 로드
def load_trailing_stop(ticker, load_path="."):
    # 트레일링 스탑 파일 경로 생성
    trailing_stop_path = os.path.join(load_path, f"{ticker}_trailingstop.json")
    
    if os.path.exists(trailing_stop_path):
        with open(trailing_stop_path, 'r') as f:
            trail_stop_data = json.load(f)

        return trail_stop_data['buy_price'], trail_stop_data['trail_stop_price']
    else:
        return None, None

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

def execute_sell(ticker, percentage):
    try:
        coin_balance = upbit.get_balance(ticker.split('-')[1])
        amount_to_sell = coin_balance * (percentage / 100)
        current_price = pyupbit.get_orderbook(ticker)['orderbook_units'][0]["ask_price"]
        if current_price * amount_to_sell > 5000:  # Ensure the order is above the minimum threshold
            result = upbit.sell_market_order(ticker, amount_to_sell)
            print("Sell order successful:", result)

            return result
    except Exception as e:
        print(f"Failed to execute sell order: {e}")
        return None

def trailing_stop():
    for COIN in COINS:
        try:
            current_price = get_current_price(COIN)
            buy_price, trail_stop_price = load_trailing_stop(COIN, ".")
            
            # 트레일링 스탑 갱신 (현재가 상승 시 상향 조정)
            if buy_price is not None and trail_stop_price is not None:
                new_trail_stop_price = current_price * (1 - TRAILING_STOP_PERCENT)
                if current_price > trail_stop_price:  # trail_stop_price 기준 상승 확인
                    trail_stop_price = max(trail_stop_price, new_trail_stop_price)
                    save_trailing_stop(COIN, buy_price, trail_stop_price, ".")

            # 트레일링 스탑 진행
            if buy_price is not None and trail_stop_price is not None and current_price <= trail_stop_price:
                result = execute_sell(COIN, SELL_PERCENT)

                timestamp = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
                slack_message = f"Timestamp: {timestamp}, Result: {result}"
                notify_slack(os.getenv("SLACK_HOOKS_URL"), slack_message, "notify")

        except Exception as e:
            print(f"Error: {e}")
            return

if __name__ == "__main__":
    # Schedule the task 5 minutes
    schedule.every(5).minutes.do(trailing_stop)

    while True:
        schedule.run_pending()
        time.sleep(1)