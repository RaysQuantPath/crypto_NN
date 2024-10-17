import pandas as pd
from binance.client import Client
import traceback
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import multiprocessing

BINANCE_API_KEY = "not needed for market data download"
BINANCE_SECRET_KEY = "not needed for market data download"

client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

coinList = client.get_all_tickers()
assets = [coin["symbol"] for coin in coinList if coin["symbol"].endswith("USDT")]


def get_market_data(symbol, window='1 YEAR UTC', interval='15m', is_spot=True):
    try:
        # 定义时间间隔字典
        interval_mapping = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR,
            '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '3d': Client.KLINE_INTERVAL_3DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1M': Client.KLINE_INTERVAL_1MONTH
        }

        # 获取对应的时间间隔
        kline_interval = interval_mapping.get(interval, Client.KLINE_INTERVAL_15MINUTE)

        # 获取k线数据
        # 现货数据
        if is_spot:
            klines = client.get_historical_klines(symbol, kline_interval, window)

        # 永续合约期货数据
        else:

            df = client.futures_klines(symbol=symbol, interval=kline_interval, limit=1500)
            klines = pd.DataFrame(df, columns=[
                "Date", 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'trade_money',
                'trade_count', 'buy_volume', 'sell_volume', 'other'
            ])
            klines = klines[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        if not klines:
            print("数据没下载成功")
            return None

        # 转换为DataFrame
        trades = pd.DataFrame(klines, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        trades['Date'] = pd.to_datetime(trades['Date'], unit='ms')
        trades.set_index('Date', inplace=True)
        trades = trades.apply(pd.to_numeric, errors='coerce')
        trades['Asset_name'] = symbol
        return trades

    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        traceback.print_exc()
        return None


def download_and_save_data(symbol, interval='15m'):
    try:
        data = get_market_data(symbol, interval=interval, is_spot=True)

        if isinstance(data, pd.DataFrame):
            os.makedirs(f'asset_data/raw_data_{interval}', exist_ok=True)

            data.to_csv(f'asset_data/raw_data_{interval}/{symbol}.csv')
        else:
            print(f"Data for {symbol} is not a valid DataFrame.")

    except Exception as e:
        print(f"Failed to download {symbol}: {e}")


if __name__ == "__main__":
    # Check the number of CPUs available
    print(f"Number of CPUs: {multiprocessing.cpu_count()}")
    num_processes = min(cpu_count() - 1, len(assets))
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(download_and_save_data, assets), total=len(assets)):
            pass

    
