import requests
import time
import os
import json
from datetime import datetime, timedelta

class BinanceDataFetcher:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT", interval="1m", request_limit=1200, folder="btc_data"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.us/api/v3/klines"
        self.symbol = symbol
        self.interval = interval
        self.request_limit = request_limit
        self.headers = {'X-MBX-APIKEY': self.api_key}
        self.folder = folder

        # Ensure the folder exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_timestamp(self, dt):
        return int(dt.timestamp() * 1000)

    def get_klines(self, start_time, end_time):
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "startTime": self.get_timestamp(start_time),
            "endTime": self.get_timestamp(end_time),
            "limit": 1000
        }
        
        response = requests.get(self.base_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching data: {response.text}")

    def save_data(self, data, start_time):
        file_path = os.path.join(self.folder, f"btc_data_{start_time.strftime('%Y-%m-%d')}.json")
        with open(file_path, "a") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")

    def get_latest_timestamp(self):
        # Check the folder for the latest saved timestamp
        file_list = sorted([f for f in os.listdir(self.folder) if f.startswith("btc_data_")])
        if not file_list:
            return None
        
        # Extract the most recent file
        latest_file = file_list[-1]
        latest_file_path = os.path.join(self.folder, latest_file)
        
        with open(latest_file_path, "r") as f:
            lines = f.readlines()
            if lines:
                last_data = json.loads(lines[-1])
                last_timestamp = int(last_data[0])
                return datetime.utcfromtimestamp(last_timestamp / 1000)
        
        return None

    def fetch_and_save(self, from_date, to_date):
        latest_timestamp = self.get_latest_timestamp()
        
        if latest_timestamp and latest_timestamp > from_date:
            print(f"Resuming from {latest_timestamp}")
            from_date = latest_timestamp + timedelta(minutes=1)

        while from_date < to_date:
            try:
                # Fetch data for a batch of 1000 minutes
                klines = self.get_klines(from_date, from_date + timedelta(minutes=999))
                if not klines:
                    print("No data returned, exiting.")
                    break

                # Save the data with the start time in the file name
                self.save_data(klines, from_date)

                # Update the start time for the next batch
                from_date += timedelta(minutes=1000)

                # Wait between requests to respect rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}. Retrying in 10 seconds...")
                time.sleep(10)

        print("Data collection complete.")

# Usage
api_key = "yTEAOFnvav9KAqiuPZ27llBzzTg4AONDOpSx6aBFsmxjaXO6TMnoRnNfZ9wqgiJ1"
api_secret = "C0JwuHRNhYSViXlbH5XQEEE3Nz9yXj74OO5XH2pCrb4oq9IrqBNmXgwGxgp5ol4L"

# Set date range
from_date = datetime.utcnow() - timedelta(days=7)  # 2 months back
to_date = datetime.utcnow()  # Up to the current time

fetcher = BinanceDataFetcher(api_key, api_secret)
fetcher.fetch_and_save(from_date, to_date)
