from src.data_collection import download_stock_prices
from src.data_processing import process_stock_data


def main():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "AMD", "META", "AVGO", "TSM", "JPM"]

    # Raw data
    prices = download_stock_prices(tickers)

    print("Downloaded price data:")
    print(prices.head())
    print("\nData shape:", prices.shape)

    # Processed Data
    returns = process_stock_data()

    print("\nProcessed daily return data:")
    print(returns.head())
    print("\nData shape:", returns.shape)

if __name__ == "__main__":
    main()
