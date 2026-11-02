from src.data_collection import download_stock_prices


def main():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    prices = download_stock_prices(tickers)

    print("Downloaded price data:")
    print(prices.head())
    print("\nData shape:", prices.shape)


if __name__ == "__main__":
    main()
