# IBKR Trading Bot (V2)

## Description

This Python script implements a basic algorithmic trading bot that interfaces with Interactive Brokers (IBKR) Trader Workstation (TWS) or Gateway. It features a graphical user interface (GUI) built with Tkinter for managing connections, inputting trading parameters, and viewing logs. The bot fetches historical data, calculates technical indicators (MACD, Stochastic Oscillator, ATR), and executes a predefined trading strategy.

This version is based on a template by Mayank Rasu, with modifications.

## Features

- **IBKR Integration:** Connects to TWS or IBKR Gateway using the `ibapi` Python client library.
- **Graphical User Interface (GUI):**
  - Input fields for IBKR connection parameters (Host, Port, Client ID).
  - Connection status display.
  - Input fields for trading parameters (Tickers, Capital per Ticker).
  - Buttons to connect/disconnect and start/stop the trading strategy.
  - A scrolled text area for real-time logging of bot activities and messages.
- **Historical Data Fetching:** Retrieves historical price data from IBKR for specified tickers.
- **Technical Indicators:** Calculates the following indicators using Pandas:
  - Moving Average Convergence Divergence (MACD).
  - Stochastic Oscillator (%K).
  - Average True Range (ATR) for stop-loss calculation.
- **Trading Strategy:**
  - **Entry:** Places a BUY Market Order when MACD is above its signal line AND the Stochastic Oscillator is above 30 and has crossed up from its previous value.
  - **Stop-Loss:** Places an initial SELL Stop Order based on ATR below the entry price.
  - **Stop-Loss Update (Trailing):** Attempts to update the stop-loss price upwards if the price moves favorably and a position is active.
- **Order Management:**
  - Places Market Orders for entries and Stop Orders for stop-losses.
  - Rudimentary tracking of open orders and positions.
- **Multithreading:** Runs the IBKR connection and trading logic in separate threads to keep the GUI responsive.

## Requirements

- Python 3.x
- `ibapi` (Interactive Brokers Python API)
- `pandas`

## Setup and Installation

1.  **Install Python:** If you don't have Python installed, download it from [python.org](https://www.python.org/).
2.  **Install Libraries:** Open your terminal or command prompt and install the required libraries:
    ```bash
    pip install pandas ibapi
    ```
3.  **Interactive Brokers TWS or Gateway:**
    - You must have either Trader Workstation (TWS) or IBKR Gateway installed and running.
    - Configure TWS/Gateway for API connections:
      - In TWS: Go to `File > Global Configuration > API > Settings`.
      - Enable "Enable ActiveX and Socket Clients".
      - Note the "Socket port" (default is `7497` for paper trading, `7496` for live trading).
      - Add your connection IP to "Trusted IPs" if necessary (e.g., `127.0.0.1` for local connections).

## Configuration

All configurations are done via the GUI when you run the script:

- **Host:** The IP address where TWS/Gateway is running (usually `127.0.0.1` for local).
- **Port:** The socket port TWS/Gateway is listening on (e.g., `7497`).
- **Client ID:** A unique ID for this API connection (e.g., `24`).
- **Tickers:** A comma-separated list of stock symbols to trade (e.g., `AAPL,MSFT,GOOG`).
- **Capital/Ticker:** The amount of capital allocated for trading each ticker.

## How to Run

1.  Ensure your Interactive Brokers TWS or Gateway is running and configured for API access.
2.  Open a terminal or command prompt.
3.  Navigate to the directory where you saved `IBKR_Trading_Bot_V2.py`.
4.  Run the script using:
    ```bash
    python IBKR_Trading_Bot_V2.py
    ```
5.  The GUI will appear.
    - Enter your IBKR connection details (Host, Port, Client ID) and click "Connect".
    - Verify the status changes to "Connected".
    - Enter the tickers and capital per ticker.
    - Click "Start Strategy" to begin the trading logic.
    - Monitor the "Output Log" for messages from the bot.
6.  To stop the strategy, click "Stop Strategy".
7.  To disconnect, click "Disconnect".

## Trading Logic Details (V2)

- **Data Interval:** The script requests historical data for `7 D` (7 days) with `5 mins` (5 minutes) candle size.
- **Indicators Used:**
  - MACD (default parameters: fast=12, slow=26, signal=9).
  - Stochastic Oscillator (default parameters: %K period=20, %D period (smoothing)=3).
  - ATR (default period=60) for stop-loss calculation.
- **Buy Condition:**
  - `MACD_line > MACD_signal_line`
  - `Stochastic_%K > 30`
  - `Current Stochastic_%K > Previous Stochastic_%K` (Stochastic is rising).
  - Only enters if current position for the ticker is less than or equal to the initially recorded position (effectively, allows one entry per initial state).
- **Order Types:**
  - Entry: Market Order (`MKT`).
  - Stop-Loss: Stop Order (`STP`).
- **Stop-Loss Placement:**
  - Initial stop-loss is placed at `Last_Close_Price - ATR_value`.
- **Stop-Loss Trailing Logic:**
  - If a position is active (current position > initial position), the bot checks if the stop-loss can be trailed.
  - It attempts to cancel the existing stop order and place a new one if the `new_stop_price (Last_Close_Price - ATR_value)` is higher than the `old_stop_price`.

## Disclaimer

- **Use at Your Own Risk:** This software is for educational and illustrative purposes only. Trading financial instruments involves substantial risk of loss.
- **Not Financial Advice:** The information and code provided are not financial advice.
- **Test Thoroughly:** Always test trading algorithms extensively in a simulated/paper trading environment before risking real capital.
- **Bugs and Issues:** This software may contain bugs or issues. The authors/contributors are not liable for any losses or damages.

## Known Issues/Limitations (for V2)

- The "Stop Strategy" button provides a basic stop; the trading thread might not terminate immediately or gracefully clean up all operations in this version.
- Waiting for API responses (e.g., after requesting data or positions) uses `time.sleep()`, which is not robust. Event-driven handling would be an improvement.
- Position and order updates rely on periodic requests rather than real-time updates for decision-making within the loop.
- Error handling can be further improved.

## Credits

- Original template: Mayank Rasu (rasuquant.com)
- Modifications: AI Assistant
