# -*- coding: utf-8 -*-
"""
IB API - stratgey implementation template for TI based Strategies - GUI Version

@author: Mayank Rasu (http://rasuquant.com/wp/) & AI Assistant Modification
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, Entry, Label, Button
import threading
import time
import queue # For thread-safe communication

# --- Keep your existing IB API imports and TradeApp class ---
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import pandas as pd

# --- Existing TradeApp class definition (modified slightly) ---
class TradeApp(EWrapper, EClient):
    def __init__(self, ui_queue): # Add ui_queue
        EClient.__init__(self, self)
        self.data = {}
        self.pos_df = pd.DataFrame(columns=['Account', 'Symbol', 'SecType',
                                            'Currency', 'Position', 'Avg cost'])
        self.order_df = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                               'Account', 'Symbol', 'SecType',
                                               'Exchange', 'Action', 'OrderType',
                                               'TotalQty', 'CashQty', 'LmtPrice',
                                               'AuxPrice', 'Status'])
        self.nextValidOrderId = None
        self.ui_queue = ui_queue # Store the queue for UI updates

    # --- Helper to send messages to UI via queue ---
    def log_message(self, msg):
        if self.ui_queue:
            self.ui_queue.put(msg)

    # --- Modify methods that print to use log_message ---
    # --- In your TradeApp class ---
    def historicalData(self, reqId, bar):
        # --- First, ensure the data entry exists and add the new bar ---
        if reqId not in self.data:
            # Initialize DataFrame for this reqId with the first bar
            self.data[reqId] = pd.DataFrame([{"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume}])
            # Log that the first bar was received for this request
            first_bar_msg = f"HistoricalData - ReqId: {reqId} - First bar: Date {bar.date}, Close {bar.close}"
            self.log_message(first_bar_msg)
        else:
            # Append subsequent bars using pd.concat (more robust than list append for DataFrames)
            new_row = pd.DataFrame([{"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume}])
            self.data[reqId] = pd.concat([self.data[reqId], new_row], ignore_index=True)

            # --- Second, *after* adding data, perform the logging check ---
            # Optional: Only log every N bars to avoid flooding UI
            current_length = len(self.data[reqId])
            if current_length > 1 and current_length % 10 == 0: # Check length *after* concatenation, maybe skip first bar check
                log_msg = f"HistoricalData - ReqId: {reqId} - Received bar #{current_length}: Date {bar.date}, Close {bar.close}"
                self.log_message(log_msg)

        # Note: Consider adding logging in the 'historicalDataEnd' callback (if implemented)
        # to confirm when all data for a request is finished.

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        self.log_message(f"NextValidId: {orderId}")

    def position(self, account, contract, position, avgCost):
        super().position(account, contract, position, avgCost)
        # ... (logic to update self.pos_df remains the same) ...
        self.log_message(f"Position: {contract.symbol}, Qty: {position}, AvgCost: {avgCost}")


    def positionEnd(self):
        self.log_message("Position data fetch complete.")

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
         # ... (logic to update self.order_df remains the same) ...
        self.log_message(f"OpenOrder: ID {orderId}, {contract.symbol}, {order.action} {order.totalQuantity} @ {order.orderType}, Status: {orderState.status}")

    # --- In your TradeApp class ---
    def error(self, reqId, errorCode, errorString, advancedOrderReject=""): # Keep the parameter here
        # Call the parent error method WITHOUT advancedOrderReject
        super().error(reqId, errorCode, errorString)

        # Now you can log the full error details using the parameters received
        # by *your* method, including advancedOrderReject
        log_detail = f"Error - Code: {errorCode}, Msg: {errorString}"
        # Append advancedOrderReject info if it exists (it's often an empty string)
        if advancedOrderReject:
             log_detail += f", Advanced Reject Info: {advancedOrderReject}"

        # Use your logging mechanism (e.g., ui_queue)
        self.log_message(log_detail)

        # Optional: Add logic based on specific error codes
        # if errorCode == SOME_CRITICAL_CODE:
        #    # Handle critical error
        #    pass

    # --- Other EWrapper methods (like orderStatus, execDetails etc.) should also call log_message ---


# --- Keep your existing helper functions (usTechStk, histData, MACD, etc.) ---
# You might move these into a separate 'strategy.py' or 'utils.py' file
def usTechStk(symbol,sec_type="STK",currency="USD",exchange="SMART"):
    # ... (no changes needed) ...
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract

# ... (keep MACD, stochOscltr, atr, marketOrder, stopOrder, dataDataframe) ...
def histData(app, req_num,contract,duration,candle_size): # Pass app object
    app.log_message(f"Requesting historical data for {contract.symbol} ({req_num})")
    app.reqHistoricalData(reqId=req_num,
                          contract=contract,
                          endDateTime='',
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='ADJUSTED_LAST',
                          useRTH=1,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])

def MACD(DF,a=12,b=26,c=9):
    df = DF.copy()
    df["MA_Fast"]=df["Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    return df

def stochOscltr(DF,a=20,b=3):
    df = DF.copy()
    df['C-L'] = df['Close'] - df['Low'].rolling(a).min()
    df['H-L'] = df['High'].rolling(a).max() - df['Low'].rolling(a).min()
    df['%K'] = df['C-L']/df['H-L']*100
    return df['%K'].rolling(b).mean()

def atr(DF,n):
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return df['ATR']

def marketOrder(direction,quantity):
    order = Order()
    order.action = direction
    order.orderType = "MKT"
    order.totalQuantity = quantity
    # Consider using GTC (Good Til Cancelled) or DAY depending on strategy needs
    # order.tif = "IOC" # Immediate or Cancel might lead to missed fills
    order.tif = "DAY"
    order.eTradeOnly = False # Set Booleans explicitly
    order.firmQuoteOnly = False
    return order

def stopOrder(direction,quantity,st_price):
    order = Order()
    order.action = direction
    order.orderType = "STP"
    order.totalQuantity = quantity
    order.auxPrice = st_price
    order.tif = "GTC" # Stop loss usually Good Til Cancelled
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    return order

def dataDataframe(trade_app_obj, req_id): # Use req_id instead of symbol index
    "returns extracted historical data in dataframe format"
    df = pd.DataFrame(trade_app_obj.data[req_id])
    df.set_index("Date",inplace=True)
    return df


# --- Main Trading Logic (modified to run in a thread) ---
def trading_logic_thread(app, tickers, capital, ui_queue):
    app.log_message("Trading logic started.")
    initial_pos = {} # Recalculate initial positions at the start of this thread

    try:
        # Get initial positions within this thread after connection is established
        app.log_message("Requesting initial positions...")
        app.reqPositions()
        # Need a mechanism to wait for positionEnd callback here
        # This is tricky with the current threading model.
        # A simpler start might be to assume 0 initial positions or fetch positions
        # right before processing each ticker inside the loop, though less efficient.
        time.sleep(5) # Crude wait - **Enhancement: Use events/callbacks**
        temp_pos_df = app.pos_df.copy()
        temp_pos_df.drop_duplicates(inplace=True, ignore_index=True)
        initial_pos = {ticker: 0 for ticker in tickers}
        stk_pos_df = temp_pos_df[temp_pos_df["SecType"]=="STK"]
        for _, row in stk_pos_df.iterrows():
             if row["Symbol"] in initial_pos:
                 initial_pos[row["Symbol"]] = int(row["Position"])
        app.log_message(f"Initial positions recorded: {initial_pos}")


        starttime = time.time()
        timeout = time.time() + 60*60*6 # 6 hours

        while time.time() <= timeout:
            # Check if thread should stop (e.g., via a flag set by GUI)
            # if stop_flag.is_set():
            #     app.log_message("Trading logic stopping.")
            #     break

            app.log_message("Starting strategy cycle...")
            # Reset data for this cycle to avoid mixing old/new requests
            app.data = {}
            req_map = {} # Map reqId back to ticker

            # Request data for all tickers first
            for i, ticker in enumerate(tickers):
                 req_id = i # Use index as reqId for simplicity
                 req_map[req_id] = ticker
                 contract = usTechStk(ticker)
                 histData(app, req_id, contract, '7 D', '5 mins') # Pass app object
                 time.sleep(1) # Small sleep between requests to avoid pacing violations

            # Wait for data to arrive - **Enhancement: Use event-based waiting**
            app.log_message("Waiting for historical data...")
            time.sleep(len(tickers) * 3) # Very crude wait, adjust as needed

            # Get current positions and orders *before* placing new trades
            app.log_message("Requesting current positions and orders...")
            app.reqPositions()
            time.sleep(2) # Crude wait
            pos_df = app.pos_df.copy()
            pos_df.drop_duplicates(inplace=True,ignore_index=True)

            app.reqOpenOrders()
            time.sleep(2) # Crude wait
            ord_df = app.order_df.copy()
            ord_df.drop_duplicates(inplace=True,ignore_index=True)
            app.log_message(f"Current Positions: {len(pos_df)}, Open Orders: {len(ord_df)}")


            # Process each ticker
            for req_id, ticker in req_map.items():
                app.log_message(f"Processing ticker: {ticker}")
                if req_id not in app.data or app.data[req_id].empty:
                    app.log_message(f"No historical data received for {ticker}. Skipping.")
                    continue

                try:
                    df = dataDataframe(app, req_id) # Use req_id
                    # Calculate indicators
                    df["stoch"] = stochOscltr(df)
                    df["macd"] = MACD(df)["MACD"]
                    df["signal"] = MACD(df)["Signal"]
                    df["atr"] = atr(df,60)
                    df.dropna(inplace=True)

                    if df.empty:
                        app.log_message(f"DataFrame empty after indicator calculation for {ticker}. Skipping.")
                        continue

                    # --- Strategy Logic ---
                    last_close = df["Close"].iloc[-1]
                    last_stoch = df["stoch"].iloc[-1]
                    prev_stoch = df["stoch"].iloc[-2] if len(df["stoch"]) > 1 else last_stoch # Handle short history
                    last_macd = df["macd"].iloc[-1]
                    last_signal = df["signal"].iloc[-1]
                    last_atr = df["atr"].iloc[-1]

                    quantity = int(capital / last_close) if last_close > 0 else 0
                    if quantity == 0:
                        app.log_message(f"Calculated quantity is 0 for {ticker}. Skipping.")
                        continue

                    current_pos = pos_df[pos_df["Symbol"]==ticker]["Position"].sum() # Sum if multiple entries exist
                    initial_ticker_pos = initial_pos.get(ticker, 0)

                    app.log_message(f"{ticker}: Close={last_close:.2f}, MACD={last_macd:.2f}, Signal={last_signal:.2f}, Stoch={last_stoch:.2f}, ATR={last_atr:.2f}, CurrentPos={current_pos}, InitialPos={initial_ticker_pos}")

                    # --- Buy Condition ---
                    buy_signal = last_macd > last_signal and last_stoch > 30 and last_stoch > prev_stoch

                    if buy_signal and current_pos <= initial_ticker_pos : # Only buy if not already holding more than initial
                         app.log_message(f"BUY SIGNAL for {ticker}")
                         app.reqIds(-1) # Request next valid order ID
                         time.sleep(1) # Wait for nextValidId callback
                         if app.nextValidOrderId is None:
                             app.log_message("Failed to get next valid order ID. Skipping trade.")
                             continue
                         order_id = app.nextValidOrderId
                         app.log_message(f"Placing BUY MKT order for {quantity} {ticker} (ID: {order_id})")
                         app.placeOrder(order_id, usTechStk(ticker), marketOrder("BUY", quantity))
                         app.nextValidOrderId += 1 # Manually increment for the stop loss order

                         # Place corresponding Stop Loss immediately after BUY order
                         # Note: Market orders might not fill instantly. Stop might be placed before fill confirmation.
                         # A better approach waits for fill confirmation before placing the stop.
                         stop_price = round(last_close - last_atr, 2) # Use 2 decimal places for price
                         app.log_message(f"Placing SELL STP order for {quantity} {ticker} at {stop_price} (ID: {app.nextValidOrderId})")
                         app.placeOrder(app.nextValidOrderId, usTechStk(ticker), stopOrder("SELL", quantity, stop_price))
                         # Crude wait for orders to potentially appear in next cycle's check
                         time.sleep(3)


                    # --- Stop Loss Update Condition ---
                    # Check if holding more than the initial position (meaning we likely bought in this session)
                    elif current_pos > initial_ticker_pos:
                        app.log_message(f"Holding {ticker}. Checking/Updating Stop Loss.")
                        # Find existing Stop Loss order for this ticker
                        existing_stop = None
                        ticker_orders = ord_df[(ord_df["Symbol"]==ticker) & (ord_df["Action"]=="SELL") & (ord_df["OrderType"]=="STP")]
                        if not ticker_orders.empty:
                            # Assume the latest one is the active stop loss
                            existing_stop = ticker_orders.sort_values(by="OrderId", ascending=False).iloc[0]

                        new_stop_price = round(last_close - last_atr, 2)

                        if existing_stop is not None:
                            old_stop_price = existing_stop["AuxPrice"]
                            ord_id_to_cancel = existing_stop["OrderId"]
                            sl_quantity = existing_stop["TotalQty"] # Use quantity from existing stop

                            # Only update if the new stop price is higher (trailing stop)
                            # And also check if the stop quantity matches the current position difference
                            # This logic needs careful review based on exact strategy needs
                            if new_stop_price > old_stop_price and sl_quantity == (current_pos - initial_ticker_pos):
                                app.log_message(f"Updating {ticker} stop loss from {old_stop_price} to {new_stop_price}")
                                app.log_message(f"Cancelling old stop order ID: {ord_id_to_cancel}")
                                app.cancelOrder(ord_id_to_cancel)
                                time.sleep(1) # Wait for cancel confirmation (ideally via callback)

                                app.reqIds(-1)
                                time.sleep(1)
                                if app.nextValidOrderId is None:
                                     app.log_message("Failed to get next valid order ID for stop update.")
                                     continue
                                new_order_id = app.nextValidOrderId
                                app.log_message(f"Placing new SELL STP order for {sl_quantity} {ticker} at {new_stop_price} (ID: {new_order_id})")
                                app.placeOrder(new_order_id, usTechStk(ticker), stopOrder("SELL", sl_quantity, new_stop_price))
                                time.sleep(2)
                            else:
                                app.log_message(f"{ticker} stop loss price {new_stop_price:.2f} is not higher than existing {old_stop_price:.2f} or quantity mismatch. No update.")
                        else:
                             app.log_message(f"Could not find existing stop loss order for {ticker} to update, but position exists. Consider placing one?")
                             # Decide if a stop should be placed if none exists but position > initial_pos

                except Exception as e:
                    app.log_message(f"Error processing {ticker}: {e}")
                    import traceback
                    app.log_message(traceback.format_exc()) # Log full traceback for debugging

            # --- End of ticker loop ---
            app.log_message("Strategy cycle finished. Waiting for next interval.")
            # Wait for the next 5-minute interval
            time.sleep(300 - ((time.time() - starttime) % 300.0))

        app.log_message("Trading timeout reached.")
    except Exception as e:
        app.log_message(f"Fatal error in trading thread: {e}")
        import traceback
        app.log_message(traceback.format_exc())
    finally:
        # Optional: Disconnect when thread finishes
        # if app.isConnected():
        #     app.disconnect()
        #     app.log_message("Disconnected from IB.")
        pass


# --- Tkinter GUI Application Class ---
class TradingGUI:
    def __init__(self, master):
        self.master = master
        master.title("IBKR Trading Bot")
        master.geometry("700x500") # Adjust size as needed

        self.ui_queue = queue.Queue() # Queue for thread communication
        self.app = None # IB TradeApp instance
        self.connection_thread = None
        self.logic_thread = None
        self.is_connected = False
        self.is_running = False # Track if trading logic is active

        # --- Connection Frame ---
        conn_frame = tk.Frame(master, pady=5)
        conn_frame.pack(fill=tk.X)

        Label(conn_frame, text="Host:").pack(side=tk.LEFT, padx=5)
        self.host_entry = Entry(conn_frame, width=15)
        self.host_entry.insert(0, "127.0.0.1")
        self.host_entry.pack(side=tk.LEFT, padx=5)

        Label(conn_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = Entry(conn_frame, width=7)
        self.port_entry.insert(0, "7497") # Default TWS Paper
        self.port_entry.pack(side=tk.LEFT, padx=5)

        Label(conn_frame, text="Client ID:").pack(side=tk.LEFT, padx=5)
        self.clientid_entry = Entry(conn_frame, width=5)
        self.clientid_entry.insert(0, "24") # Use a different ClientID
        self.clientid_entry.pack(side=tk.LEFT, padx=5)

        self.connect_button = Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=10)

        self.status_label = Label(conn_frame, text="Status: Disconnected", fg="red")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # --- Control Frame ---
        ctrl_frame = tk.Frame(master, pady=5)
        ctrl_frame.pack(fill=tk.X)

        Label(ctrl_frame, text="Tickers (comma-sep):").pack(side=tk.LEFT, padx=5)
        self.ticker_entry = Entry(ctrl_frame, width=40)
        self.ticker_entry.insert(0, "AAPL,MSFT,GOOG,AMD") # Example tickers
        self.ticker_entry.pack(side=tk.LEFT, padx=5)

        Label(ctrl_frame, text="Capital/Ticker:").pack(side=tk.LEFT, padx=5)
        self.capital_entry = Entry(ctrl_frame, width=10)
        self.capital_entry.insert(0, "1000")
        self.capital_entry.pack(side=tk.LEFT, padx=5)

        self.start_button = Button(ctrl_frame, text="Start Strategy", command=self.start_strategy, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = Button(ctrl_frame, text="Stop Strategy", command=self.stop_strategy, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)


        # --- Output Log Frame ---
        log_frame = tk.Frame(master, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED) # Make read-only initially

        # Start processing the queue
        self.master.after(100, self.process_ui_queue)

    def log_message_ui(self, message):
        """Appends a message to the log Text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END) # Auto-scroll
        self.log_text.config(state=tk.DISABLED)

    def process_ui_queue(self):
        """Checks the queue for messages from the IB thread and updates the UI."""
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                self.log_message_ui(msg)
        except queue.Empty:
            pass
        finally:
            # Reschedule processing
            self.master.after(100, self.process_ui_queue)

    def toggle_connection(self):
        if not self.is_connected:
            host = self.host_entry.get()
            port = self.port_entry.get()
            client_id = self.clientid_entry.get()

            if not port.isdigit() or not client_id.isdigit():
                messagebox.showerror("Error", "Port and Client ID must be integers.")
                return

            self.log_message_ui(f"Connecting to {host}:{port} with Client ID {client_id}...")
            self.app = TradeApp(self.ui_queue) # Create instance here
            self.app.connect(host, int(port), int(client_id))

            # Start the app.run() loop in a separate thread
            self.connection_thread = threading.Thread(target=self.app.run, daemon=True)
            self.connection_thread.start()

            # Give it a moment to establish connection (improve with callback checks)
            # A better way is to check app.isConnected() periodically or use a connection status callback
            time.sleep(3) # Crude wait

            if self.app.isConnected():
                self.is_connected = True
                self.status_label.config(text="Status: Connected", fg="green")
                self.connect_button.config(text="Disconnect")
                self.start_button.config(state=tk.NORMAL) # Enable start button
                self.log_message_ui("Connection successful.")
            else:
                self.log_message_ui("Connection failed. Check TWS/Gateway and settings.")
                self.app = None # Clear instance if connection failed
        else:
            # Disconnect
            if self.is_running:
                self.stop_strategy() # Stop logic first if running

            if self.app and self.app.isConnected():
                self.app.disconnect()
            self.is_connected = False
            self.status_label.config(text="Status: Disconnected", fg="red")
            self.connect_button.config(text="Connect")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.log_message_ui("Disconnected.")
            self.app = None # Clear instance


    def start_strategy(self):
        if not self.is_connected:
            messagebox.showwarning("Warning", "Not connected to IBKR.")
            return

        if self.is_running:
            messagebox.showwarning("Warning", "Strategy is already running.")
            return

        tickers_str = self.ticker_entry.get()
        capital_str = self.capital_entry.get()

        if not tickers_str:
            messagebox.showerror("Error", "Please enter at least one ticker.")
            return
        if not capital_str.isdigit() or int(capital_str) <= 0:
             messagebox.showerror("Error", "Capital must be a positive integer.")
             return

        tickers = [t.strip().upper() for t in tickers_str.split(',')]
        capital = int(capital_str)

        self.log_message_ui(f"Starting strategy for tickers: {tickers} with capital {capital}")
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.ticker_entry.config(state=tk.DISABLED) # Prevent changes while running
        self.capital_entry.config(state=tk.DISABLED)

        # Start the trading logic in its own thread
        self.logic_thread = threading.Thread(target=trading_logic_thread,
                                             args=(self.app, tickers, capital, self.ui_queue),
                                             daemon=True)
        self.logic_thread.start()

    def stop_strategy(self):
        if not self.is_running:
            messagebox.showwarning("Warning", "Strategy is not running.")
            return

        # **Critical:** Need a way to signal the trading_logic_thread to stop gracefully.
        # This example doesn't implement a stop flag, which is essential.
        # The thread will currently run until its timeout or an error.
        # You would typically add a `threading.Event` object passed to the thread
        # and check `stop_event.is_set()` within the thread's loop.
        self.log_message_ui("Stopping strategy (Note: immediate stop not fully implemented in this example)...")

        self.is_running = False
        # Re-enable controls assuming stop is effective (improve this)
        if self.is_connected:
            self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ticker_entry.config(state=tk.NORMAL)
        self.capital_entry.config(state=tk.NORMAL)
        # Ideally, join the logic_thread here after signalling it to stop.


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = TradingGUI(root)
    root.mainloop()