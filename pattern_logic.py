import pandas as pd
import numpy as np

# === INDICATOR CALCULATION FUNCTIONS ===

def calculate_indicators(df):
    """
    Calculates all technical indicators needed for pattern detection.
    
    This function adds the following indicators to the DataFrame:
    - Moving Averages (MA50, MA200) for trend analysis
    - Exponential Moving Averages (EMA12, EMA26) for MACD calculation
    - MACD and Signal lines for momentum analysis
    - RSI for overbought/oversold conditions
    - EMA Ribbon (5, 8, 13, 21) for trend strength confirmation
    - VWAP for volume-weighted price analysis
    
    Parameters:
    df (DataFrame): DataFrame with OHLCV data (open, high, low, close, volume)
    
    Returns:
    DataFrame: Original DataFrame with all indicators added as new columns
    """
    # Calculate 50-period and 200-period simple moving averages
    # MA50: Short-term trend indicator, MA200: Long-term trend indicator
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()

    # Calculate 12-period and 26-period exponential moving averages (for MACD)
    # EMA gives more weight to recent prices compared to simple moving average
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    
    # MACD line: difference between 12-EMA and 26-EMA
    # Positive MACD = bullish momentum, Negative MACD = bearish momentum
    df['macd'] = df['ema12'] - df['ema26']
    
    # Signal line: 9-period EMA of MACD
    # Used to generate buy/sell signals when MACD crosses above/below signal line
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # MACD Histogram: difference between MACD and signal line
    # Used for momentum confirmation
    df['macd_hist'] = df['macd'] - df['signal']

    # Calculate Relative Strength Index (RSI)
    # RSI measures speed and magnitude of price changes (0-100 scale)
    df['rsi'] = rsi(df['close'])
    
    # Calculate additional EMAs for the EMA ribbon
    # EMA ribbon helps identify trend strength and direction
    ema_list = [5, 8, 13, 21]  # Short to medium-term EMAs
    for ema in ema_list:
        df[f'ema{ema}'] = df['close'].ewm(span=ema).mean()

    # Redundant, but ensures ema21 is present for specific pattern detection
    df['ema21'] = df['close'].ewm(span=21).mean()

    # Calculate True Range and ATR (Average True Range)
    # ATR measures market volatility and is used for stop-loss calculations
    df['tr'] = np.maximum(df['high'] - df['low'],
                  np.maximum(abs(df['high'] - df['close'].shift(1)),
                             abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Calculate cumulative volume and VWAP (Volume Weighted Average Price)
    # VWAP gives more weight to prices with higher volume
    df['cum_volume'] = df['volume'].cumsum()  # Running total of volume
    df['cum_vwap'] = (df['close'] * df['volume']).cumsum()  # Running total of price*volume
    df['vwap'] = df['cum_vwap'] / df['cum_volume']  # Volume-weighted average price
    
    # Calculate ADX (Average Directional Index) for trend strength
    # ADX measures trend strength regardless of direction (0-100 scale)
    # Values above 25 indicate a strong trend
    period = 14
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    # Calculate +DM and -DM
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        up_move = df['high'].iloc[i] - df['high'].iloc[i-1]
        down_move = df['low'].iloc[i-1] - df['low'].iloc[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.iloc[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm.iloc[i] = down_move
    
    # Smooth +DM and -DM using Wilder's smoothing (same as ATR)
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean() * period
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean() * period
    
    # Calculate +DI and -DI
    atr_smooth = df['atr'].ewm(alpha=1/period, adjust=False).mean() * period if 'atr' in df.columns else df['tr'].ewm(alpha=1/period, adjust=False).mean() * period
    
    plus_di = 100 * (plus_dm_smooth / (atr_smooth + 1e-9))
    minus_di = 100 * (minus_dm_smooth / (atr_smooth + 1e-9))
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * (di_diff / (di_sum + 1e-9))
    
    # Smooth DX to get ADX
    df['adx'] = dx.ewm(alpha=1/period, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    return df

def rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a price series using Wilder's smoothing.
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100:
    - Above 70: Overbought (potential sell signal)
    - Below 30: Oversold (potential buy signal)
    - 50: Neutral level (trend change indicator)
    
    Parameters:
    series (Series): Price series (usually closing prices)
    period (int): Number of periods for RSI calculation (default: 14)
    
    Returns:
    Series: RSI values for each period
    """
    delta = series.diff()  # Price change between periods (current - previous)
    gain = delta.clip(lower=0)  # Only positive changes (gains)
    loss = -delta.clip(upper=0)  # Only negative changes (losses, made positive)
    
    # Use Wilder's smoothing (EWMA) instead of simple rolling mean
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss  # Relative strength (ratio of avg gain to avg loss)
    return 100 - (100 / (1 + rs))  # RSI formula: 100 - (100 / (1 + RS))

# === HELPER FUNCTIONS ===

def volume_above_recent(df, lookback=20, factor=0.8):
    """
    Check if current volume is above recent average volume.
    
    Parameters:
    df (DataFrame): DataFrame with volume column
    lookback (int): Number of periods to look back for average volume
    factor (float): Minimum factor above average volume (default: 0.8)
    
    Returns:
    bool: True if current volume >= factor * average volume
    """
    if len(df) < lookback + 1:
        return False
    recent_avg = df['volume'].rolling(lookback).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    return current_volume >= factor * recent_avg

def volume_above_recent_relaxed(df, lookback=20, factor=0.6):  # ADJUSTED: Added relaxed volume check for quieter markets
    """
    ADJUSTED: More lenient volume check for quieter markets.
    Check if current volume is above recent average volume with lower threshold.
    
    Parameters:
    df (DataFrame): DataFrame with volume column
    lookback (int): Number of periods to look back for average volume
    factor (float): Minimum factor above average volume (default: 0.6 - more lenient)
    
    Returns:
    bool: True if current volume >= factor * average volume
    """
    if len(df) < lookback + 1:
        return False
    recent_avg = df['volume'].rolling(lookback).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    return current_volume >= factor * recent_avg

def check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
    """
    Check if current volume is at least min_spike_pct% above the lookback-period average.
    Used for pattern confirmation - patterns are only valid with volume confirmation.
    
    Parameters:
    df (DataFrame): DataFrame with volume column
    min_spike_pct (float): Minimum percentage above average (default: 20%)
    lookback (int): Number of periods to look back for average volume
    
    Returns:
    bool: True if current volume >= (1 + min_spike_pct/100) * average volume
    """
    if len(df) < lookback + 1:
        return False
    recent_avg = df['volume'].rolling(lookback).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    threshold = recent_avg * (1 + min_spike_pct / 100.0)
    return current_volume >= threshold

def check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5, median_period=14):
    """
    Check if current ATR is within acceptable volatility range.
    Ignore signals if ATR is too high (>2x median) or too low (<0.5x median).
    
    Parameters:
    df (DataFrame): DataFrame with atr column
    max_multiplier (float): Maximum ATR multiplier vs median (default: 2.0)
    min_multiplier (float): Minimum ATR multiplier vs median (default: 0.5)
    median_period (int): Period for calculating ATR median
    
    Returns:
    bool: True if ATR is within acceptable range
    """
    if len(df) < median_period + 1 or 'atr' not in df.columns:
        return True  # Default to allowing if we can't calculate
    current_atr = df['atr'].iloc[-1]
    atr_median = df['atr'].rolling(median_period).median().iloc[-1]
    if pd.isna(atr_median) or atr_median == 0:
        return True  # Default to allowing if we can't calculate median
    atr_ratio = current_atr / atr_median
    return min_multiplier <= atr_ratio <= max_multiplier

def check_adx_trend_filter(df, min_adx=25):
    """
    Check if ADX indicates a strong trend (required for momentum entries).
    
    Parameters:
    df (DataFrame): DataFrame with adx column
    min_adx (float): Minimum ADX value for trend confirmation (default: 25)
    
    Returns:
    bool: True if ADX >= min_adx (strong trend present)
    """
    if len(df) < 1 or 'adx' not in df.columns:
        return True  # Default to allowing if we can't calculate
    current_adx = df['adx'].iloc[-1]
    if pd.isna(current_adx):
        return True  # Default to allowing if ADX is NaN
    return current_adx >= min_adx

# === PATTERN LOGIC FUNCTIONS ===

def detect_golden_cross(df):
    """
    Detects a Golden Cross pattern - a bullish signal.
    
    Golden Cross occurs when the 50-period moving average crosses above the 200-period moving average.
    This typically indicates a potential long-term bullish trend reversal.
    
    Parameters:
    df (DataFrame): DataFrame with ma50 and ma200 columns
    
    Returns:
    Series: Boolean series where True indicates Golden Cross occurred
    """
    # Current period: ma50 > ma200 AND previous period: ma50 <= ma200
    # This ensures we detect the exact crossover moment
    return (df['ma50'] > df['ma200']) & (df['ma50'].shift(1) <= df['ma200'].shift(1))

def detect_death_cross(df):
    """
    Detects a Death Cross pattern - a bearish signal.
    
    Death Cross occurs when the 50-period moving average crosses below the 200-period moving average.
    This typically indicates a potential long-term bearish trend reversal.
    
    Parameters:
    df (DataFrame): DataFrame with ma50 and ma200 columns
    
    Returns:
    Series: Boolean series where True indicates Death Cross occurred
    """
    # Current period: ma50 < ma200 AND previous period: ma50 >= ma200
    # This ensures we detect the exact crossover moment
    return (df['ma50'] < df['ma200']) & (df['ma50'].shift(1) >= df['ma200'].shift(1))

def detect_rsi_trend(df):
    """
    Detects RSI trend changes by monitoring crosses above/below the 50 level.
    
    RSI crossing above 50 indicates bullish momentum, below 50 indicates bearish momentum.
    The 50 level is considered neutral and acts as a trend change indicator.
    
    Parameters:
    df (DataFrame): DataFrame with rsi column
    
    Returns:
    tuple: (buy_signals, sell_signals) - both are boolean Series
    """
    # Buy signal: RSI crosses above 50 (current > 50, previous <= 50)
    buy = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    
    # Sell signal: RSI crosses below 50 (current < 50, previous >= 50)
    sell = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    
    return buy, sell

def detect_macd_cross(df):
    """
    Detects MACD bullish and bearish crossovers with histogram confirmation.
    
    MACD crossover occurs when the MACD line crosses above/below the signal line.
    - Bullish crossover: MACD crosses above signal line (potential buy signal)
    - Bearish crossover: MACD crosses below signal line (potential sell signal)
    
    Parameters:
    df (DataFrame): DataFrame with macd, signal, and macd_hist columns
    
    Returns:
    tuple: (bullish_cross, bearish_cross) - both are boolean Series
    """
    # Bullish crossover: MACD > signal AND previous MACD <= previous signal
    # Plus histogram confirmation: current hist > 0 and rising
    buy = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1)) &
           (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1)))
    
    # Bearish crossover: MACD < signal AND previous MACD >= previous signal
    # Plus histogram confirmation: current hist < 0 and falling
    sell = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1)) &
            (df['macd_hist'] < 0) & (df['macd_hist'] < df['macd_hist'].shift(1)))
    
    return buy, sell

def detect_bull_flag(df, flagpole_lookback=20, flag_length=5, min_flagpole_pct=0.03):
    """
    Detects a bull flag pattern - a continuation pattern.
    
    Bull flag consists of:
    1. Flagpole: Strong upward price movement
    2. Flag: Short consolidation period with lower volatility
    3. Breakout: Price breaks above the flag's resistance level
    
    This pattern typically indicates a continuation of the upward trend.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    flagpole_lookback (int): Number of bars to look back for flagpole (default: 20)
    flag_length (int): Number of bars for flag consolidation (default: 5)
    min_flagpole_pct (float): Minimum percentage gain for flagpole (default: 0.03 = 3%)
    
    Returns:
    Series: Boolean series where True indicates bull flag breakout
    """
    flag_breakout = pd.Series(False, index=df.index)
    
    # Loop through each bar starting from where we have enough history
    for i in range(flagpole_lookback + flag_length, len(df)):
        # 1. Find flagpole: sharp upward move in recent history
        start = i - flagpole_lookback - flag_length  # Start of flagpole period
        flagpole_end = i - flag_length  # End of flagpole period
        flagpole_min = df['low'].iloc[start:flagpole_end].min()  # Lowest point in flagpole
        flagpole_max = df['high'].iloc[start:flagpole_end].max()  # Highest point in flagpole
        flagpole_return = (flagpole_max - flagpole_min) / flagpole_min  # Percentage gain
        
        # Skip if flagpole gain is too small
        if flagpole_return < min_flagpole_pct:
            continue
            
        # 2. Flag: consolidation period with lower volatility
        flag_start = flagpole_end  # Start of flag period
        flag_end = i  # End of flag period
        flag_high = df['high'].iloc[flag_start:flag_end].max()  # Resistance level of flag
        flag_low = df['low'].iloc[flag_start:flag_end].min()  # Support level of flag
        flag_range = flag_high - flag_low  # Range of flag consolidation
        
        # Skip if flag is too volatile (should be tight consolidation)
        if flag_range > (flagpole_max - flagpole_min) * 0.5:
            continue  # Flag range should be less than 50% of flagpole range
            
        # 3. Breakout: close above flag resistance level with volume confirmation
        # ADJUSTED: Made volume requirement less strict for quieter markets
        volume_ok = df['volume'].iloc[i] >= 0.6 * df['volume'].iloc[i-20:i].mean()  # ADJUSTED: Reduced from 0.8 to 0.6
        if df['close'].iloc[i] > flag_high and volume_ok:
            flag_breakout.iloc[i] = True
            
    return flag_breakout

def detect_triangle_breakout(df, window=10, tolerance=0.2):
    """
    Detects triangle breakout patterns using linear regression.
    
    Triangle patterns form when price action creates converging trendlines:
    - Ascending triangle: Higher lows with flat highs
    - Descending triangle: Lower highs with flat lows
    - Symmetrical triangle: Both highs and lows converge
    
    Breakout occurs when price breaks above the upper trendline.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    window (int): Number of bars to analyze for triangle formation (default: 10)
    tolerance (float): Tolerance for trendline fitting (default: 0.2 = 20%)
    
    Returns:
    Series: Boolean series where True indicates triangle breakout
    """
    breakout = pd.Series(False, index=df.index)
    
    # Require minimum data for triangle detection
    if len(df) < window + 5:
        return breakout
    
    # Loop through each bar starting from where we have enough history
    for i in range(window, len(df)):
        # Get highs and lows for the analysis window
        highs = df['high'].iloc[i-window:i]
        lows = df['low'].iloc[i-window:i]
        
        # Fit trendlines using linear regression
        x = np.arange(window)  # X-axis: bar positions (0, 1, 2, ...)
        high_fit = np.polyfit(x, highs, 1)  # Linear fit for highs: y = mx + b
        low_fit = np.polyfit(x, lows, 1)    # Linear fit for lows: y = mx + b
        
        # Check for converging lines (upper trendline sloping down, lower trendline sloping up)
        if high_fit[0] < -0.1 and low_fit[0] > 0.1:  # More strict slope requirements
            # Calculate upper trendline value at the current bar
            upper_trend = high_fit[0] * (window-1) + high_fit[1]  # y = mx + b
            
            # Check for significant breakout above upper trendline (with tolerance)
            if df['close'].iloc[i] > upper_trend * (1 + tolerance):
                # ADJUSTED: Made volume requirement less strict for quieter markets
                volume_ok = df['volume'].iloc[i] >= 0.6 * df['volume'].iloc[i-window:i].mean()  # ADJUSTED: Reduced from 0.8 to 0.6
                # ADJUSTED: Made breakout distance requirement less strict for crypto volatility
                breakout_distance = (df['close'].iloc[i] - upper_trend) / upper_trend
                min_distance_ok = breakout_distance >= 0.001  # ADJUSTED: Reduced from 0.002 to 0.001 (0.1% breakout)
                
                if volume_ok and min_distance_ok:
                    breakout.iloc[i] = True
                    
    return breakout

def detect_head_shoulders(df, window=15, tolerance=0.05):
    """
    Detects Head & Shoulders pattern - a reversal pattern.
    
    Head & Shoulders consists of three peaks:
    - Left shoulder: First peak
    - Head: Middle peak (highest)
    - Right shoulder: Third peak (similar height to left shoulder)
    
    This pattern typically indicates a potential trend reversal from bullish to bearish.
    
    Parameters:
    df (DataFrame): DataFrame with high column
    window (int): Number of bars to analyze for pattern (default: 15)
    tolerance (float): Tolerance for shoulder height similarity (default: 0.05 = 5%)
    
    Returns:
    Series: Boolean series where True indicates Head & Shoulders pattern
    """
    hs = pd.Series(False, index=df.index)
    
    # Require minimum data for pattern detection
    if len(df) < window + 5:
        return hs
    
    # Loop through each bar starting from where we have enough history
    for i in range(window, len(df)):
        # Get price segment for analysis
        segment = df['high'].iloc[i-window:i]
        if len(segment) < window:
            continue
            
        # Find peaks in the segment using local maxima
        peaks = []
        for j in range(1, len(segment) - 1):
            if (segment.iloc[j] > segment.iloc[j-1] and 
                segment.iloc[j] > segment.iloc[j+1]):
                peaks.append((j, segment.iloc[j]))
        
        # Need at least 3 peaks for head & shoulders
        if len(peaks) < 3:
            continue
            
        # Sort peaks by height (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top 3 peaks
        top_peaks = peaks[:3]
        
        # Sort by position (left to right)
        top_peaks.sort(key=lambda x: x[0])
        
        if len(top_peaks) >= 3:
            left_shoulder_pos, left_shoulder = top_peaks[0]
            head_pos, head = top_peaks[1] 
            right_shoulder_pos, right_shoulder = top_peaks[2]
            
            # Check pattern conditions:
            # 1. Head should be the highest peak
            # 2. Shoulders should be at similar height (within tolerance)
            # 3. Head should be significantly higher than shoulders
            shoulder_similarity = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
            head_advantage = min(head - left_shoulder, head - right_shoulder) / head
            
            if (
                shoulder_similarity < tolerance and  # Shoulders similar height
                head_advantage > 0.02 and  # Head at least 2% higher than shoulders
                head > left_shoulder and  # Head higher than left shoulder
                head > right_shoulder     # Head higher than right shoulder
            ):
                hs.iloc[i-1] = True  # Mark pattern completion
                
    return hs

def detect_double_top_bottom(df, window=15, tolerance=0.02):
    """
    Detects double top and double bottom patterns - reversal patterns.
    
    Double Top: Two peaks at similar price levels, separated by a valley
    Double Bottom: Two troughs at similar price levels, separated by a peak
    
    These patterns typically indicate potential trend reversals.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    window (int): Number of bars to analyze for pattern (default: 15)
    tolerance (float): Tolerance for price level similarity (default: 0.02 = 2%)
    
    Returns:
    tuple: (double_top, double_bottom) - both are boolean Series
    """
    top = pd.Series(False, index=df.index)
    bottom = pd.Series(False, index=df.index)
    
    # Require minimum data for pattern detection
    if len(df) < window + 5:
        return top, bottom
    
    # Loop through each bar starting from where we have enough history
    for i in range(window, len(df)):
        # Get price segments for analysis
        segment_high = df['high'].iloc[i-window:i]
        segment_low = df['low'].iloc[i-window:i]
        
        # Double Top Detection
        # Find local maxima (peaks) in the segment
        peaks = []
        for j in range(1, len(segment_high) - 1):
            if (segment_high.iloc[j] > segment_high.iloc[j-1] and 
                segment_high.iloc[j] > segment_high.iloc[j+1]):
                peaks.append((j, segment_high.iloc[j]))
        
        # Need at least 2 peaks for double top
        if len(peaks) >= 2:
            # Sort peaks by height (descending) and take top 2
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1_pos, peak1_price = peaks[0]
            peak2_pos, peak2_price = peaks[1]
            
            # Ensure peaks are separated by at least 3 bars
            min_separation = 3
            if abs(peak1_pos - peak2_pos) >= min_separation:
                # Check conditions for double top:
                # 1. Peaks should be at similar price levels (within tolerance)
                # 2. There should be a valley between peaks (dip below peak level)
                price_similarity = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                valley_price = segment_high.iloc[min(peak1_pos, peak2_pos):max(peak1_pos, peak2_pos)+1].min()
                valley_exists = valley_price < max(peak1_price, peak2_price) * (1 - tolerance)
                
                if price_similarity < tolerance and valley_exists:
                    top.iloc[i-1] = True
                    
        # Double Bottom Detection
        # Find local minima (troughs) in the segment
        troughs = []
        for j in range(1, len(segment_low) - 1):
            if (segment_low.iloc[j] < segment_low.iloc[j-1] and 
                segment_low.iloc[j] < segment_low.iloc[j+1]):
                troughs.append((j, segment_low.iloc[j]))
        
        # Need at least 2 troughs for double bottom
        if len(troughs) >= 2:
            # Sort troughs by height (ascending) and take bottom 2
            troughs.sort(key=lambda x: x[1])
            trough1_pos, trough1_price = troughs[0]
            trough2_pos, trough2_price = troughs[1]
            
            # Ensure troughs are separated by at least 3 bars
            min_separation = 3
            if abs(trough1_pos - trough2_pos) >= min_separation:
                # Check conditions for double bottom:
                # 1. Troughs should be at similar price levels (within tolerance)
                # 2. There should be a peak between troughs (rally above trough level)
                price_similarity = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
                peak_price = segment_low.iloc[min(trough1_pos, trough2_pos):max(trough1_pos, trough2_pos)+1].max()
                peak_exists = peak_price > min(trough1_price, trough2_price) * (1 + tolerance)
                
                if price_similarity < tolerance and peak_exists:
                    bottom.iloc[i-1] = True
                
    return top, bottom

def detect_bullish_engulfing(df):
    """
    Detects bullish engulfing candlestick pattern.
    
    Bullish engulfing is a two-candle reversal pattern:
    - First candle: Bearish (close < open)
    - Second candle: Bullish (close > open) that completely engulfs the first candle
    
    This pattern typically indicates a potential bullish reversal.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    
    Returns:
    Series: Boolean series where True indicates bullish engulfing pattern
    """
    pattern = (
        (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish (close < open)
        (df['close'] > df['open']) &                   # Current candle is bullish (close > open)
        (df['open'] < df['close'].shift(1)) &          # Current open below previous close
        (df['close'] > df['open'].shift(1))            # Current close above previous open
    )
    # ADJUSTED: Made volume confirmation less strict for quieter markets
    volume_ok = df['volume'] >= 0.5 * df['volume'].rolling(10).mean()  # ADJUSTED: Reduced from 0.7 to 0.5
    return pattern & volume_ok

def detect_hammer(df):
    """
    Detects hammer candlestick pattern.
    
    Hammer is a single-candle reversal pattern with:
    - Small body (open and close are close together)
    - Long lower shadow (price went much lower but recovered)
    - Little or no upper shadow
    
    This pattern typically indicates a potential bullish reversal.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    
    Returns:
    Series: Boolean series where True indicates hammer pattern
    """
    body = abs(df['open'] - df['close'])  # Size of candle body
    candle_range = df['high'] - df['low']  # Total range of candle
    lower_shadow = df['close'] - df['low']  # Length of lower shadow
    
    # Hammer conditions:
    # 1. Candle range should be much larger than body (at least 3x)
    # 2. Lower shadow should be at least 60% of total range
    return (
        (candle_range > 3 * body) &  # Long range compared to body
        ((lower_shadow / (candle_range + 0.001)) > 0.6)  # Long lower shadow (60%+ of range)
    )

def detect_rsi_divergence(df):
    """
    Detects bullish RSI divergence.
    
    Bullish divergence occurs when:
    - Price makes lower lows (bearish)
    - RSI makes higher lows (bullish)
    
    This divergence suggests weakening bearish momentum and potential reversal.
    
    Parameters:
    df (DataFrame): DataFrame with close and rsi columns
    
    Returns:
    Series: Boolean series where True indicates bullish RSI divergence
    """
    # Compare current values to 5 bars ago
    # Price down: current close < close 5 bars ago
    # RSI up: current RSI > RSI 5 bars ago
    return (df['close'] < df['close'].shift(5)) & (df['rsi'] > df['rsi'].shift(5))

def detect_scalp_breakout(df):
    """
    Detects breakout above previous high - a scalping signal.
    
    This pattern looks for price breaking above the previous bar's high,
    which can indicate short-term momentum and potential scalping opportunity.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC data
    
    Returns:
    Series: Boolean series where True indicates breakout above previous high
    """
    breakout = df['close'] > df['high'].shift(1)
    # ADJUSTED: Made volume confirmation less strict for quieter markets
    volume_ok = df['volume'] >= 0.5 * df['volume'].rolling(10).mean()  # ADJUSTED: Reduced from 0.7 to 0.5
    
    # ADJUSTED: Made minimum breakout distance less strict for crypto volatility
    breakout_distance = (df['close'] - df['high'].shift(1)) / df['high'].shift(1)
    min_distance_ok = breakout_distance >= 0.001  # ADJUSTED: Reduced from 0.002 to 0.001 (0.1% breakout)
    
    return breakout & volume_ok & min_distance_ok

def detect_pullback_to_ema(df):
    """
    Detects pullback to EMA21 and subsequent bounce.
    
    This pattern looks for price that was below EMA21 but has now moved above it,
    indicating a potential bounce off the moving average support.
    
    Parameters:
    df (DataFrame): DataFrame with close and ema21 columns
    
    Returns:
    Series: Boolean series where True indicates pullback bounce
    """
    # Current close above EMA21 AND previous close below previous EMA21
    return (df['close'] > df['ema21']) & (df['close'].shift(1) < df['ema21'].shift(1))

def detect_vwap_bounce(df):
    """
    Detects bounce off VWAP (Volume Weighted Average Price).
    
    This pattern looks for price that touched or dipped below VWAP but closed above it,
    indicating potential support at the volume-weighted average price.
    
    Parameters:
    df (DataFrame): DataFrame with OHLC and vwap columns
    
    Returns:
    Series: Boolean series where True indicates VWAP bounce
    """
    # ADJUSTED: Made VWAP bounce conditions less strict for crypto volatility
    vwap_touch = (df['low'] <= df['vwap'] * 1.002)  # ADJUSTED: Increased tolerance from 1.001 to 1.002
    vwap_bounce = (df['close'] > df['vwap'] * 1.001)  # ADJUSTED: Reduced requirement from 1.002 to 1.001
    # ADJUSTED: Made volume confirmation less strict for quieter markets
    volume_confirmation = df['volume'] > df['volume'].rolling(5).mean() * 0.8  # ADJUSTED: Added 0.8 multiplier for quieter markets
    
    return vwap_touch & vwap_bounce & volume_confirmation

def detect_ema_ribbon_hold(df):
    """
    Detects if price is above all EMAs in the ribbon - indicating strong trend.
    
    EMA ribbon consists of multiple EMAs (5, 8, 13, 21). When price is above all of them,
    it indicates strong bullish momentum and trend continuation potential.
    
    Parameters:
    df (DataFrame): DataFrame with close and ema5, ema8, ema13, ema21 columns
    
    Returns:
    Series: Boolean series where True indicates price above all EMAs
    """
    # Check if close price is above all EMAs in the ribbon
    ema_columns = ['ema5', 'ema8', 'ema13', 'ema21']
    
    # Create a Series to track if price is above all EMAs
    above_all_emas = pd.Series(True, index=df.index)
    
    for ema in ema_columns:
        if ema in df.columns:
            # ADJUSTED: Made EMA ribbon requirement less strict for crypto volatility
            above_all_emas = above_all_emas & (df['close'] > df[ema] * 1.0005)  # ADJUSTED: Reduced from 1.001 to 1.0005 (0.05% above EMA)
    
    return above_all_emas

# === DECISION FUNCTION ===

def decide_trade_signals(df):
    """
    Produce structured trade setups with entry, stop, take-profit, RR target and risk%.
    Returns a list of dicts with keys:
      - name: pattern name
      - side: "BUY" or "SELL"
      - entry: float (close of the signal bar)
      - stop: float (rule-based SL)
      - tp: float (rule-based TP using 2:1 or 3:1 per rulebook)
      - rr: float (2.0 or 3.0)
      - risk_pct: float (default 1.0)
    """
    setups = []
    if len(df) < 200:
        return setups

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    def _passes_quality(side: str, require_volume=True, require_adx=False) -> bool:
        """Enhanced quality gates: trend alignment, volatility filter, volume confirmation, and ADX."""
        try:
            last_close = float(df['close'].iloc[-1])
            ma50 = float(df['ma50'].iloc[-1]) if 'ma50' in df.columns else last_close
            ma200 = float(df['ma200'].iloc[-1]) if 'ma200' in df.columns else last_close
            ema21_val = float(df['ema21'].iloc[-1]) if 'ema21' in df.columns else last_close
            macd_hist = float(df['macd_hist'].iloc[-1]) if 'macd_hist' in df.columns else 0.0
            
            # ATR volatility filter: ignore if too volatile or too flat
            atr_ok = check_atr_volatility_filter(df, max_multiplier=2.0, min_multiplier=0.5)
            if not atr_ok:
                return False
            
            # Volume-weighted confirmation (20% above average)
            if require_volume:
                volume_ok = check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20)
                if not volume_ok:
                    return False
            
            # ADX trend filter for momentum entries
            if require_adx:
                adx_ok = check_adx_trend_filter(df, min_adx=25)
                if not adx_ok:
                    return False
            
            # Trend alignment
            if side == 'BUY':
                trend_ok = (ma50 > ma200) or (last_close > ema21_val and macd_hist > 0)
            else:
                trend_ok = (ma50 < ma200) or (last_close < ema21_val and macd_hist < 0)
            
            return bool(trend_ok)
        except Exception:
            return False  # Fail safe: reject if we can't verify quality

    def add_setup(name, side, stop_rule, base_rr=None):
        entry_price = float(last['close'])
        if stop_rule == 'pattern_low':
            stop_price = float(last['low'])
        elif stop_rule == 'pattern_high':
            stop_price = float(last['high'])
        elif stop_rule == 'recent_swing_low':
            stop_price = float(df['low'].rolling(5).min().iloc[-2])
        elif stop_rule == 'recent_swing_high':
            stop_price = float(df['high'].rolling(5).max().iloc[-2])
        elif stop_rule == 'ema21':
            stop_price = float(last.get('ema21', last['close']))
        elif stop_rule == 'vwap':
            stop_price = float(last.get('vwap', last['close']))
        elif stop_rule == 'atr1':
            # Scalp stop: 1.5x ATR(14)
            atr = float(last.get('atr', (df['high'] - df['low']).rolling(14).mean().iloc[-1]))
            stop_price = entry_price - 1.5 * atr if side == 'BUY' else entry_price + 1.5 * atr
        elif stop_rule == 'atr2':
            # Swing stop: 2.5x ATR(14)
            atr = float(last.get('atr', (df['high'] - df['low']).rolling(14).mean().iloc[-1]))
            stop_price = entry_price - 2.5 * atr if side == 'BUY' else entry_price + 2.5 * atr
        else:
            stop_price = float(last['low'] if side == 'BUY' else last['high'])

        # Guard: avoid zero or equal stop
        if side == 'BUY' and stop_price >= entry_price:
            stop_price = float(min(entry_price * 0.995, last['low']))
        if side == 'SELL' and stop_price <= entry_price:
            stop_price = float(max(entry_price * 1.005, last['high']))

        # Global quality filters before computing RR
        # Determine if this is a momentum entry (requires ADX)
        is_momentum = name in ['Golden Cross', 'Death Cross', 'Bull Flag Breakout', 'Triangle Breakout']
        if not _passes_quality(side, require_volume=True, require_adx=is_momentum):
            return

        risk_per_unit = entry_price - stop_price if side == 'BUY' else stop_price - entry_price
        if risk_per_unit <= 0:
            return
        
        # LAYER 3: Dynamic RR based on confidence (minimum 2:1 R:R)
        if base_rr is None:
            base_rr = 2.0  # Default MIN_RR (increased from 1.3)
        
        # Calculate confidence based on pattern strength and confirmations
        confidence = 70  # Base confidence (matching new threshold)
        # Boost confidence for strong patterns with confirmations
        if check_volume_weighted_confirmation(df, min_spike_pct=20):
            confidence += 5
        if 'adx' in df.columns and df['adx'].iloc[-1] >= 30:
            confidence += 5
        
        try:
            from core.risk_manager import get_dynamic_rr_target
            rr = get_dynamic_rr_target(confidence)
            # Ensure minimum 2:1 R:R
            rr = max(rr, 2.0)
        except:
            rr = base_rr
        
        tp_price = entry_price + rr * risk_per_unit if side == 'BUY' else entry_price - rr * risk_per_unit

        # Calculate actual RR from final values to ensure accuracy
        actual_risk = abs(entry_price - stop_price)
        actual_reward = abs(tp_price - entry_price) if side == 'BUY' else abs(entry_price - tp_price)
        actual_rr = (actual_reward / actual_risk) if actual_risk > 0 else float(rr)

        setups.append({
            'name': name,
            'side': side,
            'entry': round(entry_price, 8),
            'stop': round(stop_price, 8),
            'tp': round(tp_price, 8),
            'rr': round(actual_rr, 2),  # Use actual calculated RR
            'risk_pct': 0.15,  # Reduced from 0.25 to 0.15
            'confidence': confidence
        })

    # Moving Average Crossovers (only on 4h or 1d, confirmed by volume)
    # Note: These should only trigger on higher timeframes (4h, 1d)
    if detect_golden_cross(df).iloc[-1]:
        # Require volume confirmation for Golden Cross
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Golden Cross', 'BUY', 'recent_swing_low', 2.5)  # Swing trade
    if detect_death_cross(df).iloc[-1]:
        # Require volume confirmation for Death Cross
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Death Cross', 'SELL', 'recent_swing_high', 2.5)  # Swing trade

    # RSI Trend Changes - DISABLED (weak pattern per requirements)
    # rsi_buy, rsi_sell = detect_rsi_trend(df)
    # if rsi_buy.iloc[-1]:
    #     add_setup('RSI Cross Above 50', 'BUY', 'recent_swing_low', 1.3)
    # if rsi_sell.iloc[-1]:
    #     add_setup('RSI Cross Below 50', 'SELL', 'recent_swing_high', 1.3)

    # MACD Crossovers - DISABLED (weak pattern per requirements, only keep divergences)
    # macd_buy, macd_sell = detect_macd_cross(df)
    # if macd_buy.iloc[-1]:
    #     add_setup('MACD Bullish Crossover', 'BUY', 'atr2', 1.3)
    # if macd_sell.iloc[-1]:
    #     add_setup('MACD Bearish Crossover', 'SELL', 'atr2', 1.3)

    # Chart Patterns (require volume confirmation)
    if detect_bull_flag(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Bull Flag Breakout', 'BUY', 'recent_swing_low', 2.5)
    if detect_triangle_breakout(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Triangle Breakout', 'BUY', 'recent_swing_low', 2.5)
    if detect_head_shoulders(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Head & Shoulders', 'SELL', 'recent_swing_high', 2.5)

    # Double Top/Bottom Patterns (require volume confirmation)
    dt, db = detect_double_top_bottom(df)
    if dt.iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Double Top', 'SELL', 'recent_swing_high', 2.5)
    if db.iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Double Bottom', 'BUY', 'recent_swing_low', 2.5)

    # Candlestick Patterns (require volume confirmation)
    if detect_bullish_engulfing(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Bullish Engulfing Candle', 'BUY', 'pattern_low', 2.0)
    if detect_hammer(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Hammer Candle', 'BUY', 'pattern_low', 2.0)

    # Divergence and Momentum (require volume confirmation)
    if detect_rsi_divergence(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('RSI Bullish Divergence', 'BUY', 'recent_swing_low', 2.5)

    # Breakout and Support/Resistance
    if detect_scalp_breakout(df).iloc[-1]:
        # Require volume-weighted confirmation (20% above average)
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('Scalp Breakout', 'BUY', 'atr1', 2.0)
    if detect_pullback_to_ema(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('EMA21 Pullback Bounce', 'BUY', 'atr1', 2.0)
    if detect_vwap_bounce(df).iloc[-1]:
        if check_volume_weighted_confirmation(df, min_spike_pct=20, lookback=20):
            add_setup('VWAP Bounce', 'BUY', 'atr1', 2.0)
    # EMA Ribbon Hold - DISABLED (weak pattern per requirements)
    # if detect_ema_ribbon_hold(df).iloc[-1]:
    #     add_setup('EMA Ribbon Hold', 'BUY', 'atr2', 1.3)

    return setups

