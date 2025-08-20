import pandas as pd
import numpy as np
import math

file = "BTCUSDT_2024_YEAR.csv"

dt_str = "2024-01-01 01:00:00"


#df = pd.read_csv(file, usecols=range(5), sep=',')
df = pd.read_csv(file, usecols=range(5))

print(df.head())
print(df.columns)
print(len(df))

a = pd.to_datetime(dt_str)
print(a)
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

print(df.head(5))
data = df[df['open_time'] > pd.to_datetime(dt_str)].head(8000)
data = df[df['open_time'] > pd.to_datetime(dt_str)]#.head(4000)
prices = data['close'].values
opens = data['open'].values
highs = data['high'].values
lows = data['low'].values
times = data['open_time'].values
print(data.head(5))
print(times)
"""
InpRadius = 0.5

InpMinHandleCandles = 5
InpMaxHandleCandles = 30

InpMinCupCandles = 10
InpMaxCupCandles = 80

"""
InpRadius = 0.6

InpMinHandleCandles = 5
InpMaxHandleCandles = 50

InpMinCupCandles = 30
InpMaxCupCandles = 300


####################################
fractal_highs = []
for i in range(1, len(data)-1):
   if highs[i-1] < highs[i] and highs[i] > highs[i+1]:
      fractal_highs.append(i)
   if highs[i-2] < highs[i-1] and highs[i-1] == highs[i] and highs[i] > highs[i+1]:
      fractal_highs.append(i)

print(len(fractal_highs))
####################################
handle_dicts = {}
for fh in fractal_highs:
   for i in range(fh+1, min(len(data)-1, fh+1+InpMaxHandleCandles)):
      if highs[i] > highs[fh]:
         if i > (fh+InpMinHandleCandles):
            handle_dicts[fh] = i
         break

print(len(handle_dicts))
print(handle_dicts)
####################################
cup_dicts = {}
for fh_l_pos in range(len(fractal_highs)-1):
   fh_l_idx = fractal_highs[fh_l_pos]
   for fh_r_pos in range(fh_l_pos+1, len(fractal_highs)):
      fh_r_idx = fractal_highs[fh_r_pos]
      cup_len = abs(fh_r_idx - fh_l_idx)
      if cup_len < InpMinCupCandles:
         continue
      if cup_len > InpMaxCupCandles:
         continue
      tip_high = min(highs[fh_l_idx], highs[fh_r_idx])
      mid_high = highs[fh_l_idx+1 : fh_r_idx]
      if any(h > tip_high for h in mid_high):
         break
      #
      try:
         cup_dicts[fh_r_idx].append(fh_l_idx)
      except:
         cup_dicts[fh_r_idx] = [fh_l_idx]
      break

print(len(cup_dicts))
print(cup_dicts)
####################################

import plotly.graph_objects as go

def getHandleLine(start_time, end_time):
   #
   # Handle pattern using avg price
   end_idx = df.index[df['open_time'] == start_time].tolist()[0]
   handle_start_idx = end_idx
   handle_end_idx = df.index[df['open_time'] == end_time].tolist()[0]
   #
   handle_snapshot = df[(df['open_time'] >= start_time) & (df['open_time'] <= end_time)].copy(deep=True)
   lowest_pos = handle_snapshot['low'].idxmin()
   lowest_positional = df.index.get_loc(lowest_pos)
   target_positional = max(end_idx + InpMinHandleCandles, lowest_positional)
   lowest_row = df.iloc[target_positional]
   #
   lowest_time = lowest_row['open_time']
   handle_snapshot = df[(df['open_time'] >= start_time) & (df['open_time'] <= lowest_time)].copy(deep=True)
   handle_prices = get_high_low(handle_snapshot['high'].values, handle_snapshot['low'].values)
   #
   handle_snapshot['x_time'] = pd.to_datetime(handle_snapshot['open_time']).dt.strftime('%d-%m %H:%M')
   handle_times = handle_snapshot['x_time'].values
   if len(handle_prices) > 0:
      decay = 0.1
      n = len(handle_snapshot)
      idx = np.arange(n)
      weights = 1.0 + decay * ((n - 1 - idx) / (n - 1))
      weights = weights / weights.mean()
      #
      handle_prices[0] = handle_snapshot['high'].iloc[0]
      #_lowest_price = handle_snapshot['low'].loc[handle_end_idx]
      _lowest_price = lowest_row['low']
      handle_prices[len(handle_prices)-1] = _lowest_price
      #
      slope, intercept = np.polyfit(idx, handle_prices, 1)
      reg_line = slope * idx + intercept
      #
      return reg_line
   return None

     
def getCupCurve(start_time, end_time):
   #
   cup_start_idx = df.index[df['open_time'] == start_time][0]
   cup_end_idx = df.index[df['open_time'] == end_time][0]
   #
   cup_snapshot = df[(df['open_time'] >= start_time) & (df['open_time'] <= end_time)].copy(deep=True)
   cup_prices = get_high_low_high(cup_snapshot['high'].values, cup_snapshot['low'].values)
   if len(cup_prices) > 0:
      x = np.arange(len(cup_prices))
      cup_prices[0] = cup_snapshot['high'].loc[cup_start_idx]
      cup_prices[len(cup_prices)-1] = cup_snapshot['high'].loc[cup_end_idx]
      coeffs = np.polyfit(x, cup_prices, 2)
      fit = np.polyval(coeffs, x)
      return fit
   return None


def plotPic(start_time, end_time, handle_end_time, i):
  start_idx = df.index[df['open_time'] == start_time].tolist()
  end_idx = df.index[df['open_time'] == handle_end_time].tolist()
  #
  cup_start_idx = df.index[df['open_time'] == start_time][0]
  cup_end_idx = df.index[df['open_time'] == end_time][0]
  #
  if not start_idx or not end_idx:
      print(f"Index not found for i={i}")
      return

  start_idx = max(start_idx[0] - 5, 0)  # 5 candles before
  end_idx = min(end_idx[0] + 5, len(df)-1)  # 5 candles after
  
  snapshot = df.iloc[start_idx:end_idx+1].copy(deep=True)
  snapshot['x_time'] = pd.to_datetime(snapshot['open_time']).dt.strftime('%d-%m %H:%M')
  # Candlestick chart
  fig = go.Figure(data=[go.Candlestick(
      x=snapshot['x_time'],
      open=snapshot['open'],
      high=snapshot['high'],
      low=snapshot['low'],
      close=snapshot['close'],
      name='Price'
      )])
  # Cup pattern
  cup_snapshot = df[(df['open_time'] >= start_time) & (df['open_time'] <= end_time)].copy(deep=True)
  cup_snapshot['x_time'] = pd.to_datetime(cup_snapshot['open_time']).dt.strftime('%d-%m %H:%M')
  cup_times = cup_snapshot['x_time'].values
  cup_curve_prices = getCupCurve(start_time, end_time)
  if len(cup_prices) > 0:
      fig.add_trace(go.Scatter(
        x=cup_times,
        y=cup_curve_prices,
        mode='lines',
        name='Cup Pattern',
        line=dict(color='blue')#, dash='dash')
        ))
  ##############################
      
  # Handle pattern using avg price
  end_idx = df.index[df['open_time'] == end_time].tolist()[0]
  handle_end_idx = df.index[df['open_time'] == handle_end_time].tolist()[0]
  #
  handle_snapshot = df[(df['open_time'] >= end_time) & (df['open_time'] <= handle_end_time)].copy(deep=True)
  lowest_pos = handle_snapshot['low'].idxmin()
  lowest_positional = df.index.get_loc(lowest_pos)
  target_positional = max(end_idx + InpMinHandleCandles, lowest_positional)
  lowest_row = df.iloc[target_positional]
  #
  lowest_time = lowest_row['open_time']
  handle_snapshot = df[(df['open_time'] >= end_time) & (df['open_time'] <= lowest_time)].copy(deep=True)
  handle_snapshot['x_time'] = pd.to_datetime(handle_snapshot['open_time']).dt.strftime('%d-%m %H:%M')
  handle_times = handle_snapshot['x_time'].values
  handle_line_prices = getHandleLine(end_time, handle_end_time)
  if len(handle_line_prices) > 0:
     fig.add_trace(go.Scatter(
        x=handle_times,
        y=handle_line_prices,
        mode='lines',
        name='Handle Pattern',
        line=dict(color='green')
        ))
  #################
  fig.update_layout(
     title="OHLC Snapshot",
     xaxis_title="Time",
     yaxis_title="Price",
     xaxis_rangeslider_visible=False
     )
  fig.write_image("ohlc_snapshot_plotly" + str(i) + ".png", width=1200, height=600)

####################################
from sklearn.metrics import mean_squared_error, r2_score

def get_high_low(_highs, _lows):
    global lows, highs
    val = []
    if len(_highs) == 0:
       return []
    mid_idx = len(_highs) // 2
    end_idx = len(_highs)-1
    _lowest = 99999999999
    for i in range(len(_highs)):
       height = (_highs[i] - _lows[i])
       dif = (end_idx - i) / end_idx
       _val = _lows[i] + height * dif
       _lowest = min(_lowest, _lows[i])
       if i == 0:
          _val = _highs[i]
       if i == len(_highs):
          _val = _lowest
       val.append(_val)
    return val

def get_high_low_high(_highs, _lows):
    global lows, highs
    val = []
    if len(_highs) == 0:
       return []
    mid_idx = len(_highs) // 2
    end_idx = len(_highs)-1
    for i in range(len(_highs)):
       height = (_highs[i] - _lows[i])
       dif = (mid_idx - i) / mid_idx
       _val = _lows[i] + height * dif
       if i > mid_idx:
          dif = (i - mid_idx) / end_idx
          _val = _lows[i] + height * dif
       if i == 0:
          _val = _highs[i]
       if i == len(_highs):
          _val = _highs[i]
       val.append(_val)
    return val

def get_radius_high_low_high(left_tip_idx, right_tip_idx):
   cup_len = abs(right_tip_idx - left_tip_idx) + 1
   cup_values = values
   ####################
   #cup_values = get_high_low_high(left_tip_idx, right_tip_idx)
   ####################
   x = np.arange(cup_len)
   coeffs = np.polyfit(x, cup_values, 2)
   if coeffs[0] <= 0:
      return -1
   fit = np.polyval(coeffs, x)
   r2 = r2_score(cup_values, fit)
   return r2

def get_radius(values):
   cup_len = len(values)
   cup_values = values
   ####################
   x = np.arange(cup_len)
   coeffs = np.polyfit(x, cup_values, 2)
   if coeffs[0] <= 0:
      return -1
   fit = np.polyval(coeffs, x)
   r2 = r2_score(cup_values, fit)
   return r2

count = 0

#import ta
#data = df[df['open_time'] > pd.to_datetime(dt_str)].head(4000)
#data['ATR_14'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
"""
{'cup_height':cup_height,
                 'handle_height':handle_height,
                 'cup_candles':cup_len,
                 'handle_candles':handle_candles,
                 'cup_radius':_radius,
                 'cup_left_idx': fh_l_idx,
                 'cup_right_idx': fh_r_idx,
                 'handle_bottom_idx': np.argmin(handle_lows),
                 'break_idx': break_idx,
                 'cup_left_tip_curve': cup_curve_prices[0],
                 'cup_right_tip_curve': cup_curve_prices[-1],
                 'cup_left_tip_candle_high': highs[fh_l_idx],
                 'cup_right_tip_candle_high': highs[fh_r_idx],
                 'cup_left_tip_candle_curve_gap': abs(cup_curve_prices[0] - highs[fh_l_idx]),
                 'cup_right_tip_candle_curve_gap': abs(cup_curve_prices[-1] - highs[fh_r_idx]),
                 'cup_tips_gap_min': cup_tips_gap_min,
                 'cup_tips_curve_gap': abs(cup_curve_prices[0]-cup_curve_prices[-1]),
                 'handle_line_angle_deg': str(handle_line_angle_deg),
                 'handle_cup_join_gap': abs(cup_curve_prices[-1] - handle_line_prices[0])
                 }
"""
def cup_score(cup):
   height = cup['cup_height']
   score = 1
   ### CUP
   score -= (cup['cup_left_tip_curve'] - cup['cup_right_tip_curve']) / height
   score -= (cup['cup_left_tip_candle_high'] - cup['cup_right_tip_candle_high']) / height
   score -= (cup['cup_left_tip_candle_curve_gap'] + cup['cup_right_tip_candle_curve_gap']) / height
   score -= max(cup['cup_left_tip_candle_curve_gap'], cup['cup_right_tip_candle_curve_gap']) / height
   score -= cup['cup_tips_curve_gap'] / height
   score -= cup['cup_tips_gap_min'] / height
   score += (cup['cup_radius'] - 0.75) * 0.1
   ### HANDLE
   score -= (cup['handle_cup_join_gap'])/height
   #score += float(cup['handle_line_angle_deg']) / height
   return score
   
high = data['high']
low = data['low']
close = data['close']

tr = pd.concat([
    high - low,
    (high - close.shift()).abs(),
    (low - close.shift()).abs()
], axis=1).max(axis=1)

atr14 = tr.rolling(window=14, min_periods=14).mean()

data['ATR_14'] = atr14

patterns = []

for fh_r_idx, fh_l_idxs in cup_dicts.items():
   #if count > 5:
   #   break
   if fh_r_idx not in handle_dicts:
      continue
   break_idx = handle_dicts[fh_r_idx]
   for fh_l_idx in fh_l_idxs:
      # Get cup highs and lows
      cup_highs = highs[fh_l_idx:fh_r_idx+1]
      cup_lows = lows[fh_l_idx:fh_r_idx+1]
      #
      cup_top_min = min(cup_highs[0], cup_highs[-1])
      cup_top_max = max(cup_highs[0], cup_highs[-1])
      cup_bottom = min(cup_lows)
      cup_height = cup_top_min - cup_bottom
      #
      handle_lows = lows[fh_r_idx:break_idx]
      handle_bottom = min(handle_lows)
      handle_top = min(highs[fh_r_idx:break_idx-1])
      handle_height = handle_top - handle_bottom
      handle_candles = max(InpMinHandleCandles, abs(fh_r_idx-break_idx))
      HANDLE_BOTTOM = min(lows[fh_r_idx:(fh_r_idx+handle_candles)])
      CUP60 = (cup_bottom+(cup_top_min-cup_bottom)*0.6)
      #
      if CUP60 > HANDLE_BOTTOM:
         continue
      """
      if _handle_bottom < (cup_top_min-(cup_top_min*0.4)):
         continue
      """
      ####################
      # Check price action after breakout
      broke_well = False
      """
      #should_broke_price = cup_top_max#handle_bottom+(handle_bottom*0.5)
      #should_broke_price = prices[fh_r_idx] + data.loc[break_idx, 'ATR_14']
      try:
         #should_broke_price = prices[fh_r_idx] + data.at[break_idx, 'ATR_14']
         should_broke_price = prices[fh_r_idx] + atr14.iloc[break_idx]
      except Exception as e:
         print(len(data['ATR_14']))
         print(break_idx)
         print(fh_r_idx)
         print(e)
         exit()
      """
      should_broke_price = prices[fh_r_idx] + atr14.iloc[break_idx]
      for i in range(break_idx, min(break_idx+20, len(lows))):
         if lows[i] <= handle_bottom:
            # Price returned to handle bottom before reaching 50%
            break
         if highs[i] >= should_broke_price:
            # Price reached 50% level - valid pattern
            broke_well = True
            break
      if broke_well == False:
         break
      ###################
      # Find left, bottom, right
      cup_prices = prices[fh_l_idx:fh_r_idx+1]
      left = prices[fh_l_idx]
      right = prices[fh_r_idx]
      bottom_idx = np.argmin(cup_prices)
      bottom = cup_prices[bottom_idx]
      # Check symmetry: left ≈ right (within 3%)
      if abs(left - right) / max(left, right) > 0.04:
         continue
      ####################
      # 2. Pattern matching using normalized MSE vs U-shape
      cup_len = len(cup_highs)
      x = np.linspace(-1, 1, cup_len)
      ideal_u = x**2
      #
      """
      normalized = (cup_highs - np.min(cup_highs)) / (np.max(cup_highs) - np.min(cup_highs))
      mse = mean_squared_error(normalized, ideal_u)
      if mse > 0.04:
         continue
      #
      normalized = (cup_lows - np.min(cup_lows)) / (np.max(cup_lows) - np.min(cup_lows))
      mse = mean_squared_error(normalized, ideal_u)
      if mse > 0.04:
         continue
      """
      #
      normalized = (cup_prices - np.min(cup_prices)) / (np.max(cup_prices) - np.min(cup_prices))
      #normalized = (cup_highs - np.min(cup_highs)) / (np.max(cup_highs) - np.min(cup_highs))
      mse = mean_squared_error(normalized, ideal_u)
      if mse > 0.06:
         continue
      ###################
      # Left/right symmetry
      """
      left = prices[fh_l_idx]
      right = prices[fh_r_idx - 1]
      bottom_idx = np.argmin(cup_prices)
      bottom = cup_prices[bottom_idx]
      if abs(left - right) / max(left, right) > 0.05:
         continue
      if max(left, right) - bottom <= 0:
         continue
      #
      """
      left = lows[fh_l_idx]
      right = lows[fh_r_idx - 1]
      bottom_idx = np.argmin(cup_lows)
      bottom = cup_lows[bottom_idx]
      if abs(left - right) / max(left, right) > 0.07:
         continue
      if max(left, right) - bottom <= 0:
         continue
      ####################
      
      """
      _radius = get_radius(cup_highs)
      print(fh_r_idx , fh_l_idx, _radius)
      if _radius < InpRadius:
         continue
      """
      """
      _radius = get_radius(cup_highs)
      print(fh_r_idx , fh_l_idx, _radius)
      if _radius < InpRadius:
         continue
      """
      _high_low_high = get_high_low_high(cup_highs, cup_lows)
      _radius = get_radius(_high_low_high)
      print(fh_r_idx , fh_l_idx, _radius)
      if _radius < InpRadius:
         continue
      ####################
      #print("hBott: " + str(handle_bottom) + ", h40%: " + str(cup_top_min-(cup_height*0.4)))
      #print((handle_bottom < (cup_top_min-(cup_height*0.4))))
      print("Found.... ", times[fh_l_idx], times[fh_r_idx], times[break_idx], 0)
      ### CUP
      cup_curve_prices = getCupCurve(times[fh_l_idx], times[fh_r_idx])
      cup_tips_gap_min = 0
      if highs[fh_l_idx] < lows[fh_r_idx]:
         cup_tips_gap_min = max((lows[fh_r_idx] - highs[fh_l_idx]), cup_tip_gap_min)
      if lows[fh_l_idx] > highs[fh_r_idx]:
         cup_tips_gap_min = max((lows[fh_l_idx] - highs[fh_r_idx]), cup_tip_gap_min)
      ### HANDLE 
      handle_line_prices = getHandleLine(times[fh_r_idx], times[break_idx])
      # angle in degrees
      handle_line_angle_deg = (max(handle_line_prices)-min(handle_line_prices)) / float(len(handle_line_prices))
      print(handle_line_prices)
      #
      pattern = {'cup_height':cup_height,
                 'handle_height':handle_height,
                 'cup_candles':cup_len,
                 'handle_candles':handle_candles,
                 'cup_radius':_radius,
                 'cup_left_idx': fh_l_idx,
                 'cup_right_idx': fh_r_idx,
                 'handle_bottom_idx': np.argmin(handle_lows),
                 'break_idx': break_idx,
                 'cup_left_tip_curve': cup_curve_prices[0],
                 'cup_right_tip_curve': cup_curve_prices[-1],
                 'cup_left_tip_candle_high': highs[fh_l_idx],
                 'cup_right_tip_candle_high': highs[fh_r_idx],
                 'cup_left_tip_candle_curve_gap': abs(cup_curve_prices[0] - highs[fh_l_idx]),
                 'cup_right_tip_candle_curve_gap': abs(cup_curve_prices[-1] - highs[fh_r_idx]),
                 'cup_tips_gap_min': cup_tips_gap_min,
                 'cup_tips_curve_gap': abs(cup_curve_prices[0]-cup_curve_prices[-1]),
                 'handle_line_angle_deg': handle_line_angle_deg,
                 'handle_cup_join_gap': abs(cup_curve_prices[-1] - handle_line_prices[0]),
                 'CUP60': str(CUP60),
                 'HANDLE_BOTTOM': str(HANDLE_BOTTOM)
                 }
      print(pattern['handle_line_angle_deg'])
      pattern['score'] = cup_score(pattern)
      patterns.append(pattern)
      count = count + 1
      if count > 100:
         print("READCHED MAX")
         exit()
         continue
      #break
   #break


def get_best_30_patterns111(patterns):
    # Get top 30
    df = pd.DataFrame(patterns)
    top_30 = df.sort_values('score', ascending=False).head(30)

    # Return as list of tuples without the score column
    return list(top_30.drop(columns='score').itertuples(index=False, name=None))

def get_best_30_patterns(patterns):
    # Sort patterns so highest score comes first
    sorted_patterns = sorted(patterns, key=lambda x: x['score'], reverse=True)
    
    # Keep only the top 30
    return sorted_patterns[:30]

best_patterns = get_best_30_patterns(patterns)

print(best_patterns)
print(len(best_patterns))
print(len(patterns))
#exit()

_count = 0
for cup_handle in best_patterns:
   print("Drawing.... ", times[fh_l_idx], times[fh_r_idx], times[break_idx], 0)
   plotPic(times[cup_handle['cup_left_idx']], times[cup_handle['cup_right_idx']], times[cup_handle['break_idx']], _count)
   _count = _count + 1
   print("Drawn : " + str(_count) + " ->> " + str(cup_handle['score']))
   print(cup_handle)
   print("-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-")
      

print(best_patterns)
print("patterns: " + str(len(patterns)))
print("best_patterns: " + str(len(best_patterns)))








