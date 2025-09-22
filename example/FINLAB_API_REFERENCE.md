# 📚 FinLab API 完整參考文檔

> **版本**: 基於 finlab 官方文檔 (更新至 2025-08-28)  
> **最後更新**: 2025-08-28  
> **用途**: 台股量化交易系統開發參考文檔  
> **來源**: 純粹基於 finlab 官方文檔

## 📋 目錄

**🚀 初學者快速入門**
1. [10分鐘快速開始](#10分鐘快速開始)
2. [第一個策略](#第一個策略)

**📘 完整API參考**
3. [核心模組總覽](#核心模組總覽)
4. [認證與登入](#認證與登入)
5. [數據獲取 (finlab.data)](#數據獲取-finlabdata)
6. [回測系統 (finlab.backtest)](#回測系統-finlabbacktest)
7. [機器學習 (finlab.ml)](#機器學習-finlabml)
   - [機器學習流程](#機器學習流程概述)
   - [特徵工程](#特徵工程-finlabmlfeature)
   - [標籤生成](#標籤生成-finlabmllabel)
   - [模型訓練與預測](#模型訓練與預測)
   - [AutoML自動機器學習](#automl自動機器學習)
   - [因子分析框架](#因子分析-finlabmlfactor_analysis)
8. [視覺化工具 (finlab.plot)](#視覺化工具-finlabplot)
   - [技術指標圖組](#技術指標圖組)
   - [板塊圖與樹狀圖](#漲跌幅與成交金額板塊圖)
   - [財務指標雷達圖](#財務指標雷達圖)
   - [估值河流圖](#本益比河流圖)
   - [策略部位監控](#策略部位旭日圖)
9. [投資組合 (finlab.portfolio)](#投資組合-finlabportfolio)
   - [權重優化](#投資組合權重優化)
   - [績效分析](#投資組合分析)
   - [風險管理](#投資組合風險分析)
10. [即時交易 (finlab.online)](#即時交易-finlabonline)
   - [帳戶連接](#帳戶連接與管理)
   - [下單API](#下單api)
   - [部位管理](#部位與資產管理)
   - [策略自動執行](#策略自動執行)
   - [風險控制](#風險控制與監控)
11. [最佳實踐](#最佳實踐)

---

## 🚀 10分鐘快速開始

### 步驟1: 安裝 FinLab
```bash
# 安裝 FinLab 主套件
pip install finlab

# 安裝量化交易相關依賴
pip install ta-lib pandas numpy matplotlib seaborn scikit-learn
```

### 步驟2: 設置認證
```python
import finlab
from finlab import data

# 設置 API token (從環境變數或直接設置)
finlab.login(api_token='your_api_token_here')
# 或使用環境變數: export FINLAB_API_TOKEN='your_token'
```

### 步驟3: 獲取數據
```python
# 獲取台積電收盤價
tsmc_close = data.get('price:收盤價')['2330']

# 獲取所有股票的收盤價 (最常用)
close_prices = data.get('price:收盤價')

# 獲取成交量
volumes = data.get('price:成交股數')

# 獲取台股上市股票列表
stocks = data.universe()
print(f"共有 {len(stocks)} 檔股票")
```

### 步驟4: 簡單技術分析
```python
import pandas as pd

# 計算20日移動平均線
ma20 = close_prices.rolling(20).mean()

# 計算RSI指標 (使用內建指標)
rsi = data.indicator('RSI', period=14)

# 找出RSI < 30的股票 (超賣訊號)
oversold_stocks = rsi[rsi < 30].dropna()
print(f"超賣股票數量: {len(oversold_stocks.columns)}")
```

## 第一個策略

### 簡單的趨勢跟隨策略
```python
from finlab import backtest

def create_simple_momentum_strategy():
    """
    創建簡單動能策略：
    1. 買進條件：股價突破20日均線 + RSI > 50
    2. 賣出條件：股價跌破20日均線 OR RSI < 30
    """
    
    # 獲取基本數據
    close = data.get('price:收盤價')
    volume = data.get('price:成交股數')
    
    # 計算技術指標
    ma20 = close.rolling(20).mean()
    rsi = data.indicator('RSI', period=14)
    
    # 流動性篩選 (日成交值 > 500萬)
    avg_dollar_volume = (close * volume).rolling(30).mean()
    liquid_stocks = avg_dollar_volume > 5_000_000
    
    # 買進條件
    buy_signal = (
        (close > ma20) &           # 價格突破均線
        (rsi > 50) &              # RSI > 50 (動能向上)
        liquid_stocks             # 流動性充足
    )
    
    # 賣出條件 (持有時使用)
    sell_signal = (
        (close < ma20) |          # 價格跌破均線
        (rsi < 30)               # RSI < 30 (過度賣出)
    )
    
    return buy_signal, sell_signal

# 執行策略
buy_signals, sell_signals = create_simple_momentum_strategy()

# 回測設置
position = backtest.sim(
    signals=buy_signals,
    resample='D',               # 日頻率再平衡
    position_limit=30,          # 最多持有30檔股票
    trade_at_price='open',      # 以開盤價交易
    fee_ratio=0.001425,         # 手續費 0.1425%
    tax_ratio=0.003,            # 交易稅 0.3%
    initial_capital=30_000_000  # 3000萬初始資金
)

# 查看策略績效
report = backtest.report(position)
print(f"總報酬率: {report.get('總報酬率', 'N/A')}")
print(f"年化報酬率: {report.get('年化報酬率', 'N/A')}")
print(f"最大回撤: {report.get('最大回撤', 'N/A')}")
print(f"夏普比率: {report.get('夏普比率', 'N/A')}")

# 視覺化結果
backtest.plot(position)
```

### 策略優化範例
```python
def optimize_strategy_parameters():
    """
    優化策略參數：測試不同的移動平均線週期
    """
    results = {}
    
    for ma_period in [10, 20, 30, 50]:
        print(f"測試 MA{ma_period} 策略...")
        
        close = data.get('price:收盤價')
        ma = close.rolling(ma_period).mean()
        rsi = data.indicator('RSI', period=14)
        
        # 修改後的買進條件
        buy_signal = (close > ma) & (rsi > 50)
        
        # 回測
        position = backtest.sim(
            signals=buy_signal,
            position_limit=20,
            initial_capital=10_000_000
        )
        
        report = backtest.report(position)
        results[f"MA{ma_period}"] = {
            '年化報酬率': report.get('年化報酬率', 0),
            '最大回撤': report.get('最大回撤', 0),
            '夏普比率': report.get('夏普比率', 0)
        }
    
    # 找出最佳參數
    best_strategy = max(results.keys(), 
                       key=lambda k: results[k]['夏普比率'])
    print(f"\n最佳策略: {best_strategy}")
    print(f"績效: {results[best_strategy]}")
    
    return results

# 執行參數優化
optimization_results = optimize_strategy_parameters()
```

---

## 核心模組總覽

```python
finlab/
├── finlab                 # 主模組：認證與基礎功能
│   ├── login()           # 登入平台
│   └── get_token()       # 取得API token
├── finlab.data           # 數據獲取與處理
│   ├── get()            # 核心數據獲取函數
│   ├── indicator()      # 技術指標計算
│   ├── universe()       # 股票池篩選
│   ├── search()         # 數據集搜尋
│   └── set_storage()    # 本地存儲設定
├── finlab.dataframe      # DataFrame擴展功能
├── finlab.backtest       # 回測引擎
│   ├── sim()            # 策略模擬
│   ├── report()         # 績效報告
│   └── plot()          # 視覺化
├── finlab.report         # 報表生成
├── finlab.ml             # 機器學習工具
│   ├── feature          # 特徵工程
│   └── label           # 標籤生成
├── finlab.portfolio      # 投資組合管理
└── finlab.online         # 即時交易介面
    ├── Account          # 帳戶管理
    ├── Position         # 部位管理
    └── OrderExecutor    # 訂單執行器
```

---

## 認證與登入

### 基本登入
```python
import finlab

# 方法1: 直接設置token
finlab.login(api_token='your_api_token_here')

# 方法2: 使用環境變數
import os
os.environ['FINLAB_API_TOKEN'] = 'your_api_token_here'
finlab.login()

# 驗證登入狀態
print(f"登入狀態: {finlab.is_login()}")
```

### 取得帳戶資訊
```python
# 取得當前token
current_token = finlab.get_token()

# 檢查API額度
quota_info = finlab.check_quota()
print(f"剩餘API調用次數: {quota_info}")
```

---

## 數據獲取 (finlab.data)

### 核心數據函數

#### `data.get()` - 主要數據獲取函數
```python
from finlab import data

# 基本用法
data.get(
    dataset: str,                    # 數據集名稱
    save_to_storage: bool = True,    # 是否保存到本地
    force_download: bool = False     # 是否強制重新下載
) -> pd.DataFrame

# 常用數據集範例
close_prices = data.get('price:收盤價')
volumes = data.get('price:成交股數') 
open_prices = data.get('price:開盤價')
high_prices = data.get('price:最高價')
low_prices = data.get('price:最低價')
```

#### `data.indicator()` - 技術指標計算

FinLab 支援超過 **100 種技術指標**，整合 talib 和 pandas_ta：

```python
# 基本技術指標
rsi = data.indicator('RSI', timeperiod=14)
sma = data.indicator('SMA', timeperiod=20)
ema = data.indicator('EMA', timeperiod=12)

# 進階技術指標
macd_line, macd_signal, macd_hist = data.indicator('MACD', 
    fastperiod=12, slowperiod=26, signalperiod=9)
bb_upper, bb_middle, bb_lower = data.indicator('BBANDS', 
    timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# KD隨機指標
k, d = data.indicator('STOCH', 
    fastk_period=9, slowk_period=3, slowd_period=3)

# pandas_ta 指標 (需安裝 pandas-ta)
supertrend = data.indicator('supertrend')

# 批量計算多個指標
indicators = {
    'rsi_14': data.indicator('RSI', timeperiod=14),
    'rsi_21': data.indicator('RSI', timeperiod=21),
    'sma_20': data.indicator('SMA', timeperiod=20),
    'ema_12': data.indicator('EMA', timeperiod=12),
}
```

### 技術指標特點
- **規模**: 支援 2000 支股票 × 10 年數據的指標計算
- **自動處理**: 自動處理 NaN 值
- **優先順序**: 優先使用 talib 計算，後備使用 pandas_ta
- **返回格式**: DataFrame，行為股票代號，列為日期

### 安裝要求
```bash
# Google Colab
!pip install ta-lib-bin

# 本地環境 (需要根據平台安裝)
pip install ta-lib
pip install pandas-ta  # 可選，支援更多指標
```

### 實戰應用
```python
# 多指標組合策略
close = data.get('price:收盤價')
rsi = data.indicator('RSI', timeperiod=14)
sma_20 = data.indicator('SMA', timeperiod=20)
sma_60 = data.indicator('SMA', timeperiod=60)

# 技術指標篩選條件
tech_filter = (
    (rsi < 30) &              # RSI 超賣
    (close > sma_20) &        # 價格突破20日線
    (sma_20 > sma_60)         # 短線均線向上
)

# 篩選符合條件的股票
selected_stocks = tech_filter.sum(axis=1).sort_values(ascending=False)
print(f"符合條件股票數: {(selected_stocks > 0).sum()}")
```

#### `data.universe()` - 股票池篩選
```python
# 獲取所有上市股票
all_stocks = data.universe()

# 獲取特定市場股票
tse_stocks = data.universe(market='TSE')    # 上市
otc_stocks = data.universe(market='OTC')    # 上櫃

# 獲取特定類別股票
tech_stocks = data.universe(category='電子工業')
finance_stocks = data.universe(category='金融保險')
```

#### `data.search()` - 數據集搜尋
```python
# 搜尋包含關鍵字的數據集
price_datasets = data.search('price')
financial_datasets = data.search('financial')
margin_datasets = data.search('融資')

# 顯示搜尋結果
for dataset in price_datasets:
    print(f"數據集: {dataset}")
```

#### `data.universe()` - 進階股票池篩選
```python
# 基本股票池
all_stocks = data.universe()                    # 所有股票
tse_stocks = data.universe(market='TSE')        # 上市
otc_stocks = data.universe(market='OTC')        # 上櫃
tse_otc_stocks = data.universe(market='TSE_OTC')# 上市上櫃
etf_stocks = data.universe(market='ETF')        # ETF

# 行業篩選 (30+ 行業類別)
cement_stocks = data.universe(category=['水泥工業'])
tech_stocks = data.universe(category=['電子工業'])
finance_stocks = data.universe(category=['金融保險'])

# 組合篩選 - 使用 context manager
with data.universe(market='TSE_OTC', category=['水泥工業']):
    cement_price = data.get('price:收盤價')
```

### 數據特點
- **覆蓋範圍**: 2000+ 支股票，10+ 年歷史數據
- **數據格式**: 返回 FinlabDataFrame (類似 Pandas DataFrame)
- **免費限制**: 免費用戶有歷史數據限制
- **登入方式**: 支援 GUI 登入、API token、環境變數

### 數據管理最佳實踐
```python
# 1. 登入設定
import os
os.environ['FINLAB_API_TOKEN'] = 'your_token'
finlab.login()

# 2. 高效數據獲取
close = data.get('price:收盤價', save_to_storage=True)  # 啟用本地緩存
volume = data.get('price:成交股數', force_download=False) # 避免重複下載

# 3. 批量數據處理
datasets = ['price:收盤價', 'price:成交股數', 'price:最高價', 'price:最低價']
data_dict = {name: data.get(name) for name in datasets}
```

---

## 回測系統 (finlab.backtest)

FinLab 提供**簡單而強大的一行式回測系統**，支援複雜策略的快速實現：

### 基本回測 - 簡單範例

```python
from finlab import data, backtest

# 超簡單策略：低於6元的股票
close = data.get('price:收盤價')
position = close < 6
report = backtest.sim(position, resample='M', name="低價股策略")

# 一行完成回測！
```

### 進階策略範例

#### 1. 新高突破策略
```python
# 選出創250日新高的股票
high = data.get('price:最高價')
new_high = high == high.rolling(250).max()
position = new_high.top(20)  # 取前20檔
backtest.sim(position, resample='W')
```

#### 2. RSI 策略
```python
# RSI 前20名策略
rsi = data.indicator('RSI', timeperiod=14)
position = rsi.top(20)
backtest.sim(position, resample='M')
```

#### 3. 基本面 + 技術面組合
```python
# 結合 ROE 和價格乖離
roe = data.get('fundamental_features:股東權益報酬率')
close = data.get('price:收盤價')
sma_20 = close.rolling(20).mean()

# 高ROE + 價格低於均線
fundamentals = roe.top(100)  # ROE前100名
technicals = close < sma_20 * 0.95  # 低於20日線5%
position = (fundamentals & technicals).top(30)

backtest.sim(position, resample='M')
```

### 策略模擬 `backtest.sim()` 完整參數

```python
position = backtest.sim(
    signals,                         # 交易信號 (DataFrame)
    resample='M',                   # 重新平衡頻率 ('D', 'W', 'M')
    position_limit=20,              # 最大持股數量
    trade_at_price='open',          # 交易價格 ('open', 'close', 'avg')
    fee_ratio=0.001425,             # 手續費率 (0.1425%)
    tax_ratio=0.003,                # 交易稅率 (0.3%)
    initial_capital=10_000_000,     # 初始資金
    market_price='close',           # 市價計算基準
    name="策略名稱"                  # 策略名稱
)
```

### 進階選股條件 - FinlabDataFrame 方法

```python
close = data.get('price:收盤價')
pb = data.get('price_earning_ratio:股價淨值比')

# 移動平均線
sma20 = close.average(20)
sma60 = close.average(60)

# 進出場條件
entries = close > sma20
exits = close < sma60

# hold_until 策略 - 持有直到條件滿足
position = entries.hold_until(
    exits, 
    nstocks_limit=10,    # 持股上限
    rank=-pb             # 依PB排序 (負號表示由小到大)
)

backtest.sim(position)
```

### 績效分析與報告

```python
# 生成完整績效報告
report = backtest.sim(position, resample='M')

# 顯示策略績效
report.display()

# 獲取交易記錄
trades = report.get_trades()
print(trades.head())

# 交易記錄包含欄位:
# - 進場/出場日期
# - 持有天數  
# - 部位大小
# - 報酬率
# - 最大回撤
# - 最大有利/不利偏移

# MAE/MFE 分析 (交易波動分析)
report.display_mae_mfe_analysis()
```

### 獨特功能

1. **自動索引對齊**: 不同頻率數據自動對齊
2. **流動性風險分析**: 內建流動性檢查
3. **雲端平台整合**: 支援策略分享
4. **一鍵回測**: 複雜策略僅需幾行代碼

### 視覺化選項

```python
# 基本績效圖
backtest.plot(position)

# 進階圖表設定
backtest.plot(
    position,
    benchmark='0050',        # 基準比較
    show_trades=True,       # 顯示交易點
    figsize=(15, 8),        # 圖表大小
    title="策略績效分析"     # 自定義標題
)
```

---

## 機器學習 (finlab.ml)

FinLab提供完整的機器學習框架，包括特徵工程、標籤生成、模型訓練和預測。

### 機器學習流程概述

**完整ML管線**:
1. 特徵工程 (Feature Engineering)
2. 標籤生成 (Label Generation) 
3. 數據準備與分割
4. 模型訓練與驗證
5. 預測與信號生成
6. 回測驗證

### 特徵工程 `finlab.ml.feature`

```python
from finlab import feature

# 組合多個因子特徵
financial_features = {
    'pe_ratio': data.get('price_earning_ratio:本益比'),
    'pb_ratio': data.get('price_book_ratio:股價淨值比'),
    'roe': data.get('fundamental_features:股東權益報酬率')
}

technical_features = {
    'rsi': data.indicator('RSI', timeperiod=14),
    'ma_ratio': data.get('price:收盤價') / data.indicator('SMA', timeperiod=20),
    'volatility': data.get('price:收盤價').rolling(20).std()
}

# 使用feature.combine整合特徵
features = feature.combine({
    **financial_features,
    **technical_features
}, resample='ME')  # 月度重新採樣
```

### 標籤生成 `finlab.ml.label`

```python
from finlab import labeling

# 價格變化標籤
close = data.get('price:收盤價')
labels = labeling.price_change(close, period=21)  # 21天後價格變化

# 分類標籤 (三分類：漲/持平/跌)
categorical_labels = labeling.price_change_bin(close, period=21, bins=3)

# 風險調整標籤
risk_adjusted_labels = labeling.risk_adjusted_return(close, period=21)

# 超額報酬標籤 (相對市場)
excess_labels = labeling.excess_over_mean(
    index=features.index,
    resample='ME'
)
```

### 模型訓練與預測

```python
from finlab.ml import MLEngine

# 創建ML引擎
ml_engine = MLEngine()

# 準備數據
ml_engine.prepare_data(
    features=features,
    labels=labels,
    start_date='2015-01-01',
    end_date='2023-12-31'
)

# 模型訓練
ml_engine.train(
    model_type='lightgbm',        # 支援: lightgbm, xgboost, random_forest
    cv_method='time_series',      # 時間序列交叉驗證
    test_size=0.2,
    params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
)

# 模型預測
predictions = ml_engine.predict(features_new)

# 生成交易信號
signals = ml_engine.generate_signals(
    predictions, 
    threshold=0.1  # 信號閾值
)
```

### AutoML自動機器學習

```python
from finlab.ml import AutoML

# 創建AutoML實例
automl = AutoML()

# 自動特徵工程和模型選擇
best_model = automl.fit(
    data=stock_data,
    target='future_return',
    time_budget=3600,  # 1小時時間限制
    metric='sharpe_ratio',
    cv_folds=5
)

# 獲取最佳特徵
best_features = automl.get_selected_features()
print(f"選擇的特徵數量: {len(best_features)}")

# 生成交易信號
signals = best_model.generate_signals(test_data)
```

### 因子分析 `finlab.ml.factor_analysis`

FinLab 提供完整的因子分析框架，用於系統性地研究和改進投資策略因子：

#### 因子特徵工程
```python
from finlab.ml import feature, label

# 組合多個因子特徵
features = feature.combine({
    'marketcap': marketcap_factor,    # 市值因子
    'revenue': revenue_factor,        # 營收因子  
    'momentum': momentum_factor       # 動能因子
}, resample='ME')  # 月度重新採樣

# 生成超額報酬標籤
labels = label.excess_over_mean(
    index=features.index, 
    resample='ME'
)
```

#### 因子績效分析
```python
# 因子報酬分析 - 衡量因子隨時間的表現
factor_returns = analyze_factor_returns(features, labels)

# 因子中心性分析 - 量化因子報酬的"共同性"
factor_centrality = calculate_factor_centrality(factor_returns)
```

#### 因子評估指標

**1. 資訊係數 (IC) 分析**
```python
# 計算因子與未來報酬的相關性
ic_analysis = calculate_ic(features, future_returns)
print(f"因子IC值: {ic_analysis['ic_mean']:.4f}")
print(f"IC標準差: {ic_analysis['ic_std']:.4f}")
print(f"IC t統計量: {ic_analysis['t_stat']:.2f}")
```

**2. Shapley值分析**
```python
# 使用Shapley值評估因子貢獻度
shapley_values = calculate_shapley_values(features, target_returns)
```

**3. 趨勢分析**
```python
# 趨勢分析統計指標
trend_analysis = analyze_factor_trends(factor_returns)
print(f"斜率: {trend_analysis['slope']:.4f}")
print(f"P值: {trend_analysis['p_value']:.4f}")
print(f"R平方: {trend_analysis['r_squared']:.4f}")
```

#### 因子中心性解讀

**高中心性因子特徵**:
- 近期表現良好
- 未來波動風險較高
- 可能面臨反轉修正

**低中心性因子特徵**:
- 近期表現不佳  
- 未來風險較低
- 突然修正的風險較小

#### 實戰應用
```python
def comprehensive_factor_analysis(factor_data, price_data):
    """
    完整的因子分析流程
    """
    # 1. 特徵工程
    features = feature.combine(factor_data, resample='ME')
    labels = label.excess_over_mean(index=features.index, resample='ME')
    
    # 2. IC分析
    ic_results = calculate_ic(features, labels)
    
    # 3. 因子中心性
    centrality_scores = calculate_factor_centrality(features)
    
    # 4. 績效統計
    performance_stats = {
        'ic_mean': ic_results['ic_mean'],
        'ic_std': ic_results['ic_std'],
        'sharpe_ratio': ic_results['ic_mean'] / ic_results['ic_std'],
        'centrality': centrality_scores.mean()
    }
    
    return performance_stats

# 執行完整分析
analysis_results = comprehensive_factor_analysis(
    factor_data={'momentum': momentum_signals, 'value': value_signals},
    price_data=close_prices
)
```

---

## 視覺化工具 (finlab.plot)

FinLab提供豐富的圖表工具，讓你更方便洞察市場數據和策略績效。

### 技術指標圖組

```python
from finlab.plot import plot_tw_stock_candles
from finlab.data import indicator

stock_id = '2330'  # 台積電
recent_days = 1000
adjust_price = False
resample = "D"      # D=日線, W=週線, M=月線

# 疊加技術指標
overlay_func = {
    'ema_5': indicator('EMA', timeperiod=5),
    'ema_10': indicator('EMA', timeperiod=10), 
    'ema_20': indicator('EMA', timeperiod=20),
    'ema_60': indicator('EMA', timeperiod=60),
}

# 副圖技術指標
k, d = indicator('STOCH')  # KD指標
rsi = indicator('RSI')     # RSI指標
technical_func = [{'K': k, 'D': d}, {'RSI': rsi}]

# 繪製K線圖與技術指標
plot_tw_stock_candles(
    stock_id, 
    recent_days, 
    adjust_price, 
    resample,
    overlay_func=overlay_func,      # 主圖疊加指標
    technical_func=technical_func   # 副圖技術指標
)
```

### 漲跌幅與成交金額板塊圖

```python
from finlab.plot import plot_tw_stock_treemap

# 巢狀樹狀圖顯示多維度資料，依產業分類顯示
plot_tw_stock_treemap(
    start='2021-07-01',
    end='2021-07-02',
    area_ind="turnover",        # market_value, turnover
    item="return_ratio"         # return_ratio, turnover_ratio
)
```

### 本益比與市值板塊圖

```python
# 本益比分布視覺化
plot_tw_stock_treemap(
    start='2021-07-01',
    end='2021-07-02',
    area_ind="market_value",              # 區域大小=市值
    item="price_earning_ratio:本益比",     # 顏色=本益比
    clip=(0, 50),                        # 數值範圍限制
    color_continuous_scale='RdBu_r'      # 顏色主題
)
```

### 財務指標雷達圖

```python
from finlab.plot import plot_tw_stock_radar

# 投資組合比較分析
portfolio = ['1101', '2330', '8942', '6263']

plot_tw_stock_radar(
    portfolio=portfolio,
    mode="bar_polar",           # line_polar, bar_polar, scatter_polar
    line_polar_fill=None       # toself, tonext, None
)

# 自定義財務指標雷達圖
custom_features = [
    'fundamental_features:營業毛利率', 
    'fundamental_features:營業利益率', 
    'fundamental_features:稅後淨利率',
    'fundamental_features:現金流量比率', 
    'fundamental_features:負債比率'
]

plot_tw_stock_radar(
    portfolio=["9939"], 
    feats=custom_features, 
    mode="line_polar", 
    cut_bins=8                 # 評分等級數
)
```

### 本益比河流圖

```python
from finlab.plot import plot_tw_stock_river

# PE或PB河流圖，判斷估值所處位階
plot_tw_stock_river(
    stock_id='2330', 
    start='2015-1-1', 
    end='2022-7-1', 
    mode='pe',          # pe=本益比, pb=股價淨值比
    split_range=10      # 區間分割數量
)
```

### 策略部位旭日圖

```python
from finlab.plot import StrategySunburst

# 多策略部位監控
strategies = StrategySunburst()
strategies.plot().show()
```

### 自定義圖表設定

```python
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 使用matplotlib客製化
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8')  # 圖表風格

# 使用plotly互動式圖表
fig = go.Figure()
fig.update_layout(
    title="台股策略績效分析",
    xaxis_title="時間",
    yaxis_title="累積報酬率",
    template="plotly_dark"  # 暗色主題
)
```

---

## 投資組合 (finlab.portfolio)

### 投資組合權重優化

```python
from finlab import portfolio

# 等權重組合
equal_weights = portfolio.equal_weight(stocks_list)

# 市值權重組合
market_cap_weights = portfolio.market_cap_weight(stocks_list)

# 風險平價組合
risk_parity_weights = portfolio.risk_parity(returns_data)

# 最大分散化組合
max_div_weights = portfolio.max_diversification(covariance_matrix)

# 最小變異數組合
min_var_weights = portfolio.min_variance(returns_data, covariance_matrix)
```

### 投資組合分析

```python
from finlab.portfolio import PortfolioAnalyzer

# 創建投資組合分析器
analyzer = PortfolioAnalyzer()

# 設定投資組合
portfolio_weights = {
    '2330': 0.3,  # 台積電 30%
    '2317': 0.2,  # 鴻海 20%
    '1101': 0.15, # 台泥 15%
    '2454': 0.15, # 聯發科 15%
    '2412': 0.2   # 中華電 20%
}

# 計算投資組合績效
performance = analyzer.calculate_performance(
    weights=portfolio_weights,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

print(f"年化報酬率: {performance['annual_return']:.2%}")
print(f"年化波動率: {performance['annual_volatility']:.2%}")
print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
print(f"最大回撤: {performance['max_drawdown']:.2%}")
```

### 投資組合風險分析

```python
# 計算VaR (Value at Risk)
var_95 = analyzer.calculate_var(
    portfolio_weights, 
    confidence_level=0.95,
    lookback_days=252
)

# 計算條件風險值 CVaR
cvar_95 = analyzer.calculate_cvar(
    portfolio_weights,
    confidence_level=0.95
)

# 風險貢獻分析
risk_contribution = analyzer.risk_contribution(
    portfolio_weights,
    returns_data
)

print(f"95% VaR: {var_95:.2%}")
print(f"95% CVaR: {cvar_95:.2%}")
```

---

## 即時交易 (finlab.online)

### 帳戶連接與管理

```python
from finlab.online import SinopacAccount, FugleAccount

# 永豐證券帳戶設定
sinopac_account = SinopacAccount(
    api_key='your_api_key',
    secret_key='your_secret_key',
    account='your_account',
    password='your_password'
)

# 富果證券帳戶設定  
fugle_account = FugleAccount(
    config_path='config.ini',  # 配置檔案路徑
    market='TW'               # 台股市場
)

# 檢查帳戶連線狀態
print(f"永豐帳戶狀態: {sinopac_account.is_connected()}")
print(f"富果帳戶狀態: {fugle_account.is_connected()}")
```

### 下單API

```python
# 市價買進訂單
market_buy_order = sinopac_account.order(
    action='Buy',
    code='2330',
    quantity=1000,
    order_type='Market'
)

# 限價賣出訂單
limit_sell_order = sinopac_account.order(
    action='Sell', 
    code='2330',
    quantity=1000,
    price=580.0,
    order_type='Limit'
)

# 停損訂單
stop_loss_order = sinopac_account.order(
    action='Sell',
    code='2330', 
    quantity=1000,
    price=550.0,
    order_type='StopLoss'
)

# 查詢訂單狀態
order_status = sinopac_account.query_order(market_buy_order.order_id)
print(f"訂單狀態: {order_status.status}")
print(f"成交數量: {order_status.filled_quantity}")
```

### 部位與資產管理

```python
# 查詢帳戶資訊
account_info = sinopac_account.get_account_info()
print(f"可用資金: {account_info.available_cash:,}")
print(f"總資產: {account_info.total_value:,}")

# 查詢持倉部位
positions = sinopac_account.get_positions()
for position in positions:
    print(f"{position.code}: {position.quantity} 股")
    print(f"成本價: {position.avg_price:.2f}")
    print(f"市值: {position.market_value:,}")
    print(f"損益: {position.pnl:+,.0f} ({position.pnl_ratio:+.1%})")

# 查詢交易記錄
trades = sinopac_account.get_trades(
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 績效分析
from finlab.online import PerformanceAnalyzer

performance = PerformanceAnalyzer(trades)
stats = performance.calculate_stats()

print(f"總報酬率: {stats['total_return']:.2%}")
print(f"勝率: {stats['win_rate']:.1%}")
print(f"平均持有天數: {stats['avg_holding_days']:.1f}")
```

### 策略自動執行

```python
from finlab.online import StrategyExecutor

# 創建策略執行器
executor = StrategyExecutor(
    account=sinopac_account,
    strategy_name="動能策略"
)

# 定義策略執行邏輯
@executor.schedule(time='09:05')  # 每日9:05執行
def momentum_strategy():
    """動能策略每日執行"""
    
    # 獲取信號
    close = data.get('price:收盤價')
    signals = generate_momentum_signals(close)
    
    # 獲取當前持倉
    current_positions = executor.get_current_positions()
    
    # 計算目標部位
    target_positions = calculate_target_positions(
        signals, 
        total_value=executor.get_account_value(),
        position_limit=20
    )
    
    # 執行調倉
    orders = executor.rebalance(
        current_positions=current_positions,
        target_positions=target_positions,
        rebalance_threshold=0.05  # 5%調倉閾值
    )
    
    # 記錄執行結果
    executor.log_execution(orders)

# 啟動自動執行
executor.start()
```

### 風險控制與監控

```python
from finlab.online import RiskManager

# 創建風險管理器
risk_manager = RiskManager(account=sinopac_account)

# 設定風險參數
risk_manager.set_risk_limits(
    max_position_size=0.1,      # 單一標的最大部位 10%
    max_sector_exposure=0.3,    # 單一產業最大曝險 30%
    max_daily_loss=0.02,        # 單日最大虧損 2%
    max_drawdown=0.15          # 最大回撤 15%
)

# 即時風險監控
@risk_manager.monitor(interval='1m')  # 每分鐘檢查
def risk_check():
    """即時風險檢查"""
    
    # 檢查部位集中度
    concentration_risk = risk_manager.check_concentration()
    
    # 檢查帳戶損失
    account_risk = risk_manager.check_account_risk()
    
    # 觸發風險警告
    if concentration_risk['violation']:
        risk_manager.send_alert("部位集中度超標")
    
    if account_risk['daily_loss'] > risk_manager.max_daily_loss:
        risk_manager.emergency_stop()  # 緊急停損

# 啟動風險監控
risk_manager.start_monitoring()
```

---

## 最佳實踐

### 數據管理
1. **使用本地緩存**: `data.get(save_to_storage=True)` 避免重複下載
2. **定期更新數據**: 設置自動更新機制
3. **數據完整性檢查**: 檢查數據缺失和異常值

### 策略開發
1. **回測時間範圍**: 使用足夠長的歷史數據 (建議>5年)
2. **交易成本考慮**: 包含手續費、交易稅、滑價成本
3. **流動性檢查**: 確保標的有足夠的成交量
4. **前瞻偏誤**: 避免使用未來資訊

### 風險管理
1. **部位大小控制**: 單一標的不超過組合5%
2. **行業分散**: 避免集中特定行業
3. **停損機制**: 設置合理的停損點
4. **壓力測試**: 模擬極端市場情況

### 程式碼品質
1. **模組化設計**: 將策略邏輯分離成獨立函數
2. **參數化設定**: 避免硬編碼參數
3. **錯誤處理**: 處理數據異常和API錯誤
4. **文件記錄**: 詳細記錄策略邏輯和參數設定

---

## 總結

本文檔基於 FinLab 官方文檔整理，提供完整的 API 參考和實戰範例。FinLab 是專為台股量化交易設計的 Python 套件，提供數據獲取、策略開發、回測分析和即時交易的完整解決方案。

**重要提醒**: 
- 本文檔僅基於官方 finlab 文檔，未包含專案特定實作
- 實際使用時請參考最新的官方文檔和API變更
- 投資有風險，策略回測結果不代表未來表現

**官方資源**:
- 官方文檔: https://doc.finlab.tw/
- API 參考: https://doc.finlab.tw/reference/finlab/
- 初學者指南: https://doc.finlab.tw/tools/guide_for_beginners/