# ============================================
# AI 投資日誌分析系統 (支援台股、雙 AI 引擎版)
# ============================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import timedelta
import yfinance as yf
import google.generativeai as genai
from openai import OpenAI # 新增 OpenAI 的載入

# ============================================
# 頁面設定
# ============================================

st.set_page_config(
    page_title="AI 投資日誌分析系統",
    page_icon="📊",
    layout="wide"
)

# ============================================
# 工具函數區
# ============================================

# CSV 驗證函數
def validate_trading_journal(df):
    """
    驗證投資日誌格式
    返回：(is_valid, error_messages, warnings)
    """
    errors = []
    warnings = []
    
    required_columns = ['Date', 'Type', 'Symbol', 'Name', 'Price', 'Quantity', 'Reason']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"❌ 缺少必要欄位: {', '.join(missing_cols)}")
        return False, errors, warnings
    
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        if df['Date'].isna().any():
            errors.append("❌ Date 欄位格式錯誤，應為 YYYY-MM-DD")
    except:
        errors.append("❌ Date 欄位格式錯誤，應為 YYYY-MM-DD")
    
    invalid_types = df[~df['Type'].isin(['Buy', 'Sell'])]
    if not invalid_types.empty:
        errors.append(f"❌ Type 欄位只能是 'Buy' 或 'Sell'")
    
    try:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        
        if df['Price'].isna().any() or df['Quantity'].isna().any():
            errors.append("❌ Price 和 Quantity 必須是有效數字")
        elif (df['Price'] <= 0).any():
            errors.append("❌ Price 必須大於 0")
        elif (df['Quantity'] <= 0).any():
            errors.append("❌ Quantity 必須大於 0")
    except:
        errors.append("❌ Price 和 Quantity 必須是數字")
    
    if df['Symbol'].str.contains(r'[^A-Z0-9\.\-]', regex=True, na=False).any():
        warnings.append("⚠️ 股票代碼包含特殊字元，請確認是否正確")
    
    if not errors:
        future_dates = df[df['Date'] > pd.Timestamp.now()]
        if not future_dates.empty:
            warnings.append(f"⚠️ 發現 {len(future_dates)} 筆未來日期的交易")
    
    if not errors:
        inventory = {}
        for idx, row in df.sort_values('Date').iterrows():
            symbol = row['Symbol']
            if symbol not in inventory:
                inventory[symbol] = 0
            
            if row['Type'] == 'Buy':
                inventory[symbol] += row['Quantity']
            else:  
                inventory[symbol] -= row['Quantity']
                if inventory[symbol] < -0.001:  
                    warnings.append(f"⚠️ {symbol} 在 {row['Date'].strftime('%Y-%m-%d')} 賣出數量超過持有數量")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

# FIFO 績效計算函數
def calculate_fifo_performance(df, current_prices):
    """
    使用 FIFO 規則計算投資績效
    """
    df = df.sort_values('Date').copy()
    holdings = {}  
    trade_history = {}
    realized_pnl = {}
    
    for idx, row in df.iterrows():
        symbol = row['Symbol']
        
        if symbol not in holdings:
            holdings[symbol] = []
            trade_history[symbol] = []
            realized_pnl[symbol] = 0
        
        if row['Type'] == 'Buy':
            holdings[symbol].append({'date': row['Date'], 'price': row['Price'], 'quantity': row['Quantity']})
            trade_history[symbol].append({'date': row['Date'], 'type': 'Buy', 'price': row['Price'], 'quantity': row['Quantity'], 'reason': row['Reason']})
        
        else:  
            sell_quantity = row['Quantity']
            sell_price = row['Price']
            trade_pnl = 0
            weighted_cost = 0
            total_sold = 0
            
            while sell_quantity > 0 and holdings[symbol]:
                oldest_buy = holdings[symbol][0]
                
                if oldest_buy['quantity'] <= sell_quantity:
                    pnl = (sell_price - oldest_buy['price']) * oldest_buy['quantity']
                    trade_pnl += pnl
                    weighted_cost += oldest_buy['price'] * oldest_buy['quantity']
                    total_sold += oldest_buy['quantity']
                    sell_quantity -= oldest_buy['quantity']
                    holdings[symbol].pop(0)
                else:
                    pnl = (sell_price - oldest_buy['price']) * sell_quantity
                    trade_pnl += pnl
                    weighted_cost += oldest_buy['price'] * sell_quantity
                    total_sold += sell_quantity
                    oldest_buy['quantity'] -= sell_quantity
                    sell_quantity = 0
            
            avg_cost = weighted_cost / total_sold if total_sold > 0 else 0
            return_pct = ((sell_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
            realized_pnl[symbol] += trade_pnl
            
            trade_history[symbol].append({
                'date': row['Date'], 'type': 'Sell', 'price': sell_price, 'quantity': row['Quantity'],
                'avg_cost': avg_cost, 'pnl': trade_pnl, 'return_pct': return_pct, 'reason': row['Reason']
            })
    
    current_holdings = []
    total_cost = 0
    total_market_value = 0
    total_realized_pnl = sum(realized_pnl.values())
    
    for symbol, holding_list in holdings.items():
        if holding_list:
            total_quantity = sum(h['quantity'] for h in holding_list)
            total_cost_basis = sum(h['price'] * h['quantity'] for h in holding_list)
            avg_cost = total_cost_basis / total_quantity if total_quantity > 0 else 0
            
            current_price = current_prices.get(symbol, 0)
            market_value = current_price * total_quantity
            unrealized_pnl = market_value - total_cost_basis
            unrealized_pnl_pct = (unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
            
            current_holdings.append({
                'symbol': symbol, 'name': df[df['Symbol'] == symbol]['Name'].iloc[0],
                'quantity': total_quantity, 'avg_cost': avg_cost, 'current_price': current_price,
                'cost_basis': total_cost_basis, 'market_value': market_value,
                'unrealized_pnl': unrealized_pnl, 'unrealized_pnl_pct': unrealized_pnl_pct
            })
            
            total_cost += total_cost_basis
            total_market_value += market_value
    
    all_buys = df[df['Type'] == 'Buy']
    total_investment = (all_buys['Price'] * all_buys['Quantity']).sum()
    total_unrealized_pnl = total_market_value - total_cost
    total_pnl = total_realized_pnl + total_unrealized_pnl
    total_return_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    
    return {
        'current_holdings': current_holdings, 'trade_history': trade_history, 'realized_pnl': realized_pnl,
        'total_realized_pnl': total_realized_pnl, 'total_unrealized_pnl': total_unrealized_pnl,
        'total_investment': total_investment, 'total_cost': total_cost, 'total_market_value': total_market_value,
        'total_pnl': total_pnl, 'total_return_pct': total_return_pct
    }

# ============================================
# API 呼叫區塊 (報價改為 yfinance)
# ============================================

@st.cache_data(ttl=300)
def get_stock_quote(symbol):
    """獲取當前報價 (改用 yfinance 支援台股)"""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.fast_info['last_price']
    except Exception as e:
        st.warning(f"⚠️ 無法獲取 {symbol} 當前報價: {e}")
        return 0

@st.cache_data(ttl=3600)
def get_historical_price(symbol, start_date, end_date):
    """獲取歷史股價 (改用 yfinance 支援台股)"""
    try:
        ticker = yf.Ticker(symbol)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        df = ticker.history(start=start_str, end=end_str)
        
        if df is None or df.empty:
            return None
            
        df = df.reset_index()
        # 移除時區，避免繪圖錯誤
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
            
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        return df.sort_values('date')
    except Exception as e:
        st.error(f"❌ 獲取 {symbol} 歷史股價失敗: {str(e)}")
        return None

# FMP 基本面資料 (僅供美股使用，可選)
@st.cache_data(ttl=3600)
def get_fmp_profile(symbol, api_key):
    if not api_key: return {}
    try:
        res = requests.get(f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={api_key}", timeout=5)
        return res.json()[0] if res.json() else {}
    except: return {}

@st.cache_data(ttl=3600)
def get_fmp_key_metrics(symbol, api_key):
    if not api_key: return []
    try:
        res = requests.get(f"https://financialmodelingprep.com/stable/key-metrics?symbol={symbol}&limit=4&apikey={api_key}", timeout=5)
        return res.json() if res.json() else []
    except: return []

@st.cache_data(ttl=3600)
def get_fmp_ratios(symbol, api_key):
    if not api_key: return []
    try:
        res = requests.get(f"https://financialmodelingprep.com/stable/ratios?symbol={symbol}&limit=4&apikey={api_key}", timeout=5)
        return res.json() if res.json() else []
    except: return []

# ============================================
# AI 分析相關函數 (雙引擎版)
# ============================================

def prepare_ai_analysis_data(df, performance, fmp_data, news_data, symbol):
    """準備 AI 分析資料"""
    symbol_trades = df[df['Symbol'] == symbol].sort_values('Date').to_dict('records')
    trade_hist = performance['trade_history'].get(symbol, [])
    
    completed_trades = [t for t in trade_hist if t['type'] == 'Sell']
    win_rate = (sum(1 for t in completed_trades if t.get('pnl', 0) > 0) / len(completed_trades) * 100) if completed_trades else 0
    
    avg_holding_days = 0
    if len(completed_trades) > 0:
        first_buy = df[(df['Symbol'] == symbol) & (df['Type'] == 'Buy')]['Date'].min()
        last_sell = df[(df['Symbol'] == symbol) & (df['Type'] == 'Sell')]['Date'].max()
        if pd.notna(first_buy) and pd.notna(last_sell):
            avg_holding_days = (last_sell - first_buy).days / len(completed_trades)
    
    current_holding = next((h for h in performance['current_holdings'] if h['symbol'] == symbol), None)
    
    perf_data = {
        'total_trades': len(symbol_trades),
        'buy_count': sum(1 for t in trade_hist if t['type'] == 'Buy'),
        'sell_count': len(completed_trades),
        'realized_pnl': performance['realized_pnl'].get(symbol, 0),
        'win_rate': win_rate,
        'avg_holding_days': int(avg_holding_days),
        'current_holding': current_holding
    }
    
    price_data = fmp_data.get('historical')
    price_context = {}
    if price_data is not None and not price_data.empty:
        price_context = {
            'first_price': price_data['close'].iloc[0], 'last_price': price_data['close'].iloc[-1],
            'period_high': price_data['high'].max(), 'period_low': price_data['low'].min(),
            'overall_change_pct': ((price_data['close'].iloc[-1] - price_data['close'].iloc[0]) / price_data['close'].iloc[0] * 100)
        }
    
    profile = fmp_data.get('profile', {})
    company_info = {
        'name': profile.get('companyName', symbol), 'sector': profile.get('sector', 'N/A'),
        'industry': profile.get('industry', 'N/A'), 'market_cap': profile.get('marketCap', 0)
    }
    
    metrics = fmp_data.get('key_metrics', [])
    key_metrics_summary = {}
    if metrics and isinstance(metrics[0] if isinstance(metrics, list) else metrics, dict):
        latest = metrics[0] if isinstance(metrics, list) else metrics
        key_metrics_summary = {
            'returnOnEquity': latest.get('returnOnEquity', 0), 'returnOnAssets': latest.get('returnOnAssets', 0),
            'freeCashFlowYield': latest.get('freeCashFlowYield', 0), 'currentRatio': latest.get('currentRatio', 0)
        }
    
    return {'symbol': symbol, 'trades': symbol_trades, 'performance': perf_data, 'price_context': price_context, 'company_info': company_info, 'key_metrics': key_metrics_summary}

def generate_ai_analysis_prompt(analysis_data):
    """生成 AI Prompt"""
    symbol = analysis_data['symbol']
    trades = analysis_data['trades']
    perf = analysis_data['performance']
    price_ctx = analysis_data['price_context']
    
    trades_text = ""
    for i, trade in enumerate(trades, 1):
        trades_text += f"\n交易 {i}: [{trade['Type']}] 日期:{trade['Date'].strftime('%Y-%m-%d')} | 價格:${trade['Price']:.2f} | 數量:{trade['Quantity']} | 理由:{trade['Reason']}"
    
    perf_text = f"總交易:{perf['total_trades']} | 勝率:{perf['win_rate']:.1f}% | 已實現損益:${perf['realized_pnl']:,.2f} | 均持有:{perf['avg_holding_days']}天"
    
    prompt = f"""
請根據以下使用者的 {symbol} 交易日誌，進行多維度客觀評估，給予具有建設性的反饋。

## 交易記錄與績效
{trades_text}
\n統計: {perf_text}

## 股價背景 (交易期間)
期初價: ${price_ctx.get('first_price', 0):.2f} | 期末價: ${price_ctx.get('last_price', 0):.2f} | 漲跌幅: {price_ctx.get('overall_change_pct', 0):.2f}%

---
請用 Markdown 格式輸出完整分析報告，包含以下章節：
# {symbol} 投資分析報告
## 一、交易執行評估 (買賣點合理性、持倉管理)
## 二、決策品質評估 (進場理由邏輯、策略一致性、風險控制)
## 三、心理因素觀察 (情緒化交易跡象、紀律性)
## 四、綜合建議 (✅優勢清單 與 📈改進方向)
"""
    return prompt

# Gemini API 呼叫函數
def call_gemini_api(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        if not available_models:
            return "❌ 您的 Google API Key 目前沒有權限使用任何文字生成模型。"

        chosen_model = available_models[0]
        for m in available_models:
            if "flash" in m.lower():
                chosen_model = m
                break
            elif "pro" in m.lower():
                chosen_model = m

        clean_model_name = chosen_model.replace("models/", "")
        print(f"👉 使用 Google 模型：{clean_model_name}")

        system_role = "【系統設定：你是一位專業且富有同理心的投資交易分析師，擅長從交易記錄中發現模式並給予客觀、一針見血的建設性建議。】\n\n"
        full_prompt = system_role + prompt

        model = genai.GenerativeModel(model_name=clean_model_name)
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"❌ Google Gemini 分析生成失敗: {str(e)}\n\n請檢查 API Key 是否正確。"

# OpenAI API 呼叫函數
def call_openai_api(prompt, api_key):
    try:
        print("👉 使用 OpenAI 模型：gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 使用目前 CP 值最高且穩定的模型
            messages=[
                {"role": "system", "content": "你是一位專業且富有同理心的投資交易分析師，擅長從交易記錄中發現模式並給予客觀、一針見血的建設性建議。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI 分析生成失敗: {str(e)}\n\n請檢查 API Key 是否正確，或確認帳號是否有額度。"

# 統整呼叫入口
def call_ai_analysis(prompt, ai_engine, api_key):
    if ai_engine == "Google Gemini":
        return call_gemini_api(prompt, api_key)
    elif ai_engine == "OpenAI (ChatGPT)":
        return call_openai_api(prompt, api_key)
    else:
        return "❌ 未知的 AI 引擎設定"

# ============================================
# 視覺化函數
# ============================================
def create_candlestick_chart(symbol, price_data, trades_data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=(f'{symbol} 股價走勢', '成交量'))
    
    fig.add_trace(go.Candlestick(x=price_data['date'], open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close'], name='K線', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    price_data['MA5'] = price_data['close'].rolling(window=5).mean()
    price_data['MA20'] = price_data['close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=price_data['date'], y=price_data['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data['date'], y=price_data['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)), row=1, col=1)
    
    buy_trades = [t for t in trades_data if t['type'] == 'Buy']
    if buy_trades:
        fig.add_trace(go.Scatter(x=[t['date'] for t in buy_trades], y=[t['price'] for t in buy_trades], mode='markers+text', marker=dict(size=12, color='green', symbol='triangle-up'), text=['▲'] * len(buy_trades), textposition='top center', name='買入', hovertext=[f"買入 {t['quantity']}股<br>{t['reason'][:20]}" for t in buy_trades]), row=1, col=1)
    
    sell_trades = [t for t in trades_data if t['type'] == 'Sell']
    if sell_trades:
        fig.add_trace(go.Scatter(x=[t['date'] for t in sell_trades], y=[t['price'] for t in sell_trades], mode='markers+text', marker=dict(size=12, color='red', symbol='triangle-down'), text=['▼'] * len(sell_trades), textposition='bottom center', name='賣出', hovertext=[f"賣出 {t['quantity']}股<br>損益: ${t.get('pnl',0):.2f}" for t in sell_trades]), row=1, col=1)
    
    colors = ['red' if c < o else 'green' for c, o in zip(price_data['close'], price_data['open'])]
    fig.add_trace(go.Bar(x=price_data['date'], y=price_data['volume'], name='成交量', marker_color=colors, showlegend=False), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, hovermode='x unified')
    return fig

# ============================================
# 側邊欄介面
# ============================================

st.sidebar.header("📊 AI 投資日誌分析系統", divider="rainbow")

# 選擇 AI 引擎
selected_ai = st.sidebar.radio(
    "🧠 選擇 AI 分析引擎",
    ["Google Gemini", "OpenAI (ChatGPT)"],
    help="選擇你想用哪一家 AI 模型來產生分析報告"
)

# 根據選擇顯示對應的 API Key 輸入框
if selected_ai == "Google Gemini":
    active_api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password", help="從 https://aistudio.google.com/ 取得免費 Key")
else:
    active_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password", help="從 https://platform.openai.com/ 取得")

st.sidebar.divider()
fmp_api_key = st.sidebar.text_input("🏢 FMP API Key (選填)", type="password", help="僅用於美股公司基本面數據。台股/不看基本面可不填")

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("📤 上傳投資日誌 CSV", type=['csv'])

csv_valid = False
df_validated = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        is_valid, errors, warnings = validate_trading_journal(df)

        if is_valid:
            csv_valid = True
            df_validated = df
            # 清除舊報告緩存
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith('ai_report_')]
            for k in keys_to_remove: del st.session_state[k]

            st.sidebar.success("✅ CSV 格式驗證通過")
            st.sidebar.info(f"📊 總交易: {len(df)} 筆\n涉及股票: {', '.join(df['Symbol'].unique())}")
        else:
            st.sidebar.error("❌ CSV 格式驗證失敗")
            for error in errors: st.sidebar.error(error)
    except Exception as e:
        st.sidebar.error(f"❌ 讀取 CSV 失敗: {str(e)}")

st.sidebar.divider()
enable_ai = st.sidebar.checkbox("啟用 AI 深度分析", value=True, disabled=not csv_valid)

# 分析按鈕邏輯：如果有勾選 AI，則必須有填入對應的 API Key
can_analyze = csv_valid and (active_api_key if enable_ai else True)
analyze_button = st.sidebar.button("🚀 開始分析", type="primary", disabled=not can_analyze, use_container_width=True)

# ============================================
# 主畫面分析流程
# ============================================

st.header("📊 AI 投資日誌分析系統", divider="rainbow")

if uploaded_file is None:
    st.markdown("""
    ### 歡迎使用！請準備好您的投資日記 CSV
    **小提醒（台股專用）：** 若要分析台股，請在 CSV 的股票代碼後方加上 `.TW`（上市）或 `.TWO`（上櫃）。例如：`2330.TW`。
    
    請在左側選擇喜歡的 AI 引擎並貼上 API Key，上傳檔案即可開始。
    """)

elif analyze_button and csv_valid:
    df = df_validated.copy()
    symbols = df['Symbol'].unique().tolist()
    
    # --- 1. 抓取當前報價 ---
    st.subheader("📡 正在獲取股票資料...")
    current_prices = {}
    for sym in symbols:
        price = get_stock_quote(sym)
        if price: current_prices[sym] = price
    
    if not current_prices: st.stop()
    
    # --- 2. 計算績效 ---
    performance = calculate_fifo_performance(df, current_prices)
    
    # --- 3. 儀表板 ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 總投入金額", f"${performance['total_investment']:,.2f}")
    col2.metric("📊 當前市值", f"${performance['total_market_value']:,.2f}")
    col3.metric("📈 總報酬率", f"{performance['total_return_pct']:.2f}%")
    col4.metric("✅ 已實現損益", f"${performance['total_realized_pnl']:,.2f}")
    
    st.header("📋 當前持倉狀況", divider="rainbow")
    if performance['current_holdings']:
        holdings_df = pd.DataFrame(performance['current_holdings'])
        holdings_df.columns = ['代碼', '名稱', '持有數量', '平均成本', '當前價格', '總成本', '當前市值', '未實現損益', '報酬率%']
        st.dataframe(holdings_df, use_container_width=True)
        print("✅ 表格渲染成功，準備抓取歷史 K 線資料...")
        
    # --- 4. 股票細節與 AI 分析 ---
    st.header("📈 股價走勢與深度分析", divider="rainbow")
    
    all_stock_data = {}
    for symbol in symbols:
        symbol_trades = df[df['Symbol'] == symbol]
        start_date = symbol_trades['Date'].min() - timedelta(days=90)  
        end_date = pd.Timestamp.now()
        
        historical = get_historical_price(symbol, start_date, end_date)
        profile = get_fmp_profile(symbol, fmp_api_key) if fmp_api_key else {}
        metrics = get_fmp_key_metrics(symbol, fmp_api_key) if fmp_api_key else []
        
        all_stock_data[symbol] = {'historical': historical, 'profile': profile, 'key_metrics': metrics}
        print(f"📡 正在向 Yahoo Finance 請求 {symbol} 的歷史股價...")
    
    for symbol in symbols:
        st.subheader(f"📊 {symbol} 詳細分析", divider="rainbow")
        stock_data = all_stock_data[symbol]
        historical = stock_data['historical']
        
        if historical is not None and not historical.empty:
            symbol_trade_history = performance['trade_history'].get(symbol, [])
            fig = create_candlestick_chart(symbol, historical, symbol_trade_history)
            st.plotly_chart(fig, use_container_width=True)
            
            # AI 分析區塊
            if enable_ai and active_api_key:
                st.subheader(f"🤖 {selected_ai} 深度分析報告")
                
                # 若已產出報告，或切換 AI 引擎時，會重新產生
                report_key = f"ai_report_{symbol}_{selected_ai}"
                
                if report_key not in st.session_state:
                    with st.spinner(f"🧠 正在使用 {selected_ai} 分析 {symbol} 的交易記錄..."):
                        analysis_data = prepare_ai_analysis_data(df, performance, stock_data, None, symbol)
                        prompt = generate_ai_analysis_prompt(analysis_data)
                        ai_report = call_ai_analysis(prompt, selected_ai, active_api_key)
                        st.session_state[report_key] = ai_report
                
                st.markdown(st.session_state[report_key])
                st.divider()