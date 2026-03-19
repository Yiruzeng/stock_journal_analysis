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
from openai import OpenAI 

# ============================================
# 頁面設定與自定義 UI 樣式 (對齊 Dashboard 參考圖)
# ============================================

st.set_page_config(
    page_title="AI 個股投資日誌分析系統",
    page_icon="📊",
    layout="wide"
)

# 注入自定義 CSS 來改變介面顏色與按鈕風格
st.markdown("""
<style>
    /* 整體背景色 (非常淺的灰白，凸顯卡片) */
    .stApp {
        background-color: #F4F6F8;
    }
    
    /* 隱藏預設頂部裝飾條 */
    header {visibility: hidden;}

    /* 側邊欄背景改為純白 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EBEBEB;
    }

    /* 🌟 新增：文字輸入框 (API Key 等) 外框設計 */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        border: 1px solid #D1D5DB !important; /* 細線灰線 */
        border-radius: 8px !important; /* 微圓角 */
        background-color: #FFFFFF !important;
        transition: all 0.2s ease-in-out;
    }
    /* 輸入框點擊/聚焦時的狀態 */
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #DF644E !important; /* 聚焦時變成主色調橘紅色 */
        box-shadow: 0 0 0 1px rgba(223, 100, 78, 0.2) !important;
    }

    /* 主要按鈕 (Primary) - 對應圖中的橘紅色按鈕 */
    button[kind="primary"] {
        background-color: #DF644E !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important; /* 大圓角 */
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    button[kind="primary"]:hover {
        background-color: #C8543F !important;
        box-shadow: 0px 4px 10px rgba(223, 100, 78, 0.3);
    }

    /* 次要按鈕 (Secondary) - 白底、黑字、淺灰邊框 */
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #E2E2E2 !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    button[kind="secondary"]:hover {
        border-color: #DF644E !important;
        color: #DF644E !important;
    }

    /* 四大數據卡片 (Metrics) - 白色圓角卡片帶輕微陰影 */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #F0F0F0;
    }
    /* 數據卡片的標籤顏色微調 */
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-weight: 500;
    }

    /* 分頁標籤 (Tabs) 設計 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #FFFFFF;
        border-radius: 12px 12px 0 0;
        padding: 0 24px;
        border: 1px solid #EBEBEB;
        border-bottom: none;
        color: #666666;
    }
    /* 選中的分頁標籤 - 套用主色調 */
    .stTabs [aria-selected="true"] {
        background-color: #DF644E !important;
        color: white !important;
        border-color: #DF644E !important;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
        font-weight: bold !important;
    }

    /* 表格背景白底化 */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 工具函數區
# ============================================

def validate_trading_journal(df):
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

def calculate_fifo_performance(df, current_prices):
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
# API 呼叫區塊
# ============================================

@st.cache_data(ttl=300)
def get_stock_quote(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.fast_info['last_price']
    except Exception as e:
        return 0

@st.cache_data(ttl=3600)
def get_historical_price(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        df = ticker.history(start=start_str, end=end_str)
        
        if df is None or df.empty:
            return None
            
        df = df.reset_index()
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
            
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        return df.sort_values('date')
    except Exception as e:
        return None

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

# ============================================
# AI 分析相關函數
# ============================================

def prepare_ai_analysis_data(df, performance, fmp_data, news_data, symbol):
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

def call_ai_analysis(prompt, ai_engine, api_key):
    if ai_engine == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not models: return "❌ 您的 Google API Key 目前沒有權限使用任何文字生成模型。"
            chosen_model = next((m for m in models if "flash" in m.lower()), models[0]).replace("models/", "")
            model = genai.GenerativeModel(model_name=chosen_model)
            return model.generate_content("【系統設定：你是一位專業且富有同理心的投資交易分析師，擅長從交易記錄中發現模式並給予客觀的建設性建議。】\n\n" + prompt).text
        except Exception as e: return f"❌ Gemini 分析生成失敗: {str(e)}"
    elif ai_engine == "OpenAI (ChatGPT)":
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[{"role": "system", "content": "你是一位專業投資交易分析師。"}, {"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e: return f"❌ OpenAI 分析生成失敗: {str(e)}"

# ============================================
# 視覺化函數
# ============================================
def create_candlestick_chart(symbol, price_data, trades_data):
    # 將圖表的背景色也配合改為白色，並隱藏邊框，符合乾淨的卡片設計
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=(f'', '成交量'))
    
    fig.add_trace(go.Candlestick(x=price_data['date'], open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close'], name='K線', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    price_data['MA5'] = price_data['close'].rolling(window=5).mean()
    price_data['MA20'] = price_data['close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=price_data['date'], y=price_data['MA5'], mode='lines', name='MA5', line=dict(color='#F2B642', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data['date'], y=price_data['MA20'], mode='lines', name='MA20', line=dict(color='#4A90E2', width=1.5)), row=1, col=1)
    
    buy_trades = [t for t in trades_data if t['type'] == 'Buy']
    if buy_trades:
        fig.add_trace(go.Scatter(x=[t['date'] for t in buy_trades], y=[t['price'] for t in buy_trades], mode='markers+text', marker=dict(size=12, color='#26a69a', symbol='triangle-up'), text=['▲'] * len(buy_trades), textposition='top center', name='買入', hovertext=[f"買入 {t['quantity']}股<br>{t['reason'][:20]}" for t in buy_trades]), row=1, col=1)
    
    sell_trades = [t for t in trades_data if t['type'] == 'Sell']
    if sell_trades:
        fig.add_trace(go.Scatter(x=[t['date'] for t in sell_trades], y=[t['price'] for t in sell_trades], mode='markers+text', marker=dict(size=12, color='#ef5350', symbol='triangle-down'), text=['▼'] * len(sell_trades), textposition='bottom center', name='賣出', hovertext=[f"賣出 {t['quantity']}股<br>損益: ${t.get('pnl',0):.2f}" for t in sell_trades]), row=1, col=1)
    
    colors = ['#ef5350' if c < o else '#26a69a' for c, o in zip(price_data['close'], price_data['open'])]
    fig.add_trace(go.Bar(x=price_data['date'], y=price_data['volume'], name='成交量', marker_color=colors, showlegend=False), row=2, col=1)
    
    fig.update_layout(
        height=500, 
        xaxis_rangeslider_visible=False, 
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F0F0F0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#F0F0F0', zeroline=False)
    
    return fig

# ============================================
# 側邊欄與頁面狀態
# ============================================

st.sidebar.markdown("### 📊 AI 投資分析系統")

selected_ai = st.sidebar.radio("🧠 選擇 AI 引擎", ["Google Gemini", "OpenAI (ChatGPT)"])
if selected_ai == "Google Gemini":
    active_api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
else:
    active_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

fmp_api_key = st.sidebar.text_input("🏢 FMP API Key (美股選填)", type="password")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# --- 📥 新增：內建 CSV 範本下載按鈕 ---
template_csv = """Date,Type,Symbol,Name,Price,Quantity,Reason
2026-02-26,Buy,2330.TW,台積電,1800,100,日macd翻紅，拉回等待突破均線小部位卡位(test)
2026-03-15,Sell,2330.TW,台積電,1843,50,獲利減碼到口袋(test) 
2026-02-26,Buy,NVDA,NVDA,170,100,波段macd日線翻紅powersqueeze增強(test)
2026-03-18,Buy,4979.TWO,華星光,373,1000,波段macd日線翻紅powersqueeze增強(test)"""

st.sidebar.download_button(
    label="📥 下載 CSV 日誌範本",
    data=template_csv.encode('utf-8-sig'), # 確保 Excel 開啟中文不會亂碼
    file_name="stock_journal_template.csv",
    mime="text/csv",
    help="點擊下載標準格式的 CSV 日誌",
    use_container_width=True
)

st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #EBEBEB;'>", unsafe_allow_html=True)

# 原本的上傳區塊
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
            st.sidebar.success("✅ 驗證通過")
        else:
            st.sidebar.error("❌ 驗證失敗")
            for error in errors: st.sidebar.error(error)
    except Exception as e:
        st.sidebar.error(f"❌ 讀取失敗: {str(e)}")

if csv_valid:
    min_date = df_validated['Date'].min().date()
    max_date = df_validated['Date'].max().date()
    date_range = st.sidebar.date_input("選擇區間", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY/MM/DD")
else:
    date_range = st.sidebar.date_input("選擇區間", disabled=True)

enable_ai = st.sidebar.checkbox("啟用 AI 深度分析", value=True, disabled=not csv_valid)

if 'start_analysis' not in st.session_state:
    st.session_state.start_analysis = False

can_analyze = csv_valid and (active_api_key if enable_ai else True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

if st.sidebar.button("開始分析", type="primary", disabled=not can_analyze, use_container_width=True):
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith('ai_report_')]
    for k in keys_to_remove: del st.session_state[k]
    st.session_state.start_analysis = True

# ============================================
# 主畫面分析流程
# ============================================

st.markdown("<h2>📊 Stock Journal Anaysis Dashboard</h2>", unsafe_allow_html=True)

if uploaded_file is None:
    # 利用三個欄位 (1:2:1 比例) 來讓圖片完美置中顯示，不會過度放大
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            # 讀取剛剛存好的圖片
            st.image("landing_icon.png", use_container_width=True)
        except FileNotFoundError:
            # 防呆機制：如果忘記放圖片，會顯示灰色佔位區塊
            st.markdown("""
            <div style='text-align: center; padding: 60px; background-color: #EBEBEB; border-radius: 16px; color: #888;'>
                圖示區塊<br><small>(請將圖片命名為 landing_icon.png 並放於同資料夾)</small>
            </div>
            """, unsafe_allow_html=True)
    
    # 加入與參考圖一致的歡迎標語，並置中對齊
    st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 30px;">
        <h2 style="color: #333333; font-weight: bold;">Welcome to investment analysis.</h2>
        <p style="color: #666666; font-size: 16px;">
            為您的努力變得更強!<br>
            結合 AI 深度覆盤、投資追蹤紀錄與數據分析檢討。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 保留原本的提示框，但稍微修改語氣更符合整體質感
    st.info("💡 準備就緒：請從左側上傳您的投資日誌 CSV 檔案以啟動儀表板。(台股代碼請加上 .TW 或 .TWO)")

elif st.session_state.get('start_analysis', False) and csv_valid:
    df = df_validated.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
        
    if df.empty:
        st.warning("⚠️ 在選擇的時間區間內沒有任何交易紀錄。")
        st.stop()
        
    symbols = df['Symbol'].unique().tolist()
    
    with st.spinner("📡 同步最新股價資料中..."):
        current_prices = {}
        for sym in symbols:
            price = get_stock_quote(sym)
            if price: current_prices[sym] = price
    
    performance = calculate_fifo_performance(df, current_prices)
    
    # --- 儀表板 (四大指標) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 總投入金額", f"${performance['total_investment']:,.2f}")
    col2.metric("📊 當前市值", f"${performance['total_market_value']:,.2f}")
    col3.metric("📈 總報酬率", f"{performance['total_return_pct']:.2f}%")
    col4.metric("✅ 已實現損益", f"${performance['total_realized_pnl']:,.2f}")
    
    # --- 總體當前持倉狀況表格 ---
    st.markdown("<h4 style='margin-top:20px; color:#333;'>📋 當前庫存部位</h4>", unsafe_allow_html=True)
    if performance['current_holdings']:
        holdings_df = pd.DataFrame(performance['current_holdings'])
        holdings_df.columns = ['代碼', '名稱', '持有數量', '平均成本', '當前價格', '總成本', '當前市值', '未實現損益', '報酬率%']
        st.dataframe(holdings_df, use_container_width=True)
    else:
        st.info("目前無持倉部位。")

    # --- 建立個股分頁 ---
    st.markdown("<h4 style='margin-top:30px; color:#333;'>🔍 個股深度覆盤</h4>", unsafe_allow_html=True)
    inventory_symbols = [h['symbol'] for h in performance['current_holdings']]
    
    tab_titles = []
    stock_names = {}
    for sym in symbols:
        name = df[df['Symbol'] == sym]['Name'].iloc[0]
        stock_names[sym] = name
        if sym in inventory_symbols:
            tab_titles.append(f"🟢 {sym} {name}")
        else:
            tab_titles.append(f"⚪ {sym} {name}")
            
    tabs = st.tabs(tab_titles)
    
    for i, symbol in enumerate(symbols):
        with tabs[i]:
            stock_name = stock_names[symbol]
            symbol_trade_history = performance['trade_history'].get(symbol, [])
            
            # --- 1. 先顯示純粹的進出場明細表 ---
            st.markdown("<h5 style='margin-top:15px; color:#555;'>📝 進出場詳細紀錄</h5>", unsafe_allow_html=True)
            if symbol_trade_history:
                history_df = pd.DataFrame(symbol_trade_history)
                display_df = pd.DataFrame()
                display_df['日期'] = pd.to_datetime(history_df['date']).dt.strftime('%Y-%m-%d')
                display_df['動作'] = history_df['type'].map({'Buy': '買入', 'Sell': '賣出'})
                display_df['價格'] = history_df['price'].apply(lambda x: f"${x:.2f}")
                display_df['數量'] = history_df['quantity']
                if 'pnl' in history_df.columns:
                    display_df['已實現損益'] = history_df['pnl'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
                else:
                    display_df['已實現損益'] = "-"
                display_df['交易理由'] = history_df['reason']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("尚無交易明細紀錄")
            
            # --- 2. 直接抓取並顯示歷史 K 線與進出場圖表 ---
            st.markdown(f"<h5 style='margin-top:15px; color:#555;'>📈 {symbol} {stock_name} 歷史 K 線與交易點位</h5>", unsafe_allow_html=True)
            symbol_trades = df[df['Symbol'] == symbol]
            start_date_hist = symbol_trades['Date'].min() - timedelta(days=90)  
            end_date_hist = pd.Timestamp.now()
            
            historical = get_historical_price(symbol, start_date_hist, end_date_hist)
            
            if historical is not None and not historical.empty:
                fig = create_candlestick_chart(symbol, historical, symbol_trade_history)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 無法抓取歷史 K 線資料，請確認股票代碼或網路狀態。")
            
            # --- 3. 延遲的 AI 核心運算：按鈕觸發 ---
            if enable_ai:
                report_key = f"ai_report_{symbol}_{selected_ai}"
                st.markdown(f"<h5 style='margin-top:15px; color:#555;'>🤖 {selected_ai} AI 深度檢討</h5>", unsafe_allow_html=True)
                
                # 如果還沒生成報告，顯示按鈕
                if report_key not in st.session_state:
                    if st.button(f"✨ 執行 {symbol} 覆盤分析", key=f"btn_ai_{symbol}", type="primary"):
                        if not active_api_key:
                            st.error("⚠️ 請先在左側欄位輸入對應的 API Key！")
                        else:
                            with st.spinner(f"🧠 分析中..."):
                                profile = get_fmp_profile(symbol, fmp_api_key) if fmp_api_key else {}
                                metrics = get_fmp_key_metrics(symbol, fmp_api_key) if fmp_api_key else []
                                stock_data = {'historical': historical, 'profile': profile, 'key_metrics': metrics}
                                
                                analysis_data = prepare_ai_analysis_data(df, performance, stock_data, None, symbol)
                                prompt = generate_ai_analysis_prompt(analysis_data)
                                ai_report = call_ai_analysis(prompt, selected_ai, active_api_key)
                                st.session_state[report_key] = ai_report
                                
                                st.rerun()
                else:
                    # 直接顯示暫存的報告結果，並包裝在一個乾淨的白色區塊中
                    st.markdown(f"<div style='background-color:#FFF; padding:20px; border-radius:12px; border:1px solid #EBEBEB;'>{st.session_state[report_key]}</div>", unsafe_allow_html=True)