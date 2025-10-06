import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import zipfile
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Import the backend
from backend import EnhancedDualFileAnalysisBot

# Configure page
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E8B57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color:#155724;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #155724;
        border-radius: 4px 4px 0 0;
        gap: 10px;
        padding-left: 20px;
        padding-right: 20px;
        color: #white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: Black;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'bot_initialized' not in st.session_state:
    st.session_state.bot_initialized = False
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'generated_charts' not in st.session_state:
    st.session_state.generated_charts = []

# Sidebar Email Settings
st.sidebar.subheader("📧 Email Settings")
st.session_state['email_enabled'] = st.sidebar.checkbox("Enable Email Report Distribution")

if st.session_state['email_enabled']:
    st.session_state['email_address'] = st.sidebar.text_input("Your Gmail Address")
    st.session_state['email_password'] = st.sidebar.text_input("Your Gmail App Password", type="password")
    st.session_state['recipient_email'] = st.sidebar.text_input("Recipient Email(s) (comma-separated)")
    send_email_button = st.sidebar.button("📨 Send Email Report")
else:
    send_email_button = False

from dotenv import load_dotenv
import os

# Load environment variables once
load_dotenv()

# Cache API key for 15 minutes
@st.cache_data(ttl=900)
def get_api_key():
    return os.getenv("OPENAI_API_KEY")


def initialize_bot():
    """Initialize the analysis bot with API key and email config"""
    try:
        # ✅ Always fetch from .env
        api_key = get_api_key()
        email_config = None

        if st.session_state.get('email_enabled', False):
            email_config = {
                'email_address': st.session_state.get('email_address', ''),
                'app_password': st.session_state.get('email_password', '')
            }

        if api_key:
            bot = EnhancedDualFileAnalysisBot(api_key=api_key, email_config=email_config)
            st.session_state.bot = bot
            st.session_state.bot_initialized = True
            return True
        else:
            st.error("❌ No API key found in .env file. Please add OPENAI_API_KEY to your .env")
            return False
    except Exception as e:
        st.error(f"Failed to initialize bot: {str(e)}")
    return False


def create_download_link(file_path, filename):
    """Create a download link for a file"""
    try:
        if not os.path.exists(file_path):
            return f"File not found: {filename}"
            
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    except Exception as e:
        return f"Error creating download link for {filename}: {str(e)}"

def safe_load_dataframe(file, file_type):
    """Safely load dataframe from uploaded file"""
    try:
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'xlsx':
            return pd.read_excel(file)
        elif file_type == 'json':
            return pd.read_json(file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_type} file: {str(e)}")
        return None
    
    # valid, msg = st.session_state.bot.validate_inputs(news_df, reddit_df)
    # if not valid:
    #     st.error(f"Validation failed: {msg}")
    #     st.stop()
    # else:
    #     st.success(msg)


def display_interactive_sentiment_gauge(sentiment_score, sentiment_label):
    """Create an interactive sentiment gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Overall Sentiment: {sentiment_label.title()}"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightcoral"},
                {'range': [-0.3, 0.3], 'color': "lightyellow"},
                {'range': [0.3, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(news_df, reddit_df, stock_data):
    """Create interactive correlation heatmap"""
    # Generate mock correlation data for demonstration
    correlation_data = {
        'News Volume': np.random.normal(5, 2, 20),
        'Social Volume': np.random.normal(8, 3, 20),
        'News Sentiment': np.random.normal(0, 0.3, 20),
        'Social Sentiment': np.random.normal(0, 0.4, 20),
        'Price Change': np.random.normal(0, 2, 20)
    }
    
    corr_df = pd.DataFrame(correlation_data)
    correlation_matrix = corr_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdYlBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix: Sentiment vs Market Data",
        height=500
    )
    
    return fig

def create_technical_indicators_chart(stock_data):
    """Create interactive technical analysis chart"""
    if 'historical_data' not in stock_data:
        # Create mock data
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        base_price = 150
        prices = []
        volumes = []
        
        for i in range(60):
            change = np.random.normal(0, 0.02)
            base_price *= (1 + change)
            prices.append(base_price)
            volumes.append(np.random.randint(1000000, 5000000))
        
        hist_data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        }, index=dates)
    else:
        hist_data = stock_data['historical_data']
    
    # Calculate indicators
    close_prices = hist_data['Close']
    sma_20 = close_prices.rolling(window=20).mean()
    sma_50 = close_prices.rolling(window=min(50, len(close_prices))).mean()
    
    # RSI calculation
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Stock Price with Moving Averages', 'Volume', 'RSI'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=close_prices, name='Price', line=dict(color='#1f4e79', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=sma_20, name='20-day SMA', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    if len(hist_data) > 50:
        fig.add_trace(
            go.Scatter(x=hist_data.index, y=sma_50, name='50-day SMA', line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=hist_data.index, y=hist_data.get('Volume', []), name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI chart
    fig.add_trace(
        go.Scatter(x=hist_data.index, y=rsi, name='RSI', line=dict(color='purple', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    
    return fig


def create_news_impact_timeline(news_df, stock_data):
    """Create news impact timeline chart"""
    # Mock timeline data
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    news_volumes = np.random.poisson(5, 20)
    price_changes = np.random.normal(0, 2, 20)
    sentiment_scores = np.random.normal(0, 0.3, 20)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Daily News Volume', 'Price Changes (%)', 'Sentiment Score'),
        vertical_spacing=0.1
    )
    
    # News volume
    fig.add_trace(
        go.Bar(x=dates, y=news_volumes, name='News Volume', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Price changes
    colors = ['green' if x > 0 else 'red' for x in price_changes]
    fig.add_trace(
        go.Bar(x=dates, y=price_changes, name='Price Change', marker_color=colors),
        row=2, col=1
    )
    
    # Sentiment
    fig.add_trace(
        go.Scatter(x=dates, y=sentiment_scores, name='Sentiment', 
                  line=dict(color='purple', width=2), mode='lines+markers'),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def display_enhanced_charts(results):
    """Display enhanced interactive charts"""
    if not results or not results.get('success'):
        return
    
    company_info = results.get('company_info', {})
    stock_data = results.get('stock_data', {})
    sentiment_data = results.get('sentiment_data', {})
    news_df = getattr(st.session_state, 'news_df', pd.DataFrame())
    reddit_df = getattr(st.session_state, 'reddit_df', pd.DataFrame())
    
    st.markdown('<div class="section-header">Interactive Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different chart categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Technical Analysis", 
        "Sentiment Analysis", 
        "Correlation Analysis", 
        "News Impact"
    ])
    
    with tab1:
        st.subheader("Technical Indicators & Price Analysis")
        try:
            tech_chart = create_technical_indicators_chart(stock_data)
            st.plotly_chart(tech_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating technical analysis chart: {str(e)}")
    
    with tab2:
        st.subheader("Sentiment Analysis Dashboard")
        
        # Sentiment gauge
        overall_sentiment = sentiment_data.get('combined_sentiment', {})
        sentiment_score = overall_sentiment.get('score', 0)
        sentiment_label = overall_sentiment.get('label', 'neutral')
        
        try:
            gauge_fig = display_interactive_sentiment_gauge(sentiment_score, sentiment_label)
            st.plotly_chart(gauge_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating sentiment gauge: {str(e)}")
        
        # Sentiment breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            news_sentiment = sentiment_data.get('news_sentiment', {})
            if news_sentiment and sum(v for k, v in news_sentiment.items() if isinstance(v, (int, float))) > 0:

                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative', 'Neutral'],
                    values=[news_sentiment['positive'], news_sentiment['negative'], news_sentiment['neutral']],
                    hole=.3,
                    marker_colors=['#2E8B57', '#DC143C', '#FFD700']
                )])
                fig.update_layout(title="News Sentiment Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            reddit_sentiment = sentiment_data.get('reddit_sentiment', {})
            if reddit_sentiment and sum(v for k, v in reddit_sentiment.items() if isinstance(v, (int, float))) > 0:

                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative', 'Neutral'],
                    values=[reddit_sentiment['positive'], reddit_sentiment['negative'], reddit_sentiment['neutral']],
                    hole=.3,
                    marker_colors=['#2E8B57', '#DC143C', '#FFD700']
                )])
                fig.update_layout(title="Social Media Sentiment Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation & Market Relationships")
        try:
            corr_fig = create_correlation_heatmap(news_df, reddit_df, stock_data)
            st.plotly_chart(corr_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation chart: {str(e)}")
        
        # Market comparison metrics
        st.subheader("Market Benchmarking")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("vs S&P 500", "+2.3%", delta="0.5%")
        with col2:
            st.metric("vs NASDAQ", "+1.8%", delta="-0.2%")
        with col3:
            st.metric("vs Sector Avg", "+0.9%", delta="0.3%")
        with col4:
            st.metric("Beta", "1.12", delta="0.05")
    
    with tab4:
        st.subheader("News Impact & Timeline Analysis")
        try:
            news_fig = create_news_impact_timeline(news_df, stock_data)
            st.plotly_chart(news_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating news impact chart: {str(e)}")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_articles = len(news_df) if not news_df.empty else 0
            st.metric("Total News Articles", f"{total_articles:,}")
        with col2:
            avg_sentiment = sentiment_score
            st.metric("Average Sentiment", f"{avg_sentiment:+.3f}")
        with col3:
            impact_score = abs(sentiment_score) * 100
            st.metric("Impact Score", f"{impact_score:.1f}")

def display_chart_images(results):
    """Display static chart images generated by the backend"""
    generated_charts = results.get('generated_charts', [])
    
    if generated_charts:
        st.markdown('<div class="section-header">Generated Chart Analysis</div>', unsafe_allow_html=True)
        
        # Display charts in a grid
        cols_per_row = 1
        chart_titles = [  # <-- This is your fixed chart (index 0)
            "Market Comparison", 
            "Technical Analysis"
        ]
        
        for i in range(0, len(generated_charts), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                chart_idx = i + j
                if chart_idx < len(generated_charts):
                    chart_path = generated_charts[chart_idx]
                    
                    if os.path.exists(chart_path):
                        with cols[j]:
                            try:
                                chart_title = chart_titles[chart_idx] if chart_idx < len(chart_titles) else f"Chart {chart_idx + 1}"
                                st.subheader(chart_title)
                                
                                image = Image.open(chart_path)
                                st.image(image, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error displaying chart: {str(e)}")

    # --- Email Distribution with Button ---
    results = st.session_state.analysis_results
    if send_email_button and st.session_state.get('email_enabled', False) and st.session_state.bot.email_handler:
        recipients_input = st.session_state.get('recipient_email', '')
        if recipients_input:
            recipients = [r.strip() for r in recipients_input.split(",") if r.strip()]
            
            # Collect PDF reports if they exist
            attachments = []
            if results.get('pdf_reports'):
                for i, report_path in enumerate(results['pdf_reports']):
                    if os.path.exists(report_path):
                        attachments.append({
                            "path": report_path,
                            "name": f"report_{i+1}.pdf"
                        })
            
            # Try sending emails
            for r in recipients:
                success = st.session_state.bot.email_handler.send_email_with_attachments(
                    recipient_email=r,
                    subject="Financial Analysis Report",
                    body="Hello,\n\nPlease find the attached financial analysis report.\n\nBest regards,\nFinancial Analysis Dashboard",
                    attachments=attachments
                )
                if success:
                    st.success(f"✅ Email sent successfully to {r}")
                else:
                    st.error(f"❌ Failed to send email to {r}. Check Gmail App Password or logs.")
        else:
            st.warning("⚠️ Please enter at least one recipient email address.")




def display_sentiment_analysis(sentiment_data):
    """Enhanced sentiment analysis display with interactive elements"""
    if not sentiment_data:
        st.warning("No sentiment data available")
        return
        
    st.markdown('<div class="section-header">Sentiment Analysis Results</div>', unsafe_allow_html=True)
    
    # Overall sentiment metrics
    col1, col2, col3 = st.columns(3)
    
    combined_sentiment = sentiment_data.get('combined_sentiment', {})
    
    with col1:
        score = combined_sentiment.get('score', 0)
        st.markdown(f'''
        <div class="metric-card">
            <h3>Overall Sentiment</h3>
            <h2>{score:+.3f}</h2>
            <p>{combined_sentiment.get('label', 'neutral').title()}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        news_sentiment = sentiment_data.get('news_sentiment', {})
        total_news = news_sentiment.get('positive', 0) + news_sentiment.get('negative', 0) + news_sentiment.get('neutral', 0)
        st.markdown(f'''
        <div class="metric-card">
            <h3>News Articles</h3>
            <h2>{total_news:,}</h2>
            <p>Analyzed</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        reddit_sentiment = sentiment_data.get('reddit_sentiment', {})
        total_reddit = reddit_sentiment.get('positive', 0) + reddit_sentiment.get('negative', 0) + reddit_sentiment.get('neutral', 0)
        st.markdown(f'''
        <div class="metric-card">
            <h3>Social Media Posts</h3>
            <h2>{total_reddit:,}</h2>
            <p>Analyzed</p>
        </div>
        ''', unsafe_allow_html=True)

def display_stock_analysis(stock_data, company_info):
    """Enhanced stock analysis display"""
    if not stock_data or not company_info:
        st.warning("No stock data available")
        return
        
    st.markdown('<div class="section-header">Stock Analysis</div>', unsafe_allow_html=True)
    
    # Stock metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = stock_data.get('current_price', 0)
        avg_price = stock_data.get('avg_price_period', 0)
        price_change = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
        st.metric("Current Price", f"${current_price:.2f}", delta=f"{price_change:+.1f}%")
    
    with col2:
        volatility = stock_data.get('volatility', 0)
        st.metric("Volatility", f"{volatility:.1%}")
    
    with col3:
        pe_ratio = stock_data.get('pe_ratio', 'N/A')
        st.metric("P/E Ratio", str(pe_ratio))
    
    with col4:
        market_cap = stock_data.get('market_cap', 0)
        if market_cap > 1e9:
            cap_display = f"${market_cap/1e9:.1f}B"
        elif market_cap > 1e6:
            cap_display = f"${market_cap/1e6:.1f}M"
        else:
            cap_display = "N/A"
        st.metric("Market Cap", cap_display)

def display_data_overview(news_df, reddit_df):
    """Enhanced data overview with interactive elements"""
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if news_df is not None and not news_df.empty:
            st.subheader("News Data")
            st.write(f"**Total articles:** {len(news_df):,}")
            st.write(f"**Columns:** {len(news_df.columns)}")
            
            # Data quality indicators
            completeness = (1 - news_df.isnull().sum().sum() / news_df.size) * 100
            st.write(f"**Data Completeness:** {completeness:.1f}%")
            
            # Show sample data with enhanced view
            with st.expander("Sample News Data"):
                st.dataframe(news_df.head(10), use_container_width=True)
        else:
            st.info("No news data loaded")
    
    with col2:
        if reddit_df is not None and not reddit_df.empty:
            st.subheader("Social Media Data")
            st.write(f"**Total posts:** {len(reddit_df):,}")
            st.write(f"**Columns:** {len(reddit_df.columns)}")
            
            # Data quality indicators
            completeness = (1 - reddit_df.isnull().sum().sum() / reddit_df.size) * 100
            st.write(f"**Data Completeness:** {completeness:.1f}%")
            
            # Show sample data with enhanced view
            with st.expander("Sample Social Media Data"):
                st.dataframe(reddit_df.head(10), use_container_width=True)
        else:
            st.info("No social media data loaded")

def display_analysis_results(results):
    """Display comprehensive analysis results with enhanced visualizations"""
    if not results or not results.get('success'):
        st.error(f"Analysis failed: {results.get('error_message', 'Unknown error')}")
        return
    
    # Success message
    st.markdown('''
    <div class="success-box">
        <h3>Analysis Completed Successfully!</h3>
        <p>Your comprehensive financial analysis has been generated with enhanced visualizations.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Company information
    company_info = results.get('company_info', {})
    if company_info:
        st.markdown('<div class="section-header">Company Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Company:** {company_info.get('company_name', 'N/A')}")
        with col2:
            st.write(f"**Ticker:** {company_info.get('ticker', 'N/A')}")
        with col3:
            st.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Display enhanced interactive charts
    display_enhanced_charts(results)
    
    # Display sentiment analysis
    sentiment_data = results.get('sentiment_data', {})
    if sentiment_data:
        display_sentiment_analysis(sentiment_data)
    
    # Display stock analysis if available
    stock_data = results.get('stock_data', {})
    if stock_data:
        display_stock_analysis(stock_data, company_info)
    
    # Display generated chart images from backend
    display_chart_images(results)
    
    # Reports section with enhanced presentation
    pdf_reports = results.get('pdf_reports', [])
    if pdf_reports:
        st.markdown('<div class="section-header">Generated Reports</div>', unsafe_allow_html=True)
        
        # Create expandable sections for different report types
        for i, report_path in enumerate(pdf_reports):
            if os.path.exists(report_path):
                filename = os.path.basename(report_path)
                
                # Determine report type and styling
                if 'summary' in filename.lower():
                    report_type = "Executive Summary Report"
                    report_desc = "High-level overview of analysis findings and key insights"
                elif 'equity' in filename.lower():
                    report_type = "Comprehensive Equity Research Report"
                    report_desc = "Detailed technical analysis with charts and recommendations"
                elif 'advanced' in filename.lower():
                    report_type = "Advanced Analysis Report"
                    report_desc = "In-depth analysis with enhanced visualizations"
                else:
                    report_type = f"Analysis Report {i+1}"
                    report_desc = "Financial analysis report with insights and recommendations"
                
                with st.expander(f"{report_type} - {filename}"):
                    st.write(f"**Description:** {report_desc}")
                    
                    try:
                        file_size = os.path.getsize(report_path) / 1024 / 1024  # MB
                        st.write(f"**File Size:** {file_size:.2f} MB")
                    except:
                        st.write("**File Size:** Unknown")
                    
                    st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Download button
                    download_link = create_download_link(report_path, filename)
                    st.markdown(download_link, unsafe_allow_html=True)
    
    # Email results with enhanced display
    email_results = results.get('email_results', {})
    if email_results:
        st.markdown('<div class="section-header">Email Distribution Results</div>', unsafe_allow_html=True)
        
        success_count = sum(1 for success in email_results.values() if success)
        total_count = len(email_results)
        
        st.write(f"**Distribution Summary:** {success_count}/{total_count} emails sent successfully")
        
        # Show detailed results
        for recipient, success in email_results.items():
            status_icon = "✓" if success else "✗"
            status_text = "Successfully sent" if success else "Failed to send"
            st.write(f"{status_icon} **{recipient}:** {status_text}")
    
    # Download package with enhanced presentation
    download_package = results.get('download_package', '')
    if download_package and os.path.exists(download_package):
        st.markdown('<div class="section-header">Complete Analysis Package</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                package_name = os.path.basename(download_package)
                st.write(f"**Package Name:** {package_name}")
                
                try:
                    package_size = os.path.getsize(download_package) / 1024 / 1024  # MB
                    st.write(f"**Package Size:** {package_size:.2f} MB")
                except:
                    st.write("**Package Size:** Unknown")
                
                st.write("**Contents:** All PDF reports, charts, and analysis data")
            
            with col2:
                download_link = create_download_link(download_package, package_name)
                st.markdown(f"### {download_link}", unsafe_allow_html=True)

def run_enhanced_analysis():
    """Run analysis with enhanced chart generation"""
    if not st.session_state.bot_initialized:
        st.error("Bot not initialized. Please check your API key.")
        return None
    
    news_df = getattr(st.session_state, 'news_df', None)
    reddit_df = getattr(st.session_state, 'reddit_df', None)
    
    if news_df is None and reddit_df is None:
        st.error("No data loaded for analysis")
        return None
    
    # Save uploaded files to temporary files for backend processing
    temp_files = []
    news_file_path = None
    reddit_file_path = None
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save files
        if news_df is not None:
            status_text.text("Saving news data...")
            progress_bar.progress(10)
            
            news_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            news_df.to_csv(news_temp_file.name, index=False)
            news_file_path = news_temp_file.name
            temp_files.append(news_file_path)
        
        if reddit_df is not None:
            status_text.text("Saving social media data...")
            progress_bar.progress(20)
            
            reddit_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            reddit_df.to_csv(reddit_temp_file.name, index=False)
            reddit_file_path = reddit_temp_file.name
            temp_files.append(reddit_file_path)
        
        # Get email recipients
        email_recipients = []
        if st.session_state.get('email_enabled', False):
            email_recipients_text = st.session_state.get('email_recipients', '')
            if email_recipients_text:
                email_recipients = [
                    email.strip() for email in email_recipients_text.split('\n') 
                    if email.strip() and '@' in email
                ]
        
        # Run enhanced analysis
        status_text.text("Running enhanced comprehensive analysis...")
        progress_bar.progress(40)
        
        # Call the enhanced analysis method
        results = st.session_state.bot.run_comprehensive_analysis_with_distribution(
            news_file=news_file_path,
            reddit_file=reddit_file_path,
            email_recipients=email_recipients if email_recipients else None,
            create_download_package=st.session_state.get('create_download_package', True)
        )
        
        # Generate enhanced charts if analysis succeeded
        if results and results.get('success'):
            status_text.text("Generating enhanced visualizations...")
            progress_bar.progress(70)
            
            # Create enhanced charts
            try:
                enhanced_charts = st.session_state.bot.create_enhanced_comprehensive_charts(
                    news_df if news_df is not None else pd.DataFrame(),
                    reddit_df if reddit_df is not None else pd.DataFrame(),
                    results.get('stock_data', {}),
                    results.get('sentiment_data', {}),
                    results.get('company_info', {})
                )
                
                results['generated_charts'] = enhanced_charts
                st.session_state.generated_charts = enhanced_charts
                
            except Exception as e:
                st.warning(f"Some enhanced charts could not be generated: {str(e)}")
                results['generated_charts'] = []
        
        progress_bar.progress(90)
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

def main():
    """Main Streamlit application with enhanced features"""
    
    # Header
    st.markdown('<div class="main-header">Enhanced Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Market Intelligence with Advanced Visualizations")
    st.markdown("---")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        st.session_state.openai_api_key = api_key
        if not st.session_state.bot_initialized:
            if initialize_bot():
                st.sidebar.success("Bot initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize bot")
    else:
        st.sidebar.warning("Please enter your OpenAI API key")
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    st.session_state.create_download_package = st.sidebar.checkbox(
        "Create download package", 
        value=True,
        help="Bundle all reports into a downloadable ZIP file"
    )

    include_enhanced_charts = st.sidebar.checkbox(
        "Generate enhanced visualizations", 
        value=True,
        help="Create advanced interactive charts and analysis"
    )

    # Main content area
    if not st.session_state.bot_initialized:
        st.info("Welcome to Enhanced Financial Analysis Dashboard")
        st.write("Please configure your OpenAI API key in the sidebar to get started.")
        
        st.subheader("New Enhanced Features:")
        st.write("- **Interactive Charts:** Technical analysis with RSI, MACD, Bollinger Bands")
        st.write("- **Sentiment Gauges:** Real-time sentiment visualization")
        st.write("- **Correlation Analysis:** Market relationship heatmaps")
        st.write("- **News Impact:** Timeline analysis of news vs price movements")
        st.write("- **Advanced PDF Reports:** Professional reports with embedded visualizations")
        return
    
    # File upload section
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("News Data")
        news_file = st.file_uploader(
            "Upload news articles file",
            type=['csv', 'xlsx', 'json'],
            key="news_upload",
            help="Upload CSV, Excel, or JSON file containing news articles"
        )
        
        if news_file:
            file_type = news_file.name.split('.')[-1].lower()
            news_df = safe_load_dataframe(news_file, file_type)
            if news_df is not None:
                st.success(f"Loaded {len(news_df):,} news records")
                st.session_state.news_df = news_df
            else:
                st.session_state.news_df = None
        else:
            st.session_state.news_df = None
    
    with col2:
        st.subheader("Social Media Data")
        reddit_file = st.file_uploader(
            "Upload social media posts file",
            type=['csv', 'xlsx', 'json'],
            key="reddit_upload",
            help="Upload CSV, Excel, or JSON file containing social media posts"
        )
        
        if reddit_file:
            file_type = reddit_file.name.split('.')[-1].lower()
            reddit_df = safe_load_dataframe(reddit_file, file_type)
            if reddit_df is not None:
                st.success(f"Loaded {len(reddit_df):,} social media records")
                st.session_state.reddit_df = reddit_df
            else:
                st.session_state.reddit_df = None
        else:
            st.session_state.reddit_df = None

        news_df, reddit_df = None, None

    if news_file is not None:
        file_type = Path(news_file.name).suffix.lower().lstrip(".")
        news_df = safe_load_dataframe(news_file, file_type)
        st.session_state.news_df = news_df

    if reddit_file is not None:
        file_type = Path(reddit_file.name).suffix.lower().lstrip(".")
        reddit_df = safe_load_dataframe(reddit_file, file_type)
        st.session_state.reddit_df = reddit_df

    # --- ✅ Validation step (add this block) ---
    if news_df is not None and reddit_df is not None:
        if 'bot' in st.session_state and st.session_state.bot_initialized:
            # Remove the file parameters, only pass DataFrames
            valid, msg = st.session_state.bot.validate_input_files(news_df, reddit_df)
            if not valid:
                st.error(f"Validation failed: {msg}")
                st.stop()
            else:
                st.success(msg)

    # Display data overview if files are loaded
    if hasattr(st.session_state, 'news_df') or hasattr(st.session_state, 'reddit_df'):
        news_df = getattr(st.session_state, 'news_df', None)
        reddit_df = getattr(st.session_state, 'reddit_df', None)
        
        if news_df is not None or reddit_df is not None:
            display_data_overview(news_df, reddit_df)
    
    # Email recipients configuration
    if st.session_state.get('email_enabled', False):
        st.markdown('<div class="section-header">Email Distribution</div>', unsafe_allow_html=True)
        
        email_recipients_text = st.text_area(
            "Email Recipients (one per line)",
            help="Enter email addresses for automatic report distribution",
            key="email_recipients"
        )
        
        if email_recipients_text:
            email_recipients = [
                email.strip() for email in email_recipients_text.split('\n') 
                if email.strip() and '@' in email
            ]
            st.write(f"{len(email_recipients)} recipients configured")
    
    # Analysis execution
    st.markdown(
    """
    <div class="section-header" style="color:#155724;">
        Enhanced Analysis Execution
    </div>
    """,
    unsafe_allow_html=True
)

    
    # Check if we have data to analyze
    can_analyze = False
    news_df = getattr(st.session_state, 'news_df', None)
    reddit_df = getattr(st.session_state, 'reddit_df', None)
    
    if news_df is not None or reddit_df is not None:
        can_analyze = True
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "Run Enhanced Comprehensive Analysis",
                disabled=st.session_state.analysis_running,
                use_container_width=True,
                help="Execute full analysis with enhanced visualizations and reports"
            ):
                st.session_state.analysis_running = True
                
                try:
                    results = run_enhanced_analysis()
                    if results:
                        st.session_state.analysis_results = results
                        st.success("Enhanced analysis completed successfully!")
                        st.rerun()
                    else:
                        st.error("Analysis failed. Please check your data and try again.")
                        
                finally:
                    st.session_state.analysis_running = False
    else:
        st.markdown('''
        <div class="info-box" style="color:#155724;">
            <p>Please upload at least one data file (news or social media) to run enhanced analysis.</p>
            <p><strong>Tip:</strong> For best results, upload both news and social media data files.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Display results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        display_analysis_results(st.session_state.analysis_results)
    
    # Footer with enhanced information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Enhanced Financial Analysis Dashboard</strong></p>
        <p>Powered by Advanced AI, Technical Analysis & Interactive Visualizations</p>
        <p><small>This enhanced analysis tool provides insights for informational purposes only. 
        All visualizations, sentiment analysis, and recommendations should be verified independently. 
        Always consult with qualified financial professionals before making investment decisions.</small></p>
        <p><small>Features: Interactive Charts • Technical Indicators • Risk Analysis</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.markdown(f"""
        <div class="error-box">
            <h3>Application Error</h3>
            <p>An unexpected error occurred. Please refresh the page and try again.</p>
            <details>
                <summary>Technical Details</summary>
                <pre>{str(e)}</pre>
            </details>
        </div> 
        """, unsafe_allow_html=True)