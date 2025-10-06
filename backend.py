import pandas as pd
import json
import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
import hashlib
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import re
import warnings
import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
import tempfile
import zipfile
import mimetypes
warnings.filterwarnings('ignore')

# PDF generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wordcloud
from wordcloud import WordCloud
from dotenv import load_dotenv
load_dotenv()  # this loads variables from .env into os.environ
print("API Key loaded:", os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Will use mock price data.")


class EmailHandler:
    """Simplified email handler for sending reports"""
    
    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address
        self.app_password = app_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
    def send_email_with_attachments(self, recipient_email: str, subject: str, body: str, 
                                  attachments: List[Dict[str, str]] = None) -> bool:
        """Send email with multiple attachments"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    file_path = attachment['path']
                    filename = attachment['name']
                    
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment_file:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment_file.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {filename}'
                        )
                        msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.app_password)
            text = msg.as_string()
            server.sendmail(self.email_address, recipient_email, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    # Example usage function
    def example_usage_with_email():
        """Example of how to use the enhanced analysis with email distribution"""
        
        # Initialize with email configuration
        email_config = {
            'email_address': 'your_email@gmail.com',
            'app_password': 'your_app_password'  # Gmail app password
        }
        
        # Create analysis bot
        bot = EnhancedDualFileAnalysisBot(
            api_key="your_openai_api_key",
            email_config=email_config
        )
        
        # Define email recipients
        email_recipients = [
            'investor1@example.com',
            'analyst@company.com',
            'portfolio.manager@firm.com'
        ]
        
        # Run comprehensive analysis with distribution
        results = bot.run_comprehensive_analysis_with_distribution(
            news_file='path/to/news.csv',
            reddit_file='path/to/reddit.csv',
            email_recipients=email_recipients,
            create_download_package=True
        )
        
        # Check results
        if results['success']:
            print(f"Analysis completed successfully!")
            print(f"Company: {results['company_info'].get('company_name', 'N/A')}")
            print(f"Overall Sentiment: {results['sentiment_data']['combined_sentiment']['label']}")
            print(f"PDF Reports Created: {len(results['pdf_reports'])}")
            
            # Email results
            for recipient, success in results['email_results'].items():
                status = "✓ Sent" if success else "✗ Failed"
                print(f"Email to {recipient}: {status}")
            
            # Download package
            if results['download_package']:
                print(f"Download package created: {results['download_package']}")
        else:
            print(f"Analysis failed: {results['error_message']}")
# if __name__ == "__main__":
#     EmailHandler.example_usage_with_email()


class EnhancedDualFileAnalysisBot:
    """Enhanced analysis bot with comprehensive visualization capabilities"""
    
    def __init__(self, api_key: str, email_config: Dict[str, str] = None):
        """Initialize the enhanced analysis bot"""
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats with comprehensive encoding handling"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                # Try multiple encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'windows-1252', 'utf-16']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                        logger.info(f"Successfully loaded CSV with {encoding} encoding")
                        return df
                    except (UnicodeDecodeError, pd.errors.ParserError) as e:
                        logger.warning(f"Failed to load with {encoding}: {str(e)}")
                        continue
                
                # Final attempt with error replacement
                try:
                    df = pd.read_csv(file_path, encoding='latin1', errors='replace')
                    logger.warning("Loaded CSV with latin1 encoding (errors replaced)")
                    return df
                except Exception as e:
                    logger.error(f"Final CSV load attempt failed: {str(e)}")
                    return None
                
            elif file_extension == '.xlsx':
                try:
                    df = pd.read_excel(file_path)
                    logger.info("Loaded Excel file successfully")
                    return df
                except Exception as e:
                    logger.error(f"Failed to load Excel: {str(e)}")
                    return None
                
            elif file_extension == '.json':
                try:
                    df = pd.read_json(file_path)
                    logger.info("Loaded JSON file successfully")
                    return df
                except Exception as e:
                    logger.error(f"Failed to load JSON: {str(e)}")
                    return None
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def identify_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the text column from the dataframe"""
        text_column_candidates = [
            'text', 'content', 'news', 'article', 'description', 
            'summary', 'body', 'message', 'title', 'headline', 'headline_or_text'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in text_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in text_column_candidates:
                if candidate in col.lower():
                    return col
        
        # Find column with longest average text length
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length
        
        if text_lengths:
            return max(text_lengths, key=text_lengths.get)
        
        return None
    
    def extract_company_ticker(self, df: pd.DataFrame, text_column: str = None) -> Dict[str, str]:
        """Enhanced company extraction - prioritize ticker column detection"""
        company_info = {'company_name': '', 'ticker': ''}
        
        # ENHANCED: Better ticker column detection
        ticker_columns = [
            'ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL',
            'stock_ticker', 'stock_symbol', 'company_ticker', 'company_symbol',
            'tick', 'sym', 'stockticker', 'stocksymbol'
        ]
        
        # Check for ticker in dedicated columns with better validation
        for col in ticker_columns:
            if col in df.columns and not df[col].isna().all():
                # Get all unique non-null tickers
                unique_tickers = df[col].dropna().unique()
                for ticker_value in unique_tickers:
                    ticker_str = str(ticker_value).strip().upper()
                    # Validate ticker format (1-5 letters, common patterns)
                    if ticker_str and ticker_str != 'NAN' and re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker_str):
                        company_info['ticker'] = ticker_str
                        logger.info(f"Found ticker in column '{col}': {ticker_str}")
                        break
                if company_info['ticker']:
                    break
        
        # Enhanced company name detection
        company_columns = [
            'company', 'Company', 'company_name', 'Company_Name', 'COMPANY_NAME',
            'firm', 'corporation', 'corp', 'organization', 'org', 'entity',
            'companyname', 'name', 'Name', 'business_name'
        ]
        
        for col in company_columns:
            if col in df.columns and not df[col].isna().all():
                company_name = str(df[col].iloc[0]).strip()
                if company_name and company_name.lower() != 'nan':
                    company_info['company_name'] = company_name
                    logger.info(f"Found company name in column '{col}': {company_name}")
                    break
        
        # If no ticker found in columns, try to extract from text content
        if not company_info['ticker'] and text_column and text_column in df.columns:
            sample_texts = df[text_column].dropna().head(10).tolist()
            if sample_texts:
                combined_sample = ' '.join([str(text)[:200] for text in sample_texts])
                extracted_info = self._extract_company_from_text_enhanced(combined_sample)
                
                if extracted_info.get('ticker') and not company_info['ticker']:
                    company_info['ticker'] = extracted_info['ticker']
                if extracted_info.get('company_name') and not company_info['company_name']:
                    company_info['company_name'] = extracted_info['company_name']
        
        # Get company name from ticker if we have ticker but not name
        if company_info['ticker'] and not company_info['company_name']:
            mapped_name = self._get_company_name_from_ticker(company_info['ticker'])
            if mapped_name:
                company_info['company_name'] = mapped_name
        
        logger.info(f"Final extracted company info: {company_info}")
        return company_info
    
    def validate_input_files(self, news_df: Optional[pd.DataFrame], reddit_df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
        try:
            if news_df is None and reddit_df is None:
                return False, "At least one file (news or reddit) must be provided."

            # If only one file is provided, it's valid
            if news_df is None or reddit_df is None:
                return True, "Validation passed: Single file provided."

            # If both are provided, do ticker checks
            if news_df.equals(reddit_df):
                return False, "Both input files appear to be identical."
            
            ticker_cols_news = [col for col in news_df.columns if 'ticker' in col.lower()]
            ticker_cols_reddit = [col for col in reddit_df.columns if 'ticker' in col.lower()]
            if ticker_cols_news and ticker_cols_reddit:
                news_tickers = set(news_df[ticker_cols_news[0]].dropna().unique())
                reddit_tickers = set(reddit_df[ticker_cols_reddit[0]].dropna().unique())
                if news_tickers != reddit_tickers:
                    return False, "Ticker mismatch: both files must have the same ticker symbols."

            return True, "Validation passed: Files are valid."
        except Exception as e:
            return False, f"Validation error: {str(e)}"



    def _extract_company_from_text_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced AI extraction with better ticker recognition"""
        prompt = f"""
        From the following text, extract the main company name and stock ticker symbol.
        Focus on finding valid stock ticker symbols (1-5 uppercase letters, may have dots).
        
        Common ticker patterns: AAPL, MSFT, GOOGL, TSLA, BRK.A, etc.
        
        Text: {text}
        
        Please respond in this exact format:
        Company: [company name or "Not found"]
        Ticker: [ticker symbol or "Not found"]
        
        Rules:
        - Ticker should be uppercase letters only (1-5 chars)
        - Company name should be the full official name
        - If multiple tickers mentioned, pick the most prominent one
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting company information from financial text. Focus on accuracy and standard ticker formats."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            company_name = ""
            ticker = ""
            
            for line in content.split('\n'):
                if line.strip().startswith('Company:'):
                    company_name = line.split(':', 1)[1].strip()
                    if company_name == "Not found":
                        company_name = ""
                elif line.strip().startswith('Ticker:'):
                    ticker = line.split(':', 1)[1].strip().upper()
                    if ticker == "NOT FOUND":
                        ticker = ""
            
            # Validate ticker format
            if ticker and not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
                ticker = ""
                
            return {'company_name': company_name, 'ticker': ticker}
            
        except Exception as e:
            logger.warning(f"Failed to extract company info from text: {str(e)}")
            return {'company_name': '', 'ticker': ''}
    
    def _get_company_name_from_ticker(self, ticker: str) -> str:
        """Enhanced ticker to company name mapping with more companies"""
        ticker_to_company = {
            # Technology
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'INTC': 'Intel Corporation',
            'AMD': 'Advanced Micro Devices Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'TXN': 'Texas Instruments Inc.',
            'IBM': 'International Business Machines Corp.',
            'ORCL': 'Oracle Corporation',
            'AVGO': 'Broadcom Inc.',
            
            # Financial
            'BRK.A': 'Berkshire Hathaway Inc.',
            'BRK.B': 'Berkshire Hathaway Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'WFC': 'Wells Fargo & Co.',
            'V': 'Visa Inc.',
            'MA': 'Mastercard Inc.',
            'GS': 'Goldman Sachs Group Inc.',
            'MS': 'Morgan Stanley',
            'C': 'Citigroup Inc.',
            'AXP': 'American Express Co.',
            'BLK': 'BlackRock Inc.',
            
            # Healthcare & Pharma
            'JNJ': 'Johnson & Johnson',
            'UNH': 'UnitedHealth Group Inc.',
            'LLY': 'Eli Lilly and Company',
            'ABBV': 'AbbVie Inc.',
            'MRK': 'Merck & Co. Inc.',
            'ABT': 'Abbott Laboratories',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'DHR': 'Danaher Corporation',
            'PFE': 'Pfizer Inc.',
            'BMY': 'Bristol Myers Squibb Co.',
            
            # Consumer & Retail
            'WMT': 'Walmart Inc.',
            'COST': 'Costco Wholesale Corporation',
            'HD': 'Home Depot Inc.',
            'PG': 'Procter & Gamble Co.',
            'KO': 'Coca-Cola Co.',
            'PEP': 'PepsiCo Inc.',
            'NKE': 'Nike Inc.',
            'MCD': 'McDonald\'s Corp.',
            'SBUX': 'Starbucks Corporation',
            'DIS': 'Walt Disney Co.',
            
            # Energy
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            'COP': 'ConocoPhillips',
            'SLB': 'Schlumberger N.V.',
            'EOG': 'EOG Resources Inc.',
            'NEE': 'NextEra Energy Inc.',
            
            # Telecom
            'VZ': 'Verizon Communications Inc.',
            'T': 'AT&T Inc.',
            'CMCSA': 'Comcast Corporation',
            'TMUS': 'T-Mobile US Inc.',
            
            # Industrial
            'BA': 'Boeing Co.',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Co.',
            'MMM': '3M Co.',
            'HON': 'Honeywell International Inc.',
            'UPS': 'United Parcel Service Inc.',
            'RTX': 'Raytheon Technologies Corp.',
            
            # Real Estate & REITs
            'AMT': 'American Tower Corporation',
            'PLD': 'Prologis Inc.',
            'CCI': 'Crown Castle International Corp.',
            
            # Other Major Companies
            'ACN': 'Accenture Plc',
            'LOW': 'Lowe\'s Companies Inc.',
            'SPGI': 'S&P Global Inc.',
            'BDX': 'Becton Dickinson and Co.',
            'MDT': 'Medtronic Plc',
            'ISRG': 'Intuitive Surgical Inc.',
            'NOW': 'ServiceNow Inc.',
            'INTU': 'Intuit Inc.',
            'TJX': 'TJX Companies Inc.',
            'UNP': 'Union Pacific Corporation',
            'DE': 'Deere & Company'
        }
        
        return ticker_to_company.get(ticker.upper(), '')
    
    def get_stock_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get stock price data using yfinance"""
        if not YFINANCE_AVAILABLE:
            return self._generate_mock_price_data()
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            stock = yf.Ticker(ticker)
            hist = stock.download(start=start_date, end=end_date, progress=False)
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return self._generate_mock_price_data()
            
            current_price = hist['Close'].iloc[-1]
            avg_price = hist['Close'].mean()
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Get additional stock info
            info = stock.info
            
            return {
                'current_price': round(current_price, 2),
                'avg_price_period': round(avg_price, 2),
                'volatility': volatility,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'price_data_available': True,
                'stock_info': info,
                'historical_data': hist  # Added for chart creation
            }
            
        except Exception as e:
            logger.warning(f"Failed to get real price data for {ticker}: {str(e)}")
            return self._generate_mock_price_data()
    
    def _generate_mock_price_data(self) -> Dict[str, Any]:
        """Generate mock price data when real data is not available"""
        mock_current_price = np.random.uniform(150, 200)
        # Generate mock historical data for charts
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_prices = [mock_current_price * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
        mock_hist = pd.DataFrame({'Close': mock_prices}, index=dates)
        
        return {
            'current_price': round(mock_current_price, 2),
            'avg_price_period': round(mock_current_price * np.random.uniform(0.98, 1.02), 2),
            'volatility': 0.25,
            'market_cap': 0,
            'pe_ratio': 0,
            'price_data_available': False,
            'historical_data': mock_hist
        }
    
    def analyze_data_patterns(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data pattern analysis for visualizations"""
        analysis = {
            'news_analysis': {},
            'reddit_analysis': {},
            'comparative_analysis': {}
        }
        
        # News data analysis
        if news_df is not None and not news_df.empty:
            news_text_col = self.identify_text_column(news_df)
            
            # Basic statistics
            analysis['news_analysis'] = {
                'total_articles': len(news_df),
                'columns': list(news_df.columns),
                'text_column': news_text_col,
                'date_columns': [col for col in news_df.columns if 'date' in col.lower() or 'time' in col.lower()],
                'avg_text_length': news_df[news_text_col].astype(str).str.len().mean() if news_text_col else 0,
                'word_frequency': self._get_word_frequency(news_df, news_text_col) if news_text_col else {}
            }
            
            # Time series analysis if date column exists
            date_col = None
            for col in analysis['news_analysis']['date_columns']:
                try:
                    news_df[col] = pd.to_datetime(news_df[col])
                    date_col = col
                    break
                except:
                    continue
            
            if date_col:
                daily_counts = news_df.groupby(news_df[date_col].dt.date).size()
                analysis['news_analysis']['time_series'] = daily_counts.to_dict()
        
        # Reddit data analysis
        if reddit_df is not None and not reddit_df.empty:
            reddit_text_col = self.identify_text_column(reddit_df)
            
            analysis['reddit_analysis'] = {
                'total_posts': len(reddit_df),
                'columns': list(reddit_df.columns),
                'text_column': reddit_text_col,
                'date_columns': [col for col in reddit_df.columns if 'date' in col.lower() or 'time' in col.lower()],
                'avg_text_length': reddit_df[reddit_text_col].astype(str).str.len().mean() if reddit_text_col else 0,
                'word_frequency': self._get_word_frequency(reddit_df, reddit_text_col) if reddit_text_col else {}
            }
            
            # Engagement metrics if available
            engagement_cols = ['upvotes', 'score', 'ups', 'comments', 'comment_count']
            for col in engagement_cols:
                if col in reddit_df.columns:
                    analysis['reddit_analysis'][f'{col}_stats'] = {
                        'mean': reddit_df[col].mean(),
                        'median': reddit_df[col].median(),
                        'max': reddit_df[col].max(),
                        'min': reddit_df[col].min()
                    }
        
        # Comparative analysis
        if news_df is not None and reddit_df is not None:
            analysis['comparative_analysis'] = {
                'volume_comparison': {
                    'news': len(news_df),
                    'reddit': len(reddit_df)
                },
                'text_length_comparison': {
                    'news_avg': analysis['news_analysis'].get('avg_text_length', 0),
                    'reddit_avg': analysis['reddit_analysis'].get('avg_text_length', 0)
                }
            }
        
        return analysis
    
    def _get_word_frequency(self, df: pd.DataFrame, text_column: str, top_n: int = 20) -> Dict[str, int]:
        """Get word frequency analysis"""
        if not text_column or text_column not in df.columns:
            return {}
        
        try:
            # Combine all text
            all_text = ' '.join(df[text_column].astype(str).str.lower())
            
            # Simple word extraction (remove common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            words = re.findall(r'\b[a-z]+\b', all_text)
            words = [word for word in words if len(word) > 2 and word not in stop_words]
            
            word_freq = Counter(words)
            return dict(word_freq.most_common(top_n))
            
        except Exception as e:
            logger.warning(f"Failed to get word frequency: {str(e)}")
            return {}
    
    def create_comprehensive_charts(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame, 
                                  stock_data: Dict[str, Any], data_analysis: Dict[str, Any]) -> List[str]:
        """Create comprehensive set of charts for analysis"""
        chart_paths = []
        
        try:
            # Set style
            plt.style.use('seaborn') 
            # Chart 3: Word Frequency Analysis
            chart_paths.append(self._create_word_frequency_chart(data_analysis))
            
            # Chart 4: Stock Price Analysis
            chart_paths.append(self._create_stock_price_chart(stock_data))
            
            # Chart 5: Returns Distribution
            chart_paths.append(self._create_returns_distribution_chart(stock_data))
            
            # Chart 6: Risk Assessment
            chart_paths.append(self._create_risk_assessment_chart(stock_data))
            
            # Chart 7: Data Quality Assessment
            chart_paths.append(self._create_data_quality_chart(news_df, reddit_df))
            
            return [path for path in chart_paths if path]  # Remove None values
            
        except Exception as e:
            logger.error(f"Error creating comprehensive charts: {str(e)}")
            return []
    
    def _create_word_frequency_chart(self, data_analysis: Dict[str, Any]) -> str:
        """Create word frequency analysis chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # News word frequency
            news_words = data_analysis.get('news_analysis', {}).get('word_frequency', {})
            if news_words:
                words = list(news_words.keys())[:15]
                counts = list(news_words.values())[:15]
                
                bars = axes[0, 0].barh(words, counts, color='#1f4e79', alpha=0.8)
                axes[0, 0].set_title('Top News Keywords', fontweight='bold')
                axes[0, 0].set_xlabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    axes[0, 0].text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                                   f'{int(width)}', ha='left', va='center', fontsize=8)
            
            # Reddit word frequency
            reddit_words = data_analysis.get('reddit_analysis', {}).get('word_frequency', {})
            if reddit_words:
                words = list(reddit_words.keys())[:15]
                counts = list(reddit_words.values())[:15]
                
                bars = axes[0, 1].barh(words, counts, color='#2E8B57', alpha=0.8)
                axes[0, 1].set_title('Top Social Media Keywords', fontweight='bold')
                axes[0, 1].set_xlabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    axes[0, 1].text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                                   f'{int(width)}', ha='left', va='center', fontsize=8)
            
            # Combined word cloud for news
            if news_words:
                try:
                    wordcloud_news = WordCloud(width=400, height=300, background_color='white',
                                             colormap='Blues').generate_from_frequencies(news_words)
                    axes[1, 0].imshow(wordcloud_news, interpolation='bilinear')
                    axes[1, 0].axis('off')
                    axes[1, 0].set_title('News Word Cloud', fontweight='bold')
                except:
                    axes[1, 0].text(0.5, 0.5, 'Word cloud not available', ha='center', va='center',
                                   transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('News Word Cloud', fontweight='bold')
            
            # Combined word cloud for reddit
            if reddit_words:
                try:
                    wordcloud_reddit = WordCloud(width=400, height=300, background_color='white',
                                               colormap='Greens').generate_from_frequencies(reddit_words)
                    axes[1, 1].imshow(wordcloud_reddit, interpolation='bilinear')
                    axes[1, 1].axis('off')
                    axes[1, 1].set_title('Social Media Word Cloud', fontweight='bold')
                except:
                    axes[1, 1].text(0.5, 0.5, 'Word cloud not available', ha='center', va='center',
                                   transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Social Media Word Cloud', fontweight='bold')
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating word frequency chart: {str(e)}")
            plt.close()
            return ""
    
    def _create_stock_price_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create comprehensive stock price analysis chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price trend
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                axes[0, 0].plot(hist_data.index, hist_data['Close'], linewidth=2, color='#1f4e79')
                axes[0, 0].set_title('Stock Price Trend', fontweight='bold')
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Price ($)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add current price line
                current_price = stock_data.get('current_price', 0)
                axes[0, 0].axhline(y=current_price, color='red', linestyle='--', alpha=0.7,
                                  label=f'Current: ${current_price:.2f}')
                axes[0, 0].legend()
            
            # Volume analysis (if available)
            if 'historical_data' in stock_data and 'Volume' in stock_data['historical_data'].columns:
                volume_data = stock_data['historical_data']['Volume']
                axes[0, 1].bar(volume_data.index, volume_data, color='#2E8B57', alpha=0.7)
                axes[0, 1].set_title('Trading Volume', fontweight='bold')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Volume')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                # Price change chart instead
                if 'historical_data' in stock_data:
                    price_changes = stock_data['historical_data']['Close'].pct_change().dropna() * 100
                    axes[0, 1].bar(range(len(price_changes)), price_changes, 
                                  color=['green' if x > 0 else 'red' for x in price_changes], alpha=0.7)
                    axes[0, 1].set_title('Daily Price Changes (%)', fontweight='bold')
                    axes[0, 1].set_xlabel('Days')
                    axes[0, 1].set_ylabel('Change (%)')
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Moving averages
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                ma_7 = hist_data['Close'].rolling(window=7).mean()
                ma_14 = hist_data['Close'].rolling(window=14).mean()
                
                axes[1, 0].plot(hist_data.index, hist_data['Close'], label='Price', linewidth=2, color='#1f4e79')
                axes[1, 0].plot(hist_data.index, ma_7, label='7-day MA', linewidth=1.5, color='orange')
                axes[1, 0].plot(hist_data.index, ma_14, label='14-day MA', linewidth=1.5, color='red')
                axes[1, 0].set_title('Price with Moving Averages', fontweight='bold')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Price ($)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Key metrics summary
            axes[1, 1].axis('off')
            metrics_data = [
                ['Metric', 'Value'],
                ['Current Price', f"${stock_data.get('current_price', 0):.2f}"],
                ['Average Price', f"${stock_data.get('avg_price_period', 0):.2f}"],
                ['Volatility', f"{stock_data.get('volatility', 0):.1%}"],
                ['Market Cap', f"${stock_data.get('market_cap', 0):,.0f}" if stock_data.get('market_cap') else 'N/A'],
                ['P/E Ratio', f"{stock_data.get('pe_ratio', 'N/A')}"],
                ['Data Source', 'Real-time' if stock_data.get('price_data_available') else 'Simulated']
            ]
            
            table = axes[1, 1].table(cellText=metrics_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            # Color header row
            for i in range(len(metrics_data[0])):
                table[(0, i)].set_facecolor('#1f4e79')
                table[(0, i)].set_text_props(weight='bold', color='white')
            axes[1, 1].set_title('Key Metrics Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating stock price chart: {str(e)}")
            plt.close()
            return ""

    def analyze_sentiment_comprehensive(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive sentiment analysis for both datasets"""
        try:
            sentiment_data = {
                'news_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0, 'scores': []},
                'reddit_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0, 'scores': []},
                'combined_sentiment': {'score': 0, 'label': 'neutral'}
            }
            
            # Handle None or empty dataframes
            if news_df is None:
                news_df = pd.DataFrame()
            if reddit_df is None:
                reddit_df = pd.DataFrame()
            
            # Simple keyword-based sentiment analysis
            positive_keywords = ['buy', 'bullish', 'positive', 'growth', 'strong', 'good', 'great', 
                            'excellent', 'rise', 'gain', 'profit', 'success', 'optimistic', 'upgrade']
            negative_keywords = ['sell', 'bearish', 'negative', 'decline', 'weak', 'bad', 'poor', 
                            'terrible', 'fall', 'loss', 'fail', 'pessimistic', 'downgrade']
            
            def calculate_sentiment_score(text):
                if pd.isna(text) or text is None:
                    return 0
                text_lower = str(text).lower()
                pos_count = sum(text_lower.count(word) for word in positive_keywords)
                neg_count = sum(text_lower.count(word) for word in negative_keywords)
                
                if pos_count + neg_count == 0:
                    return 0
                return (pos_count - neg_count) / (pos_count + neg_count)
            
            # News sentiment analysis
            if not news_df.empty:
                news_text_col = self.identify_text_column(news_df)
                if news_text_col and news_text_col in news_df.columns:
                    try:
                        news_scores = news_df[news_text_col].apply(calculate_sentiment_score)
                        sentiment_data['news_sentiment']['scores'] = news_scores.tolist()
                        sentiment_data['news_sentiment']['positive'] = int((news_scores > 0.1).sum())
                        sentiment_data['news_sentiment']['negative'] = int((news_scores < -0.1).sum())
                        sentiment_data['news_sentiment']['neutral'] = int(((news_scores >= -0.1) & (news_scores <= 0.1)).sum())
                    except Exception as e:
                        logger.warning(f"Error processing news sentiment: {str(e)}")
            
            # Reddit sentiment analysis
            if not reddit_df.empty:
                reddit_text_col = self.identify_text_column(reddit_df)
                if reddit_text_col and reddit_text_col in reddit_df.columns:
                    try:
                        reddit_scores = reddit_df[reddit_text_col].apply(calculate_sentiment_score)
                        sentiment_data['reddit_sentiment']['scores'] = reddit_scores.tolist()
                        sentiment_data['reddit_sentiment']['positive'] = int((reddit_scores > 0.1).sum())
                        sentiment_data['reddit_sentiment']['negative'] = int((reddit_scores < -0.1).sum())
                        sentiment_data['reddit_sentiment']['neutral'] = int(((reddit_scores >= -0.1) & (reddit_scores <= 0.1)).sum())
                    except Exception as e:
                        logger.warning(f"Error processing reddit sentiment: {str(e)}")
            
            # Combined sentiment
            all_scores = sentiment_data['news_sentiment']['scores'] + sentiment_data['reddit_sentiment']['scores']
            if all_scores:
                combined_score = float(np.mean(all_scores))
                sentiment_data['combined_sentiment']['score'] = combined_score
                if combined_score > 0.1:
                    sentiment_data['combined_sentiment']['label'] = 'positive'
                elif combined_score < -0.1:
                    sentiment_data['combined_sentiment']['label'] = 'negative'
                else:
                    sentiment_data['combined_sentiment']['label'] = 'neutral'
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'news_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0, 'scores': []},
                'reddit_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0, 'scores': []},
                'combined_sentiment': {'score': 0, 'label': 'neutral'}
            }

    def create_sentiment_chart(self, sentiment_data: Dict[str, Any]) -> str:
        """Create comprehensive sentiment analysis chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # News sentiment pie chart
            news_data = sentiment_data['news_sentiment']
            if news_data['positive'] + news_data['negative'] + news_data['neutral'] > 0:
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [news_data['positive'], news_data['negative'], news_data['neutral']]
                colors = ['#2E8B57', '#DC143C', '#FFD700']
                
                axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('News Sentiment Distribution', fontweight='bold')
            
            # Reddit sentiment pie chart
            reddit_data = sentiment_data['reddit_sentiment']
            if reddit_data['positive'] + reddit_data['negative'] + reddit_data['neutral'] > 0:
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [reddit_data['positive'], reddit_data['negative'], reddit_data['neutral']]
                colors = ['#2E8B57', '#DC143C', '#FFD700']
                
                axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Social Media Sentiment Distribution', fontweight='bold')
            
            # Sentiment comparison bar chart
            categories = ['News', 'Social Media']
            positive_counts = [news_data['positive'], reddit_data['positive']]
            negative_counts = [news_data['negative'], reddit_data['negative']]
            neutral_counts = [news_data['neutral'], reddit_data['neutral']]
            
            x = np.arange(len(categories))
            width = 0.25
            
            bars1 = axes[1, 0].bar(x - width, positive_counts, width, label='Positive', color='#2E8B57')
            bars2 = axes[1, 0].bar(x, negative_counts, width, label='Negative', color='#DC143C')
            bars3 = axes[1, 0].bar(x + width, neutral_counts, width, label='Neutral', color='#FFD700')
            
            axes[1, 0].set_title('Sentiment Comparison by Source', fontweight='bold')
            axes[1, 0].set_xlabel('Source')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            # Overall sentiment gauge
            combined_score = sentiment_data['combined_sentiment']['score']
            combined_label = sentiment_data['combined_sentiment']['label']
            
            # Create gauge chart
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Color zones
            axes[1, 1].fill_between(theta[:33], 0, r[:33], color='#DC143C', alpha=0.3, label='Negative')
            axes[1, 1].fill_between(theta[33:67], 0, r[33:67], color='#FFD700', alpha=0.3, label='Neutral')
            axes[1, 1].fill_between(theta[67:], 0, r[67:], color='#2E8B57', alpha=0.3, label='Positive')
            
            # Sentiment needle
            needle_angle = (combined_score + 1) * np.pi / 2  # Convert [-1,1] to [0,π]
            axes[1, 1].arrow(0, 0, 0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle),
                            head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            axes[1, 1].set_xlim(-1.2, 1.2)
            axes[1, 1].set_ylim(0, 1.2)
            axes[1, 1].set_aspect('equal')
            axes[1, 1].set_title(f'Overall Sentiment: {combined_label.title()}\nScore: {combined_score:.3f}', 
                            fontweight='bold')
            axes[1, 1].text(0, -0.3, f'({combined_label.title()})', ha='center', fontsize=12, fontweight='bold',
                        color={'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'}[combined_label])
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {str(e)}")
            plt.close()
            return ""
        
    def _create_returns_distribution_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create returns distribution analysis chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                returns = hist_data['Close'].pct_change().dropna() * 100
                
                # Histogram of returns
                axes[0, 0].hist(returns, bins=20, alpha=0.7, color='#2E8B57', edgecolor='black')
                axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                                    label=f'Mean: {returns.mean():.2f}%')
                axes[0, 0].set_title('Daily Returns Distribution', fontweight='bold')
                axes[0, 0].set_xlabel('Daily Return (%)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                
                # Cumulative returns
                cumulative_returns = (1 + returns/100).cumprod() - 1
                axes[0, 1].plot(cumulative_returns.index, cumulative_returns * 100, 
                                linewidth=2, color='#1f4e79')
                axes[0, 1].set_title('Cumulative Returns', fontweight='bold')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Cumulative Return (%)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Rolling volatility
                rolling_vol = returns.rolling(window=7).std() * np.sqrt(252)
                axes[1, 0].plot(rolling_vol.index, rolling_vol, linewidth=2, color='orange')
                axes[1, 0].set_title('7-Day Rolling Volatility', fontweight='bold')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Annualized Volatility')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Risk metrics table
                axes[1, 1].axis('off')
                risk_metrics = [
                    ['Risk Metric', 'Value'],
                    ['Daily Volatility', f"{returns.std():.2f}%"],
                    ['Annualized Volatility', f"{stock_data.get('volatility', 0):.1%}"],
                    ['Max Daily Gain', f"{returns.max():.2f}%"],
                    ['Max Daily Loss', f"{returns.min():.2f}%"],
                    ['Positive Days', f"{(returns > 0).sum()}/{len(returns)}"],
                    ['Sharpe Ratio (approx)', f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}" if returns.std() > 0 else 'N/A']
                ]
                
                table = axes[1, 1].table(cellText=risk_metrics, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                # Color header row
                for i in range(len(risk_metrics[0])):
                    table[(0, i)].set_facecolor('#8B0000')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                axes[1, 1].set_title('Risk Metrics', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating returns distribution chart: {str(e)}")
            plt.close()
            return ""

    def _create_risk_assessment_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create risk assessment visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Risk level comparison
            current_vol = stock_data.get('volatility', 0.25) * 100
            risk_categories = ['Low Risk\n(0-15%)', 'Medium Risk\n(15-25%)', 'High Risk\n(25%+)']
            risk_thresholds = [15, 25, 35]
            colors = ['green', 'orange', 'red']
            
            bars = axes[0, 0].bar(risk_categories, risk_thresholds, color=colors, alpha=0.6, edgecolor='black')
            axes[0, 0].axhline(y=current_vol, color='blue', linestyle='--', linewidth=3,
                                label=f'Current: {current_vol:.1f}%')
            axes[0, 0].set_title('Risk Assessment - Volatility Comparison', fontweight='bold')
            axes[0, 0].set_ylabel('Volatility (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk-Return scatter (simulated benchmark comparison)
            benchmarks = {
                'S&P 500': (8, 16),
                'NASDAQ': (10, 20),
                'Bonds': (3, 5),
                'Real Estate': (6, 12),
                'Commodities': (5, 18)
            }
            
            current_return = 0  # Placeholder - could calculate from historical data
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                if len(hist_data) > 1:
                    total_return = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1) * 100
                    annualized_return = total_return * (365 / len(hist_data))
                    current_return = annualized_return
            
            benchmark_returns = [data[0] for data in benchmarks.values()]
            benchmark_vols = [data[1] for data in benchmarks.values()]
            
            axes[0, 1].scatter(benchmark_vols, benchmark_returns, s=100, alpha=0.7, c='lightblue', edgecolors='black')
            axes[0, 1].scatter([current_vol], [current_return], s=200, c='red', marker='*', 
                                edgecolors='black', label='Current Stock')
            
            for i, (name, _) in enumerate(benchmarks.items()):
                axes[0, 1].annotate(name, (benchmark_vols[i], benchmark_returns[i]), 
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[0, 1].set_title('Risk-Return Comparison', fontweight='bold')
            axes[0, 1].set_xlabel('Volatility (%)')
            axes[0, 1].set_ylabel('Expected Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # VaR simulation (simplified)
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                returns = hist_data['Close'].pct_change().dropna() * 100
                
                # Calculate VaR at different confidence levels
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                axes[1, 0].hist(returns, bins=20, alpha=0.7, color='#2E8B57', edgecolor='black')
                axes[1, 0].axvline(var_95, color='orange', linestyle='--', linewidth=2,
                                    label=f'95% VaR: {var_95:.2f}%')
                axes[1, 0].axvline(var_99, color='red', linestyle='--', linewidth=2,
                                    label=f'99% VaR: {var_99:.2f}%')
                axes[1, 0].set_title('Value at Risk (VaR) Analysis', fontweight='bold')
                axes[1, 0].set_xlabel('Daily Return (%)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # Risk summary
            axes[1, 1].axis('off')
            
            # Determine risk level
            if current_vol < 15:
                risk_level = "LOW"
                risk_color = "green"
            elif current_vol < 25:
                risk_level = "MEDIUM"
                risk_color = "orange"
            else:
                risk_level = "HIGH"
                risk_color = "red"
            
            risk_summary = [
                ['Risk Assessment', 'Result'],
                ['Overall Risk Level', risk_level],
                ['Volatility Rating', f"{current_vol:.1f}%"],
                ['Risk Category', f"{risk_level} Risk Investment"],
                ['Recommendation', 'Suitable for aggressive investors' if risk_level == 'HIGH' 
                    else 'Suitable for moderate investors' if risk_level == 'MEDIUM' 
                    else 'Suitable for conservative investors']
            ]
            
            table = axes[1, 1].table(cellText=risk_summary, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            # Color header row
            for i in range(len(risk_summary[0])):
                table[(0, i)].set_facecolor('#8B0000')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color risk level cell
            table[(1, 1)].set_facecolor(risk_color)
            table[(1, 1)].set_text_props(weight='bold', color='white')
            
            axes[1, 1].set_title('Risk Assessment Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating risk assessment chart: {str(e)}")
            plt.close()
            return ""

    def _create_data_quality_chart(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame) -> str:
        """Create data quality assessment chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Data completeness analysis
            completeness_data = []
            if news_df is not None and not news_df.empty:
                news_completeness = (1 - news_df.isnull().sum() / len(news_df)) * 100
                completeness_data.append(('News', news_completeness.mean()))
            
            if reddit_df is not None and not reddit_df.empty:
                reddit_completeness = (1 - reddit_df.isnull().sum() / len(reddit_df)) * 100
                completeness_data.append(('Social Media', reddit_completeness.mean()))
            
            if completeness_data:
                sources, completeness = zip(*completeness_data)
                bars = axes[0, 0].bar(sources, completeness, color=['#1f4e79', '#2E8B57'], alpha=0.8)
                axes[0, 0].set_title('Data Completeness (%)', fontweight='bold')
                axes[0, 0].set_ylabel('Completeness (%)')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, comp in zip(bars, completeness):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'{comp:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Missing data heatmap for news
            if news_df is not None and not news_df.empty:
                missing_data = news_df.isnull().sum()
                if missing_data.sum() > 0:
                    axes[0, 1].bar(range(len(missing_data)), missing_data.values, color='red', alpha=0.7)
                    axes[0, 1].set_title('Missing Data - News', fontweight='bold')
                    axes[0, 1].set_xlabel('Columns')
                    axes[0, 1].set_ylabel('Missing Values')
                    axes[0, 1].set_xticks(range(len(missing_data)))
                    axes[0, 1].set_xticklabels(missing_data.index, rotation=45, ha='right')
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No Missing Data\nin News Dataset', ha='center', va='center',
                                    transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold')
                    axes[0, 1].set_title('Missing Data - News', fontweight='bold')
            
            # Missing data heatmap for reddit
            if reddit_df is not None and not reddit_df.empty:
                missing_data = reddit_df.isnull().sum()
                if missing_data.sum() > 0:
                    axes[1, 0].bar(range(len(missing_data)), missing_data.values, color='orange', alpha=0.7)
                    axes[1, 0].set_title('Missing Data - Social Media', fontweight='bold')
                    axes[1, 0].set_xlabel('Columns')
                    axes[1, 0].set_ylabel('Missing Values')
                    axes[1, 0].set_xticks(range(len(missing_data)))
                    axes[1, 0].set_xticklabels(missing_data.index, rotation=45, ha='right')
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Missing Data\nin Social Media Dataset', ha='center', va='center',
                                    transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold')
                    axes[1, 0].set_title('Missing Data - Social Media', fontweight='bold')
            
            # Data quality summary
            axes[1, 1].axis('off')
            quality_metrics = [['Quality Metric', 'News', 'Social Media']]
            
            if news_df is not None and reddit_df is not None:
                quality_metrics.extend([
                    ['Total Records', f"{len(news_df):,}", f"{len(reddit_df):,}"],
                    ['Columns', f"{len(news_df.columns)}", f"{len(reddit_df.columns)}"],
                    ['Missing Data %', f"{(news_df.isnull().sum().sum() / news_df.size * 100):.1f}%",
                        f"{(reddit_df.isnull().sum().sum() / reddit_df.size * 100):.1f}%"],
                    ['Data Types', f"{len(news_df.dtypes.unique())}", f"{len(reddit_df.dtypes.unique())}"]
                ])
            
            table = axes[1, 1].table(cellText=quality_metrics, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            # Color header row
            for i in range(len(quality_metrics[0])):
                table[(0, i)].set_facecolor('#1f4e79')
                table[(0, i)].set_text_props(weight='bold', color='white')
            axes[1, 1].set_title('Data Quality Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating data quality chart: {str(e)}")
            plt.close()
            return ""

    def create_chart_for_pdf(self, stock_data: Dict[str, Any], chart_type: str = 'price') -> str:
        """Create individual charts for PDF inclusion (kept for backward compatibility)"""
        try:
            plt.style.use('seaborn')
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if chart_type == 'price' and 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                ax.plot(hist_data.index, hist_data['Close'], linewidth=2, color='#1f4e79')
                ax.set_title('Stock Price Trend (Last 30 Days)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                
            elif chart_type == 'returns' and 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                returns = hist_data['Close'].pct_change().dropna() * 100
                ax.hist(returns, bins=15, alpha=0.7, color='#2E8B57', edgecolor='black')
                ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Daily Return (%)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
            elif chart_type == 'volatility':
                # Simple volatility visualization
                current_vol = stock_data.get('volatility', 0.25) * 100
                categories = ['Low Risk\n(0-15%)', 'Medium Risk\n(15-25%)', 'High Risk\n(25%+)']
                values = [15, 25, 35]
                colors = ['green', 'orange', 'red']
                
                bars = ax.bar(categories, values, color=colors, alpha=0.6, edgecolor='black')
                ax.axhline(y=current_vol, color='blue', linestyle='--', linewidth=2, 
                            label=f'Current: {current_vol:.1f}%')
                ax.set_title('Risk Assessment - Volatility Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel('Volatility (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to temporary file for ReportLab
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            plt.close()
            return ""
        
    def _create_correlation_analysis_chart(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame, stock_data: Dict[str, Any]) -> str:
        """Create correlation analysis between sentiment, volume, and price movements"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Prepare data for correlation analysis
            dates = []
            news_sentiment_scores = []
            reddit_sentiment_scores = []
            news_volumes = []
            reddit_volumes = []
            price_changes = []
            
            # Get historical data
            if 'historical_data' in stock_data and not stock_data['historical_data'].empty:
                hist_data = stock_data['historical_data']
                daily_returns = hist_data['Close'].pct_change().fillna(0) * 100
                
                # Create mock correlation data (in real implementation, you'd align by actual dates)
                n_days = min(len(hist_data), 20)  # Last 20 days
                for i in range(n_days):
                    dates.append(hist_data.index[-n_days + i])
                    news_sentiment_scores.append(np.random.normal(0, 0.3))  # Mock sentiment
                    reddit_sentiment_scores.append(np.random.normal(0, 0.4))  # Mock sentiment
                    news_volumes.append(np.random.poisson(5))  # Mock volume
                    reddit_volumes.append(np.random.poisson(8))  # Mock volume
                    price_changes.append(daily_returns.iloc[-n_days + i])
            
            # Chart 1: Sentiment vs Price Movement
            if dates and price_changes:
                combined_sentiment = [(n + r) / 2 for n, r in zip(news_sentiment_scores, reddit_sentiment_scores)]
                
                axes[0, 0].scatter(combined_sentiment, price_changes, alpha=0.7, s=60, color='#1f4e79')
                axes[0, 0].set_xlabel('Combined Sentiment Score')
                axes[0, 0].set_ylabel('Daily Price Change (%)')
                axes[0, 0].set_title('Sentiment vs Price Movement Correlation', fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add trend line
                if len(combined_sentiment) > 1:
                    z = np.polyfit(combined_sentiment, price_changes, 1)
                    p = np.poly1d(z)
                    axes[0, 0].plot(sorted(combined_sentiment), p(sorted(combined_sentiment)), 
                                    "r--", alpha=0.8, linewidth=2)
            
            # Chart 2: Volume vs Volatility
            if dates and news_volumes and reddit_volumes:
                total_volumes = [n + r for n, r in zip(news_volumes, reddit_volumes)]
                volatility_proxy = [abs(pc) for pc in price_changes]
                
                axes[0, 1].scatter(total_volumes, volatility_proxy, alpha=0.7, s=60, color='#2E8B57')
                axes[0, 1].set_xlabel('Total Discussion Volume')
                axes[0, 1].set_ylabel('Price Volatility (|Daily Change|)')
                axes[0, 1].set_title('Discussion Volume vs Volatility', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Chart 3: News vs Social Media Sentiment Comparison
            if news_sentiment_scores and reddit_sentiment_scores:
                axes[1, 0].scatter(news_sentiment_scores, reddit_sentiment_scores, 
                                alpha=0.7, s=60, color='orange')
                axes[1, 0].set_xlabel('News Sentiment Score')
                axes[1, 0].set_ylabel('Social Media Sentiment Score')
                axes[1, 0].set_title('News vs Social Media Sentiment Correlation', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add diagonal line for perfect correlation
                min_val = min(min(news_sentiment_scores), min(reddit_sentiment_scores))
                max_val = max(max(news_sentiment_scores), max(reddit_sentiment_scores))
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
                axes[1, 0].legend()
            
            # Chart 4: Correlation Matrix Heatmap
            correlation_data = {
                'News Sentiment': news_sentiment_scores,
                'Social Sentiment': reddit_sentiment_scores,
                'News Volume': news_volumes,
                'Social Volume': reddit_volumes,
                'Price Change': price_changes
            }
            
            if all(correlation_data.values()):
                corr_df = pd.DataFrame(correlation_data)
                correlation_matrix = corr_df.corr()
                
                im = axes[1, 1].imshow(correlation_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
                axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
                axes[1, 1].set_yticks(range(len(correlation_matrix.columns)))
                axes[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
                axes[1, 1].set_yticklabels(correlation_matrix.columns)
                axes[1, 1].set_title('Correlation Matrix Heatmap', fontweight='bold')
                
                # Add correlation values as text
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        text = axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                            ha="center", va="center", color="black", fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating correlation analysis chart: {str(e)}")
            plt.close()
            return ""

    def create_enhanced_comprehensive_charts(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame, 
                                        stock_data: Dict[str, Any], sentiment_data: Dict[str, Any],
                                        company_info: Dict[str, str]) -> List[str]:
        """Create the enhanced comprehensive set of charts including all new visualizations"""
        chart_paths = []
        
        try:
            logger.info("Creating enhanced comprehensive chart set...")
            
            # Original charts (keeping existing functionality) - WITH SAFE HANDLING
            try:
                original_charts = self.create_comprehensive_charts(news_df, reddit_df, stock_data, {})
                # Ensure original_charts is a list before extending
                if isinstance(original_charts, list):
                    chart_paths.extend(original_charts)
                elif original_charts:  # If it's a single path string
                    chart_paths.append(str(original_charts))
                logger.info(f"Added {len(chart_paths)} original charts")
            except Exception as e:
                logger.warning(f"Could not create original charts: {e}")
            
            # Enhanced visualizations
            enhanced_charts = []
            
            # 3. Market Comparison
            try:
                market_comparison = self._create_market_comparison_chart(stock_data, company_info)
                if market_comparison:
                    enhanced_charts.append(market_comparison)
                    logger.info("Created market comparison chart")
            except Exception as e:
                logger.warning(f"Could not create market comparison chart: {e}")
            
            # 4. Technical Analysis
            try:
                technical_analysis = self._create_technical_analysis_chart(stock_data)
                if technical_analysis:
                    enhanced_charts.append(technical_analysis)
                    logger.info("Created technical analysis chart")
            except Exception as e:
                logger.warning(f"Could not create technical analysis chart: {e}")
            
            # 5. News Impact Analysis
            try:
                news_impact = self._create_news_impact_analysis_chart(news_df, stock_data)
                if news_impact:
                    enhanced_charts.append(news_impact)
                    logger.info("Created news impact analysis chart")
            except Exception as e:
                logger.warning(f"Could not create news impact analysis chart: {e}")
            
            # 6. Social Media Engagement Analysis
            try:
                social_engagement = self._create_social_media_engagement_chart(reddit_df)
                if social_engagement:
                    enhanced_charts.append(social_engagement)
                    logger.info("Created social media engagement chart")
            except Exception as e:
                logger.warning(f"Could not create social media engagement chart: {e}")
            
            # 7. Comprehensive Dashboard (Master Chart)
            try:
                dashboard = self._create_comprehensive_dashboard(news_df, reddit_df, stock_data, 
                                                            sentiment_data, company_info)
                if dashboard:
                    enhanced_charts.append(dashboard)
                    logger.info("Created comprehensive dashboard")
            except Exception as e:
                logger.warning(f"Could not create comprehensive dashboard: {e}")
            
            # Safely extend with enhanced charts
            chart_paths.extend(enhanced_charts)
            
            logger.info(f"Successfully created {len(chart_paths)} total charts ({len(enhanced_charts)} enhanced)")
            return chart_paths
            
        except Exception as e:
            logger.error(f"Error creating enhanced comprehensive charts: {str(e)}")
            return chart_paths  # Return what we have so far

    def create_advanced_pdf_with_enhanced_charts(self, company_info: Dict[str, str], stock_data: Dict[str, Any], 
                                               sentiment_data: Dict[str, Any], news_df: pd.DataFrame,
                                               reddit_df: pd.DataFrame, chart_paths: List[str]) -> Optional[str]:
        """Create an advanced PDF report that includes all the enhanced visualizations"""
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import tempfile
            import os

            # Generate filename
            ticker = company_info.get('ticker', 'analysis')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"advanced_analysis_report_{ticker}_{timestamp}.pdf"
            temp_path = os.path.join(tempfile.gettempdir(), filename)

            doc = SimpleDocTemplate(temp_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            styles.add(ParagraphStyle(
                name="ChartHeader", 
                fontSize=16, 
                leading=20, 
                spaceAfter=12, 
                textColor=colors.HexColor("#1f4e79"),
                fontName='Helvetica-Bold'
            ))
            
            elements = []

            # Title page
            company_name = company_info.get('company_name', 'Market Analysis')
            ticker_symbol = company_info.get('ticker', 'N/A')
            title = f"Advanced Financial Analysis Report\n{company_name} ({ticker_symbol})"
            elements.append(Paragraph(title, styles["Title"]))
            elements.append(Spacer(1, 30))

            # Report overview
            overview_text = f"""
            This comprehensive report presents advanced financial analysis using multiple visualization techniques
            and data sources. The analysis includes {len(news_df)} news articles and {len(reddit_df)} social media posts,
            providing insights into market sentiment, technical indicators, risk assessment, and portfolio optimization.
            
            Report generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            Analysis period: Last 30 days (estimated)
            """
            
            elements.append(Paragraph("Executive Overview", styles["Heading1"]))
            elements.append(Paragraph(overview_text, styles["Normal"]))
            elements.append(PageBreak())

            # Chart descriptions for better context
            chart_descriptions = { 
                0: "Stock Price and Technical Indicators - Price trends and moving averages",
                1: "Risk Assessment - Volatility analysis and risk categorization",
                2: "Correlation Analysis - Relationships between sentiment, volume, and price movements",
                3: "Advanced Sentiment Timeline - Temporal sentiment patterns with volatility analysis",
                4: "Market Comparison - Performance benchmarking against market indices",
                5: "Technical Analysis - RSI, MACD, and Bollinger Bands analysis",
                6: "News Impact Analysis - News volume correlation with price movements",
                7: "Social Media Engagement - Engagement patterns and sentiment correlation",
                8: "Comprehensive Dashboard - Executive summary of all key metrics"
            }

            # Add charts with descriptions
            for i, chart_path in enumerate(chart_paths):
                if chart_path and os.path.exists(chart_path):
                    try:
                        # Chart title and description
                        chart_title = f"Analysis Chart {i+1}"
                        chart_desc = chart_descriptions.get(i, "Advanced financial analysis visualization")
                        
                        elements.append(Paragraph(chart_title, styles["ChartHeader"]))
                        elements.append(Paragraph(chart_desc, styles["Normal"]))
                        elements.append(Spacer(1, 12))
                        
                        # Add chart image
                        chart_img = Image(chart_path, width=7*inch, height=5*inch, kind='proportional')
                        elements.append(chart_img)
                        elements.append(Spacer(1, 20))
                        
                        # Add page break after every 2 charts (except the last one)
                        if (i + 1) % 2 == 0 and i < len(chart_paths) - 1:
                            elements.append(PageBreak())
                            
                    except Exception as e:
                        logger.warning(f"Could not add chart {i+1} to PDF: {str(e)}")
                        elements.append(Paragraph(f"Chart {i+1}: Unable to display", styles["Normal"]))
                        elements.append(Spacer(1, 20))

            # Analysis summary and recommendations
            elements.append(PageBreak())
            elements.append(Paragraph("Key Insights and Recommendations", styles["Heading1"]))
            
            # Generate insights based on the data
            overall_sentiment = sentiment_data.get('combined_sentiment', {})
            sentiment_score = overall_sentiment.get('score', 0)
            sentiment_label = overall_sentiment.get('label', 'neutral')
            current_price = stock_data.get('current_price', 0)
            volatility = stock_data.get('volatility', 0)
            
            insights_text = f"""
            SENTIMENT ANALYSIS INSIGHTS:
            • Overall market sentiment is {sentiment_label.upper()} with a score of {sentiment_score:+.3f}
            • Analysis of {len(news_df)} news articles and {len(reddit_df)} social media posts
            • Sentiment correlation with price movements shows {'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'} relationship
            
            TECHNICAL ANALYSIS INSIGHTS:
            • Current stock price: ${current_price:.2f}
            • Annualized volatility: {volatility:.1%}
            • Risk classification: {'HIGH' if volatility > 0.3 else 'MEDIUM' if volatility > 0.2 else 'LOW'} risk
            
            PORTFOLIO RECOMMENDATIONS:
            • Recommended allocation: {min(15, 100/(1 + volatility*10)):.0f}% of equity portfolio
            • Risk-adjusted position sizing based on volatility analysis
            • Consider diversification across multiple asset classes
            
            MONITORING RECOMMENDATIONS:
            • Track sentiment changes weekly
            • Monitor technical indicator convergence/divergence
            • Watch for correlation breakdown between sentiment and price
            • Review portfolio allocation monthly based on volatility changes
            """
            
            elements.append(Paragraph(insights_text, styles["Normal"]))
            elements.append(Spacer(1, 20))

            # Disclaimer
            elements.append(Paragraph("Risk Disclaimer", styles["Heading2"]))
            disclaimer = """
            This report is generated using AI analysis of publicly available financial data, news articles, 
            and social media content. All analysis, recommendations, and insights are for informational 
            purposes only and do not constitute professional financial advice, investment recommendations, 
            or solicitation to buy or sell securities.

            Investment decisions should be made only after consulting with qualified financial professionals 
            and conducting independent research. Past performance and sentiment analysis do not guarantee 
            future results. All investments carry risk of loss.
            """
            
            elements.append(Paragraph(disclaimer, styles["Normal"]))

            # Build the PDF
            doc.build(elements)
            logger.info(f"Advanced PDF report with enhanced charts created: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating advanced PDF report: {str(e)}")
            return None

    def get_chart_creation_summary(self) -> Dict[str, Any]:
        """Get a summary of all available chart types and their purposes"""
        return {
            "original_charts": {
                "volume_comparison": "Compares data volume between news and social media sources",
                "text_length_analysis": "Analyzes text length distributions and patterns", 
                "word_frequency": "Shows most frequent words and creates word clouds",
                "stock_price_analysis": "Stock price trends with moving averages and metrics",
                "returns_distribution": "Daily returns histogram and cumulative returns",
                "risk_assessment": "Volatility analysis and risk categorization",
                "data_quality": "Data completeness and missing value analysis",
                "timeline_analysis": "Temporal patterns in data availability"
            },
            "enhanced_charts": {
                "correlation_analysis": "Relationships between sentiment, volume, and price movements",
                "advanced_sentiment_timeline": "Sentiment evolution over time with volatility",
                "market_comparison": "Performance vs market indices and benchmarks",
                "technical_analysis": "RSI, MACD, Bollinger Bands, and other indicators",
                "news_impact_analysis": "News volume correlation with price movements",
                "social_engagement": "Social media engagement patterns and metrics",
                "comprehensive_dashboard": "Executive summary dashboard with all key metrics"
            },
            "total_visualizations": 16,
            "pdf_integration": "All charts can be integrated into professional PDF reports",
            "email_distribution": "Charts can be automatically distributed via email",
            "customization": "Charts adapt to available data and company-specific information"
        }

    def _create_advanced_sentiment_timeline(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame, 
                                             sentiment_data: Dict[str, Any]) -> str:
        """Create advanced sentiment timeline with moving averages and trend analysis"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # Generate mock time series data (in real implementation, extract from actual dates)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Mock sentiment time series
            news_sentiment_ts = np.random.normal(0, 0.3, 30)
            reddit_sentiment_ts = np.random.normal(0, 0.4, 30)
            combined_sentiment_ts = (news_sentiment_ts + reddit_sentiment_ts) / 2
            
            # Add some trend and noise
            trend = np.linspace(-0.2, 0.3, 30)
            news_sentiment_ts += trend + np.random.normal(0, 0.1, 30)
            reddit_sentiment_ts += trend * 0.8 + np.random.normal(0, 0.15, 30)
            combined_sentiment_ts = (news_sentiment_ts + reddit_sentiment_ts) / 2
            
            # Chart 1: Sentiment Timeline with Moving Averages
            axes[0].plot(dates, news_sentiment_ts, label='News Sentiment', color='#1f4e79', alpha=0.7, linewidth=1)
            axes[0].plot(dates, reddit_sentiment_ts, label='Social Media Sentiment', color='#2E8B57', alpha=0.7, linewidth=1)
            axes[0].plot(dates, combined_sentiment_ts, label='Combined Sentiment', color='red', linewidth=2)
            
            # Add moving averages
            if len(combined_sentiment_ts) >= 7:
                ma_7 = pd.Series(combined_sentiment_ts).rolling(window=7).mean()
                axes[0].plot(dates, ma_7, label='7-day MA', color='orange', linewidth=2, linestyle='--')
            
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0].fill_between(dates, 0, combined_sentiment_ts, 
                                where=(combined_sentiment_ts > 0), color='green', alpha=0.2, label='Positive Periods')
            axes[0].fill_between(dates, 0, combined_sentiment_ts, 
                                where=(combined_sentiment_ts <= 0), color='red', alpha=0.2, label='Negative Periods')
            
            axes[0].set_title('Sentiment Timeline with Moving Averages', fontweight='bold', fontsize=14)
            axes[0].set_ylabel('Sentiment Score')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Chart 2: Sentiment Volatility
            sentiment_volatility = pd.Series(combined_sentiment_ts).rolling(window=3).std().fillna(0)
            axes[1].fill_between(dates, sentiment_volatility, alpha=0.6, color='purple')
            axes[1].plot(dates, sentiment_volatility, color='darkpurple', linewidth=2)
            axes[1].set_title('Sentiment Volatility (3-day Rolling Std)', fontweight='bold', fontsize=14)
            axes[1].set_ylabel('Volatility')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            # Chart 3: Sentiment Distribution Histogram
            axes[2].hist(combined_sentiment_ts, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            axes[2].axvline(np.mean(combined_sentiment_ts), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(combined_sentiment_ts):.3f}')
            axes[2].axvline(np.median(combined_sentiment_ts), color='green', linestyle='--', linewidth=2,
                            label=f'Median: {np.median(combined_sentiment_ts):.3f}')
            axes[2].set_title('Sentiment Score Distribution', fontweight='bold', fontsize=14)
            axes[2].set_xlabel('Sentiment Score')
            axes[2].set_ylabel('Density')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating advanced sentiment timeline: {str(e)}")
            plt.close()
            return ""

    def _create_market_comparison_chart(self, stock_data: Dict[str, Any], company_info: Dict[str, str]) -> str:
        """Create market comparison chart with sector benchmarks"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Market indices for comparison (mock data)
            indices = {
                'S&P 500': {'return': 8.5, 'volatility': 16.2, 'pe': 22.1},
                'NASDAQ': {'return': 12.3, 'volatility': 20.8, 'pe': 28.5},
                'Russell 2000': {'return': 6.8, 'volatility': 24.1, 'pe': 18.9},
                'Sector Average': {'return': 9.2, 'volatility': 18.5, 'pe': 19.7}
            }
            
            current_stock = {
                'return': np.random.uniform(5, 15),  # Mock return
                'volatility': stock_data.get('volatility', 0.25) * 100,
                'pe': stock_data.get('pe_ratio', 20) if stock_data.get('pe_ratio') else 20
            }
            
            # Chart 1: Risk-Return Comparison
            index_names = list(indices.keys()) + [company_info.get('ticker', 'Stock')]
            returns = [data['return'] for data in indices.values()] + [current_stock['return']]
            volatilities = [data['volatility'] for data in indices.values()] + [current_stock['volatility']]
            
            # Different colors for indices vs our stock
            colors = ['lightblue'] * len(indices) + ['red']
            sizes = [100] * len(indices) + [200]
            
            scatter = axes[0, 0].scatter(volatilities, returns, s=sizes, c=colors, alpha=0.7, edgecolors='black')
            
            for i, name in enumerate(index_names):
                axes[0, 0].annotate(name, (volatilities[i], returns[i]), 
                                    xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            axes[0, 0].set_xlabel('Volatility (%)')
            axes[0, 0].set_ylabel('Expected Return (%)')
            axes[0, 0].set_title('Risk-Return Comparison vs Market Indices', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Chart 2: P/E Ratio Comparison
            pe_ratios = [data['pe'] for data in indices.values()] + [current_stock['pe']]
            bars = axes[0, 1].bar(index_names, pe_ratios, color=colors, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('P/E Ratio Comparison', fontweight='bold')
            axes[0, 1].set_ylabel('P/E Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, pe in zip(bars, pe_ratios):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{pe:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Chart 3: Performance Percentile Ranking
            metrics = ['Return', 'Low Volatility', 'Reasonable P/E']
            
            # Calculate percentiles (higher is better for return, lower is better for volatility and P/E)
            return_percentile = (sum(1 for r in returns[:-1] if r < current_stock['return']) / len(returns[:-1])) * 100
            vol_percentile = (sum(1 for v in volatilities[:-1] if v > current_stock['volatility']) / len(volatilities[:-1])) * 100
            pe_percentile = (sum(1 for pe in pe_ratios[:-1] if pe > current_stock['pe']) / len(pe_ratios[:-1])) * 100
            
            percentiles = [return_percentile, vol_percentile, pe_percentile]
            colors_perc = ['green' if p > 50 else 'red' for p in percentiles]
            
            bars = axes[1, 0].barh(metrics, percentiles, color=colors_perc, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Percentile Ranking (%)')
            axes[1, 0].set_title('Performance Percentile vs Benchmarks', fontweight='bold')
            axes[1, 0].axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50th Percentile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add percentile labels
            for bar, perc in zip(bars, percentiles):
                width = bar.get_width()
                axes[1, 0].text(width + 1, bar.get_y() + bar.get_height()/2.,
                                f'{perc:.0f}%', ha='left', va='center', fontweight='bold')
            
            # Chart 4: Market Cap Category Analysis
            axes[1, 1].axis('off')
            
            # Determine market cap category
            market_cap = stock_data.get('market_cap', 0)
            if market_cap > 200e9:
                cap_category = "Large Cap (>$200B)"
            elif market_cap > 10e9:
                cap_category = "Mid Cap ($10B-$200B)"
            elif market_cap > 2e9:
                cap_category = "Small Cap ($2B-$10B)"
            elif market_cap > 0:
                cap_category = "Micro Cap (<$2B)"
            else:
                cap_category = "N/A (Private/Unknown)"
            
            comparison_data = [
                ['Comparison Metric', 'Value', 'Market Position'],
                ['Market Cap Category', cap_category, ''],
                ['Return Percentile', f'{return_percentile:.0f}%', 'vs Benchmarks'],
                ['Volatility Percentile', f'{vol_percentile:.0f}%', 'Lower = Better'],
                ['P/E Percentile', f'{pe_percentile:.0f}%', 'Depends on Strategy'],
                ['Overall Assessment', 
                    'Above Average' if np.mean(percentiles) > 60 else 
                    'Average' if np.mean(percentiles) > 40 else 'Below Average', '']
            ]
            
            table = axes[1, 1].table(cellText=comparison_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Color header row
            for i in range(len(comparison_data[0])):
                table[(0, i)].set_facecolor('#1f4e79')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            axes[1, 1].set_title('Market Position Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating market comparison chart: {str(e)}")
            plt.close()
            return ""

    def _create_technical_analysis_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create technical analysis chart with multiple indicators"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            if 'historical_data' not in stock_data or stock_data['historical_data'].empty:
                # Create mock data for demonstration
                dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
                base_price = 150
                prices = []
                for i in range(60):
                    change = np.random.normal(0, 0.02)
                    base_price *= (1 + change)
                    prices.append(base_price)
                
                hist_data = pd.DataFrame({
                    'Close': prices,
                    'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    'Volume': [np.random.randint(1000000, 5000000) for _ in range(60)]
                }, index=dates)
            else:
                hist_data = stock_data['historical_data'].copy()
            
            # Chart 1: Price with Bollinger Bands and Moving Averages
            close_prices = hist_data['Close']
            
            # Calculate technical indicators
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=min(50, len(close_prices))).mean()
            
            # Bollinger Bands
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = sma_20 + (bb_std * 2)
            bb_lower = sma_20 - (bb_std * 2)
            
            # Plot price and indicators
            axes[0].plot(hist_data.index, close_prices, label='Close Price', color='black', linewidth=2)
            axes[0].plot(hist_data.index, sma_20, label='20-day SMA', color='blue', linewidth=1.5)
            axes[0].plot(hist_data.index, sma_50, label='50-day SMA', color='red', linewidth=1.5)
            
            # Bollinger Bands
            axes[0].fill_between(hist_data.index, bb_lower, bb_upper, alpha=0.2, color='gray', label='Bollinger Bands')
            axes[0].plot(hist_data.index, bb_upper, color='gray', linewidth=1, linestyle='--')
            axes[0].plot(hist_data.index, bb_lower, color='gray', linewidth=1, linestyle='--')
            
            axes[0].set_title('Technical Analysis - Price with Moving Averages & Bollinger Bands', fontweight='bold')
            axes[0].set_ylabel('Price ($)')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Chart 2: RSI (Relative Strength Index)
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calculate_rsi(close_prices)
            
            axes[1].plot(hist_data.index, rsi, color='purple', linewidth=2)
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].axhline(y=50, color='black', linestyle='-', alpha=0.3)
            axes[1].fill_between(hist_data.index, 70, 100, alpha=0.1, color='red')
            axes[1].fill_between(hist_data.index, 0, 30, alpha=0.1, color='green')
            
            axes[1].set_title('Relative Strength Index (RSI)', fontweight='bold')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            # Chart 3: MACD (Moving Average Convergence Divergence)
            def calculate_macd(prices, fast=12, slow=26, signal=9):
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.ewm(span=signal).mean()
                histogram = macd - signal_line
                return macd, signal_line, histogram
            
            macd, signal_line, histogram = calculate_macd(close_prices)
            
            # MACD histogram
            axes[2].bar(hist_data.index, histogram, color=['green' if h > 0 else 'red' for h in histogram],
                        alpha=0.6, width=0.8)
            
            # MACD and Signal lines
            axes[2].plot(hist_data.index, macd, color='blue', linewidth=2, label='MACD')
            axes[2].plot(hist_data.index, signal_line, color='red', linewidth=1.5, label='Signal Line')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating technical analysis chart: {str(e)}")
            plt.close()
            return ""

    def generate_summary_report(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame, 
                          company_info: Dict[str, str]) -> str:
        """Generate comprehensive AI summary report from both datasets"""
        try:
            # Handle empty or None DataFrames
            if news_df is None:
                news_df = pd.DataFrame()
            if reddit_df is None:
                reddit_df = pd.DataFrame()
                
            # Get text columns with proper error handling
            news_text_col = None
            reddit_text_col = None
            
            if not news_df.empty:
                news_text_col = self.identify_text_column(news_df)
            if not reddit_df.empty:
                reddit_text_col = self.identify_text_column(reddit_df)
            
            # Sample texts from both sources
            news_sample = []
            reddit_sample = []
            
            if news_text_col and news_text_col in news_df.columns:
                news_sample = news_df[news_text_col].dropna().head(10).tolist()
            
            if reddit_text_col and reddit_text_col in reddit_df.columns:
                reddit_sample = reddit_df[reddit_text_col].dropna().head(10).tolist()
            
            # Handle case where no text samples are available
            if not news_sample and not reddit_sample:
                return "Unable to generate summary report: No text content available for analysis."
            
            # Prepare context
            company_context = "Market Analysis"
            if company_info.get('company_name'):
                company_context = f"Company: {company_info['company_name']}"
                if company_info.get('ticker'):
                    company_context += f" ({company_info['ticker']})"
            
            # Create comprehensive prompt
            news_text = "\n".join([f"- {text[:200]}..." for text in news_sample[:5]]) if news_sample else "No news articles available"
            reddit_text = "\n".join([f"- {text[:200]}..." for text in reddit_sample[:5]]) if reddit_sample else "No social media posts available"
            
            prompt = f"""
            {company_context}
            
            Please analyze the following news articles and social media posts to create a comprehensive summary report. 
            
            NEWS ARTICLES SAMPLE:
            {news_text}
            
            REDDIT/SOCIAL MEDIA POSTS SAMPLE:
            {reddit_text}
            
            Please provide a detailed analysis covering:
            1. EXECUTIVE SUMMARY (2-3 paragraphs)
            2. KEY THEMES AND TRENDS
            3. NEWS MEDIA PERSPECTIVE
            4. SOCIAL MEDIA SENTIMENT
            5. NOTABLE DEVELOPMENTS
            6. MARKET IMPLICATIONS (if applicable)
            7. CONCLUSION AND OUTLOOK
            
            Make the report professional, informative, and comprehensive (800-1200 words).
            If limited data is available, focus on what can be reasonably analyzed and note the limitations.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst and reporter specializing in comprehensive market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")
            return f"Summary report generation failed due to technical error: {str(e)}. Please check the data sources and try again."

    # Add this method to the EnhancedDualFileAnalysisBot class in backend.py

    def run_comprehensive_analysis_with_distribution(self, news_file: str, reddit_file: str, 
                                              email_recipients: List[str] = None,
                                              create_download_package: bool = True) -> Dict[str, Any]:
        
        # Initialize email handler if config provided
        import os
        from dotenv import load_dotenv
        load_dotenv()
        email_address = os.getenv("EMAIL_ADDRESS")
        app_password = os.getenv("EMAIL_APP_PASSWORD")

        if email_address and app_password:
            self.email_handler = EmailHandler(
                email_address=email_address,
                app_password=app_password
            )
            logger.info("Email handler initialized successfully")

        """Run complete analysis with email distribution and download options"""
        try:
            results = {
                'success': False,
                'company_info': {},
                'pdf_reports': [],
                'sentiment_data': {},
                'email_results': {},
                'download_package': '',
                'error_message': ''
            }
            
            # Load data files
            logger.info("Loading data files...")
            news_df = None
            reddit_df = None
            
            if news_file and os.path.exists(news_file):
                news_df = self.load_file(news_file)
                if news_df is not None:
                    logger.info(f"Loaded news data: {len(news_df)} records")
                else:
                    logger.warning(f"Failed to load news file: {news_file}")
            
            if reddit_file and os.path.exists(reddit_file):
                reddit_df = self.load_file(reddit_file)
                if reddit_df is not None:
                    logger.info(f"Loaded social media data: {len(reddit_df)} records")
                else:
                    logger.warning(f"Failed to load reddit file: {reddit_file}")
            
            if news_df is None and reddit_df is None:
                results['error_message'] = "Failed to load both data files"
                return results
            
            # Handle empty dataframes
            if news_df is not None and news_df.empty:
                logger.warning("News dataframe is empty")
                news_df = None
                
            if reddit_df is not None and reddit_df.empty:
                logger.warning("Reddit dataframe is empty")
                reddit_df = None

            # ✅ Validate inputs before continuing
            valid, message = self.validate_input_files(news_df, reddit_df)
            if not valid:
                logger.error(message)
                return {"success": False, "error_message": message}



            
            # Extract company information
            logger.info("Extracting company information...")
            company_info = {'company_name': '', 'ticker': ''}
            
            if news_df is not None:
                news_text_col = self.identify_text_column(news_df)
                company_info = self.extract_company_ticker(news_df, news_text_col)
            
            if not company_info.get('ticker') and reddit_df is not None:
                reddit_text_col = self.identify_text_column(reddit_df)
                reddit_company_info = self.extract_company_ticker(reddit_df, reddit_text_col)
                if reddit_company_info.get('ticker'):
                    company_info = reddit_company_info
            
            results['company_info'] = company_info
            logger.info(f"Extracted company info: {company_info}")
            
            # Get stock data
            logger.info("Fetching stock data...")
            stock_data = {}
            if company_info.get('ticker'):
                stock_data = self.get_stock_data(company_info['ticker'])
            else:
                logger.warning("No ticker found, using mock data")
                stock_data = self._generate_mock_price_data()
            
            # Comprehensive sentiment analysis
            logger.info("Performing sentiment analysis...")
            sentiment_data = self.analyze_sentiment_comprehensive(
                news_df if news_df is not None else pd.DataFrame(),
                reddit_df if reddit_df is not None else pd.DataFrame()
            )
            results['sentiment_data'] = sentiment_data
            
            # Generate reports
            logger.info("Generating analysis reports...")
            summary_report = self.generate_summary_report(
                news_df if news_df is not None else pd.DataFrame(),
                reddit_df if reddit_df is not None else pd.DataFrame(),
                company_info
            )
            
            # Create PDF reports
            logger.info("Creating PDF reports...")
            pdf_reports = []
            
            # Summary PDF
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                ticker_safe = company_info.get('ticker', 'analysis').replace('/', '_')
                summary_pdf_filename = f"summary_report_{ticker_safe}_{timestamp}.pdf"
                summary_pdf_path = os.path.join(tempfile.gettempdir(), summary_pdf_filename)
                
                summary_pdf = self.create_summary_pdf_report(
                    summary_report, 
                    company_info, 
                    len(news_df) if news_df is not None else 0,
                    len(reddit_df) if reddit_df is not None else 0,
                    summary_pdf_path
                )
                
                if summary_pdf and os.path.exists(summary_pdf):
                    pdf_reports.append(summary_pdf)
                    logger.info(f"Summary PDF created: {summary_pdf}")
                else:
                    logger.warning("Summary PDF creation failed")
                    
            except Exception as e:
                logger.error(f"Error creating summary PDF: {str(e)}")
            
            # Equity research PDF
            try:
                equity_pdf = self.create_equity_research_report(
                    company_info, 
                    stock_data, 
                    sentiment_data,
                    news_df if news_df is not None else pd.DataFrame(),
                    reddit_df if reddit_df is not None else pd.DataFrame()
                )
                
                if equity_pdf and os.path.exists(equity_pdf):
                    pdf_reports.append(equity_pdf)
                    logger.info(f"Equity research PDF created: {equity_pdf}")
                else:
                    logger.warning("Equity research PDF creation failed, creating placeholder")
                    # Create placeholder PDF
                    placeholder_path = self._create_placeholder_equity_pdf(company_info)
                    if placeholder_path:
                        pdf_reports.append(placeholder_path)
                        
            except Exception as e:
                logger.error(f"Error creating equity research PDF: {str(e)}")
                # Create placeholder PDF
                placeholder_path = self._create_placeholder_equity_pdf(company_info)
                if placeholder_path:
                    pdf_reports.append(placeholder_path)

            results['pdf_reports'] = pdf_reports
            logger.info(f"Created {len(pdf_reports)} PDF reports")
            
            # Email distribution
            if email_recipients and self.email_handler and pdf_reports:
                logger.info(f"Distributing reports to {len(email_recipients)} recipients...")
                
                # Create analysis summary for email
                overall_sentiment = sentiment_data.get('combined_sentiment', {})
                analysis_summary = f"""
    Sentiment Analysis Summary:
    - Overall Sentiment: {overall_sentiment.get('label', 'neutral').title()}
    - Sentiment Score: {overall_sentiment.get('score', 0):.3f}
    - News Articles Analyzed: {len(news_df) if news_df is not None else 0}
    - Social Media Posts Analyzed: {len(reddit_df) if reddit_df is not None else 0}
    - Company: {company_info.get('company_name', 'N/A')} ({company_info.get('ticker', 'N/A')})
    """
                
                email_results = self.send_reports_to_recipients(
                    email_recipients, pdf_reports, company_info, analysis_summary
                )
                results['email_results'] = email_results
                
                # Log email results
                for recipient, success in email_results.items():
                    status = "sent successfully" if success else "failed"
                    logger.info(f"Email to {recipient}: {status}")
            else:
                if not email_recipients:
                    logger.info("No email recipients specified")
                elif not self.email_handler:
                    logger.warning("Email handler not configured")
                elif not pdf_reports:
                    logger.warning("No PDF reports to send")
            
            # Create download package
            if create_download_package and pdf_reports:
                logger.info("Creating downloadable reports package...")
                try:
                    download_package = self.create_downloadable_reports_package(pdf_reports, company_info)
                    if download_package and os.path.exists(download_package):
                        results['download_package'] = download_package
                        logger.info(f"Download package created: {download_package}")
                    else:
                        logger.warning("Download package creation failed")
                except Exception as e:
                    logger.error(f"Error creating download package: {str(e)}")
            
            results['success'] = True
            logger.info("Comprehensive analysis completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            results['error_message'] = str(e)
            return results

    def _create_placeholder_equity_pdf(self, company_info: Dict[str, str]) -> str:
        """Create a placeholder PDF when equity research report generation fails"""
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            import tempfile
            import os

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ticker_safe = company_info.get('ticker', 'analysis').replace('/', '_')
            filename = f"equity_research_placeholder_{ticker_safe}_{timestamp}.pdf"
            placeholder_path = os.path.join(tempfile.gettempdir(), filename)

            doc = SimpleDocTemplate(placeholder_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            # Title
            company_name = company_info.get('company_name', 'N/A')
            ticker = company_info.get('ticker', 'N/A')
            title = f"Equity Research Report - {company_name} ({ticker})"
            elements.append(Paragraph(title, styles['Title']))
            elements.append(Spacer(1, 30))

            # Error message
            error_msg = """
            ⚠️ Report Generation Error
            
            An error occurred while generating the comprehensive equity research report. 
            This is a placeholder document to ensure you receive some analysis output.
            
            Possible causes:
            • Chart generation issues
            • Data processing errors
            • PDF formatting problems
            
            Please check the system logs for detailed error information and try running 
            the analysis again. The summary report should still contain valuable insights.
            
            For support, please contact your system administrator.
            """
            
            elements.append(Paragraph(error_msg, styles['Normal']))
            elements.append(Spacer(1, 20))

            # Basic info
            basic_info = f"""
            Analysis Information:
            • Company: {company_name}
            • Ticker: {ticker}
            • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            • Status: Placeholder (Original report generation failed)
            """
            
            elements.append(Paragraph(basic_info, styles['Normal']))

            doc.build(elements)
            logger.info(f"Placeholder equity research PDF created: {placeholder_path}")
            return placeholder_path
            
        except Exception as e:
            logger.error(f"Error creating placeholder PDF: {str(e)}")
            return ""

    def send_reports_to_recipients(self, recipients: List[str], pdf_paths: List[str], 
                            company_info: Dict[str, str], analysis_summary: str = "") -> Dict[str, bool]:
        """Send PDF reports to multiple email recipients"""
        if not self.email_handler:
            logger.error("Email handler not initialized")
            return {recipient: False for recipient in recipients}
        
        results = {}
        company_name = company_info.get('company_name', 'Market Analysis')
        ticker = f" ({company_info['ticker']})" if company_info.get('ticker') else ""
        
        # Email subject and body
        subject = f"Financial Analysis Report - {company_name}{ticker}"
        
        body = f"""
    Dear Recipient,

    Please find attached the comprehensive financial analysis report for {company_name}{ticker}.

    Report Summary:
    - Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
    - Company: {company_name}
    - Stock Ticker: {company_info.get('ticker', 'N/A')}

    {analysis_summary if analysis_summary else 'This report includes detailed sentiment analysis, market data, and investment insights based on news and social media coverage.'}

    The attached reports include:
    - Summary Analysis Report
    - Equity Research Report with Charts
    - Sentiment Analysis

    Please note that this analysis is generated using AI and is for informational purposes only. 
    It should not be considered as personalized investment advice.

    Best regards,
    Financial Analysis System
        """
        
        # Prepare attachments
        attachments = []
        for pdf_path in pdf_paths:
            if pdf_path and os.path.exists(pdf_path):
                filename = os.path.basename(pdf_path)
                attachments.append({'path': pdf_path, 'name': filename})
        
        # Send to each recipient
        for recipient in recipients:
            try:
                success = self.email_handler.send_email_with_attachments(
                    recipient_email=recipient,
                    subject=subject,
                    body=body,
                    attachments=attachments
                )
                results[recipient] = success
                if success:
                    logger.info(f"Successfully sent reports to {recipient}")
                else:
                    logger.error(f"Failed to send reports to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending to {recipient}: {str(e)}")
                results[recipient] = False
        
        return results

    def create_downloadable_reports_package(self, pdf_paths: List[str], company_info: Dict[str, str]) -> str:
        """Create a downloadable ZIP package with all reports"""
        try:
            company_name = company_info.get('company_name', 'Company').replace(' ', '_')
            ticker = company_info.get('ticker', 'STOCK')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            zip_filename = f"Financial_Reports_{company_name}_{ticker}_{timestamp}.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for pdf_path in pdf_paths:
                    if pdf_path and os.path.exists(pdf_path):
                        # Add PDF to zip with clean filename
                        filename = os.path.basename(pdf_path)
                        clean_filename = f"{company_name}_{ticker}_{filename}"
                        zipf.write(pdf_path, clean_filename)
                
                # Add a README file
                readme_content = f"""
    Financial Analysis Reports Package
    ================================

    Company: {company_info.get('company_name', 'N/A')}
    Stock Ticker: {company_info.get('ticker', 'N/A')}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    This package contains:
    1. Summary Analysis Report - Overview of news and social media analysis
    2. Equity Research Report - Detailed financial analysis with charts
    3. Additional analytical charts and visualizations

    IMPORTANT DISCLAIMER:
    These reports are generated using AI analysis of publicly available data.
    The content is for informational purposes only and should not be considered
    as personalized investment advice. Always conduct your own research and
    consult with qualified financial professionals before making investment decisions.

    For questions or support, please contact your system administrator.
                """
                
                zipf.writestr("README.txt", readme_content)
            
            logger.info(f"Created downloadable reports package: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error creating reports package: {str(e)}")
            return ""

    def create_equity_research_report(self, company_info: Dict[str, str], stock_data: Dict[str, Any], 
                                 sentiment_data: Dict[str, Any], news_df: pd.DataFrame,
                                 reddit_df: pd.DataFrame, start_date: str = None,
                                 end_date: str = None) -> Optional[str]:
        """Create a structured Equity Research PDF report with charts"""
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import matplotlib.pyplot as plt
            import tempfile
            import os

            # Generate filename
            ticker = company_info.get('ticker', 'analysis')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"equity_research_{ticker}_{timestamp}.pdf"
            temp_path = os.path.join(tempfile.gettempdir(), filename)

            doc = SimpleDocTemplate(temp_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            styles.add(ParagraphStyle(
                name="SectionHeader", 
                fontSize=14, 
                leading=18, 
                spaceAfter=10, 
                textColor=colors.HexColor("#003366"),
                fontName='Helvetica-Bold'
            ))
            
            elements = []

            # --- Title ---
            company_name = company_info.get('company_name', 'N/A')
            ticker_symbol = company_info.get('ticker', 'N/A')
            title = f"Equity Research Report - {company_name} ({ticker_symbol})"
            elements.append(Paragraph(title, styles["Title"]))
            elements.append(Spacer(1, 20))

            # --- Report Metadata ---
            elements.append(Paragraph("Report Information", styles["SectionHeader"]))
            metadata = [
                ["Ticker", ticker_symbol],
                ["Company", company_name],
                ["Report Date", datetime.now().strftime("%B %d, %Y")],
                ["Analysis Period", f"{start_date or 'Last 30 days'} to {end_date or datetime.now().strftime('%Y-%m-%d')}"],
                ["Data Sources", f"News: {len(news_df)}, Social Media: {len(reddit_df)}"]
            ]
            
            table = Table(metadata, colWidths=[120, 350])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT")
            ]))
            elements.append(table)
            elements.append(Spacer(1, 15))

            # --- Executive Summary ---
            elements.append(Paragraph("Executive Summary", styles["SectionHeader"]))
            
            # Calculate total mentions
            total_mentions = len(news_df) + len(reddit_df)
            overall_sentiment = sentiment_data.get("combined_sentiment", {})
            sentiment_score = overall_sentiment.get('score', 0)
            sentiment_label = overall_sentiment.get('label', 'neutral')
            
            summary_text = f"""
            Based on analysis of {total_mentions} data points across news articles and social media posts, 
            the overall sentiment for {company_name} is {sentiment_label.upper()} with a sentiment score of {sentiment_score:+.3f}.
            
            This analysis incorporates {len(news_df)} news articles and {len(reddit_df)} social media posts 
            to provide comprehensive market sentiment insights.
            """
            
            elements.append(Paragraph(summary_text, styles["Normal"]))
            elements.append(Spacer(1, 15))

            # --- Key Metrics ---
            elements.append(Paragraph("Key Financial Metrics", styles["SectionHeader"]))
            
            current_price = stock_data.get("current_price", 0)
            avg_price = stock_data.get("avg_price_period", 0)
            volatility = stock_data.get("volatility", 0)
            pe_ratio = stock_data.get("pe_ratio", None)
            market_cap = stock_data.get("market_cap", 0)

            metrics_data = [
                ["Metric", "Value"],
                ["Current Price", f"${current_price:.2f}"],
                ["Average Price (Period)", f"${avg_price:.2f}"],
                ["Annualized Volatility", f"{volatility:.1%}"],
                ["P/E Ratio", f"{pe_ratio}" if pe_ratio else "N/A"],
                ["Market Cap", f"${market_cap:,.0f}" if market_cap > 0 else "N/A"],
                ["Data Quality", "Real-time" if stock_data.get('price_data_available') else "Simulated"]
            ]

            table = Table(metrics_data, colWidths=[180, 300])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]))
            elements.append(table)
            elements.append(Spacer(1, 15))

            # --- Investment Recommendation ---
            elements.append(Paragraph("Investment Recommendation", styles["SectionHeader"]))
            
            # Generate recommendation based on sentiment and metrics
            recommendation = "HOLD"
            target_price = current_price * 1.02  # Default 2% target
            
            if sentiment_score > 0.2 and volatility < 0.3:
                recommendation = "BUY"
                target_price = current_price * 1.08
            elif sentiment_score < -0.2 or volatility > 0.4:
                recommendation = "SELL"
                target_price = current_price * 0.95
            
            recommendation_text = f"""
            RECOMMENDATION: {recommendation}
            
            Target Price: ${target_price:.2f}
            Current Price: ${current_price:.2f}
            
            Rationale: Based on sentiment analysis showing {sentiment_label} sentiment ({sentiment_score:+.3f}) 
            and volatility of {volatility:.1%}, we recommend a {recommendation} position. 
            The analysis of {total_mentions} data points suggests {sentiment_label} market perception.
            """
            
            elements.append(Paragraph(recommendation_text, styles["Normal"]))
            elements.append(Spacer(1, 20))

            # --- Charts Section ---
            elements.append(PageBreak())
            elements.append(Paragraph("Analysis Charts", styles["SectionHeader"]))

            chart_paths = []

            # Create charts for the report
            try:
                # Chart 1: Sentiment Distribution
                sentiment_chart_path = self._create_pdf_sentiment_chart(sentiment_data)
                if sentiment_chart_path:
                    chart_paths.append((sentiment_chart_path, "Sentiment Analysis"))

                # Chart 2: Stock Price Analysis
                if stock_data.get('historical_data') is not None:
                    price_chart_path = self._create_pdf_stock_chart(stock_data)
                    if price_chart_path:
                        chart_paths.append((price_chart_path, "Stock Price Analysis"))

                # Chart 3: Risk Assessment
                risk_chart_path = self._create_pdf_risk_chart(stock_data)
                if risk_chart_path:
                    chart_paths.append((risk_chart_path, "Risk Assessment"))

            except Exception as e:
                logger.warning(f"Some charts could not be generated: {str(e)}")

            # Add charts to PDF
            for chart_path, chart_title in chart_paths:
                try:
                    elements.append(Paragraph(chart_title, styles["SectionHeader"]))
                    elements.append(Spacer(1, 10))
                    
                    # Add chart image
                    chart_img = Image(chart_path, width=6*inch, height=4*inch)
                    elements.append(chart_img)
                    elements.append(Spacer(1, 15))
                    
                except Exception as e:
                    logger.warning(f"Could not add chart {chart_title}: {str(e)}")
                    elements.append(Paragraph(f"Chart: {chart_title} (Generation Error)", styles["Normal"]))
                    elements.append(Spacer(1, 10))

            # --- Risk Factors ---
            elements.append(Paragraph("Risk Factors & Considerations", styles["SectionHeader"]))
            
            risk_level = "LOW"
            if volatility > 0.3:
                risk_level = "HIGH"
            elif volatility > 0.2:
                risk_level = "MEDIUM"
                
            risk_text = f"""
            Risk Level: {risk_level}
            
            Key Risk Factors:
            • Market Volatility: {volatility:.1%} annualized volatility indicates {risk_level.lower()} risk
            • Sentiment Risk: Current sentiment is {sentiment_label}, which may shift rapidly
            • Data Limitations: Analysis based on {total_mentions} data points over limited time period
            • AI Analysis: Recommendations are AI-generated and should be verified independently
            
            Investment Considerations:
            • Suitable for {risk_level.lower()} risk tolerance investors
            • Monitor sentiment changes and market developments
            • Consider portfolio diversification
            • Consult with financial advisors for personalized advice
            """
            
            elements.append(Paragraph(risk_text, styles["Normal"]))
            elements.append(Spacer(1, 20))

            # --- Disclaimer ---
            elements.append(Paragraph("Important Disclaimer", styles["SectionHeader"]))
            disclaimer = f"""
            This equity research report is generated using AI analysis of publicly available data including 
            news articles and social media posts. The analysis is based on sentiment analysis, historical 
            price data, and statistical calculations for {company_name} ({ticker_symbol}).

            IMPORTANT NOTICES:
            • This report is for informational purposes only and does not constitute investment advice
            • Past performance and sentiment analysis do not guarantee future results
            • All investments carry risk of loss and may not be suitable for all investors
            • Consult qualified financial professionals before making investment decisions
            • The AI-generated analysis may contain errors or biases
            • Market conditions can change rapidly, affecting the validity of this analysis

            Data Sources: News articles, social media posts, and financial market data as of {datetime.now().strftime('%Y-%m-%d')}
            """
            
            elements.append(Paragraph(disclaimer, styles["Normal"]))

            # --- Build PDF ---
            try:
                doc.build(elements)
                logger.info(f"Equity research report created successfully: {temp_path}")
                
                # Cleanup chart files
                for chart_path, _ in chart_paths:
                    try:
                        if os.path.exists(chart_path):
                            os.unlink(chart_path)
                    except:
                        pass
                
                return temp_path
                
            except Exception as e:
                logger.error(f"Error building PDF document: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error creating equity research report: {str(e)}")
            return None



    def _generate_investment_thesis(self, sentiment_score: float, current_price: float, 
                                    volatility: float, total_mentions: int) -> str:
        """Generate investment thesis based on analysis"""
        if sentiment_score > 0.2:
            return f"""
    Based on our analysis of {total_mentions} data points, the investment thesis is POSITIVE.
    The sentiment analysis reveals strong positive momentum with a score of {sentiment_score:+.3f}.
    Key factors supporting this thesis include favorable news coverage, positive social media
    sentiment, and strong community engagement. However, investors should consider the
    volatility of {volatility:.1%} and conduct additional fundamental analysis."""
        
        elif sentiment_score < -0.2:
            return f"""
    Based on our analysis of {total_mentions} data points, the investment thesis is NEGATIVE.
    The sentiment analysis shows concerning negative momentum with a score of {sentiment_score:+.3f}.
    Key risk factors include unfavorable news coverage, negative social media sentiment, and
    potential market headwinds. The volatility of {volatility:.1%} adds additional risk."""
        
        else:
            return f"""
    Based on our analysis of {total_mentions} data points, the investment thesis is NEUTRAL.
    The sentiment analysis shows mixed signals with a score of {sentiment_score:+.3f}.
    The investment appears fairly valued at current levels, but investors should monitor
    developments closely. The volatility of {volatility:.1%} suggests moderate risk."""

    def create_summary_pdf_report(self, summary_content: str, company_info: Dict[str, str],
                                news_count: int, reddit_count: int, output_file: str) -> str:
        """Create PDF summary report"""
        try:
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,
                textColor=colors.HexColor('#2E8B57')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#1f4e79')
            )
            
            # Title
            company_name = company_info.get('company_name', 'Market Analysis')
            ticker = f" ({company_info['ticker']})" if company_info.get('ticker') else ""
            title = f"Summary Report: {company_name}{ticker}"
            
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            story.append(Paragraph("Report Information", heading_style))
            
            meta_data = [
                ['Generated Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['News Articles Analyzed', str(news_count)],
                ['Social Media Posts Analyzed', str(reddit_count)],
                ['Total Data Points', str(news_count + reddit_count)]
            ]
            
            if company_info.get('company_name'):
                meta_data.insert(1, ['Company', company_info['company_name']])
            if company_info.get('ticker'):
                meta_data.insert(2, ['Stock Ticker', company_info['ticker']])
            
            meta_table = Table(meta_data)
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(meta_table)
            story.append(Spacer(1, 20))
            
            # Summary content
            story.append(Paragraph("Analysis Summary", heading_style))
            
            # Split content into paragraphs
            paragraphs = summary_content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # Footer
            story.append(PageBreak())
            story.append(Paragraph("Disclaimer", heading_style))
            
            disclaimer = """
            This summary report is generated using AI analysis of news articles and social media posts. 
            The insights and conclusions are based on sentiment analysis and text processing algorithms. 
            This report is for informational purposes only and should not be considered as financial advice, 
            investment recommendations, or professional consultation. Always conduct your own research and 
            consult with qualified professionals before making any financial or investment decisions.
            """
            
            story.append(Paragraph(disclaimer, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"Summary PDF report generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create summary PDF report: {str(e)}")
            return ""

    def _create_pdf_stock_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create stock price chart optimized for PDF"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('default')  # Use default style for better PDF compatibility
            
            if 'historical_data' in stock_data:
                hist_data = stock_data['historical_data']
                ax.plot(hist_data.index, hist_data['Close'], linewidth=2, color='#1f4e79', label='Stock Price')
                
                # Add moving average
                if len(hist_data) > 7:
                    ma_7 = hist_data['Close'].rolling(window=7).mean()
                    ax.plot(hist_data.index, ma_7, linewidth=1.5, color='orange', label='7-day MA', alpha=0.8)
                
                ax.set_title('Stock Price Trend with Moving Average', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=45)
                
                # Add current price annotation
                current_price = stock_data.get('current_price', 0)
                if current_price > 0:
                    ax.axhline(y=current_price, color='red', linestyle='--', alpha=0.7)
                    ax.text(0.02, 0.95, f'Current: ${current_price:.2f}', 
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating PDF stock chart: {str(e)}")
            plt.close()
            return ""

    def _create_pdf_sentiment_chart(self, sentiment_data: Dict[str, Any]) -> str:
        """Create sentiment chart optimized for PDF"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            plt.style.use('default')
            
            # News sentiment pie chart
            news_data = sentiment_data['news_sentiment']
            if news_data['positive'] + news_data['negative'] + news_data['neutral'] > 0:
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [news_data['positive'], news_data['negative'], news_data['neutral']]
                colors = ['#2E8B57', '#DC143C', '#FFD700']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('News Sentiment Distribution', fontweight='bold')
            
            # Reddit sentiment pie chart
            reddit_data = sentiment_data['reddit_sentiment']
            if reddit_data['positive'] + reddit_data['negative'] + reddit_data['neutral'] > 0:
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [reddit_data['positive'], reddit_data['negative'], reddit_data['neutral']]
                colors = ['#2E8B57', '#DC143C', '#FFD700']
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Social Media Sentiment Distribution', fontweight='bold')
            
            # Add overall sentiment score
            combined_score = sentiment_data['combined_sentiment']['score']
            combined_label = sentiment_data['combined_sentiment']['label']
            
            fig.suptitle(f'Overall Sentiment: {combined_label.title()} (Score: {combined_score:+.3f})', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating PDF sentiment chart: {str(e)}")
            plt.close()
            return ""

    def _create_pdf_volume_chart(self, news_df: pd.DataFrame, reddit_df: pd.DataFrame) -> str:
        """Create volume comparison chart optimized for PDF"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            plt.style.use('default')
            
            # Bar chart comparison
            categories = ['News Articles', 'Social Media Posts']
            volumes = [
                len(news_df) if news_df is not None else 0,
                len(reddit_df) if reddit_df is not None else 0
            ]
            
            colors = ['#1f4e79', '#2E8B57']
            bars = ax1.bar(categories, volumes, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Data Volume Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Records')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, volume in zip(bars, volumes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(volume):,}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            if sum(volumes) > 0:
                ax2.pie(volumes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Data Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating PDF volume chart: {str(e)}")
            plt.close()
            return ""

    def _create_pdf_risk_chart(self, stock_data: Dict[str, Any]) -> str:
        """Create risk assessment chart optimized for PDF"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('default')
            
            # Risk level comparison
            current_vol = stock_data.get('volatility', 0.25) * 100
            risk_categories = ['Low Risk\n(0-15%)', 'Medium Risk\n(15-25%)', 'High Risk\n(25%+)']
            risk_thresholds = [15, 25, 35]
            colors = ['green', 'orange', 'red']
            
            bars = ax.bar(risk_categories, risk_thresholds, color=colors, alpha=0.6, edgecolor='black')
            ax.axhline(y=current_vol, color='blue', linestyle='--', linewidth=3,
                    label=f'Current Volatility: {current_vol:.1f}%')
            ax.set_title('Risk Assessment - Volatility Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Volatility (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add risk level text
            if current_vol < 15:
                risk_level = "LOW RISK"
                risk_color = "green"
            elif current_vol < 25:
                risk_level = "MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_level = "HIGH RISK"
                risk_color = "red"
            
            ax.text(0.02, 0.95, f'Risk Level: {risk_level}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=risk_color, alpha=0.3))
            
            plt.tight_layout()
            
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating PDF risk chart: {str(e)}")
            plt.close()
            return ""