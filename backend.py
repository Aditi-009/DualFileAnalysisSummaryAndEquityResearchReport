import pandas as pd
import json
import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import re
import warnings
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import yfinance, make it optional
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Will use mock price data.")

def get_available_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get available columns categorized by type"""
    return {
        'text_candidates': [col for col in df.columns if df[col].dtype == 'object'],
        'date_candidates': [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()],
        'numeric_candidates': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
        'all_columns': list(df.columns)
    }

class NewsAndSocialMediaAnalysisBot:
    def __init__(self, api_key: str):
        """Initialize the news and social media analysis bot with OpenAI API key"""
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return None
                
            logger.info(f"Successfully loaded file with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
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
        
        logger.warning("Could not identify text column automatically")
        return None
    
    def identify_source_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the source/publisher column from the dataframe"""
        source_column_candidates = [
            'source', 'publisher', 'publication', 'outlet', 'provider',
            'news_source', 'media_source', 'author', 'site', 'website'
        ]
        
        url_column_candidates = [
            'url', 'link', 'href', 'web_url', 'article_url', 'source_url'
        ]
        
        # Check for exact matches with traditional source columns first
        for col in df.columns:
            if col.lower() in source_column_candidates:
                return col
        
        # Check for partial matches with traditional source columns
        for col in df.columns:
            for candidate in source_column_candidates:
                if candidate in col.lower():
                    return col
        
        # Check for URL columns
        for col in df.columns:
            if col.lower() in url_column_candidates:
                logger.info(f"Found URL column '{col}' - will extract sources from URLs")
                return col
        
        logger.info("No source or URL column found")
        return None
    
    def identify_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the date column from the dataframe"""
        date_column_candidates = [
            'date', 'time', 'timestamp', 'published', 'created',
            'pub_date', 'publish_date', 'datetime', 'created_at'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in date_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in date_column_candidates:
                if candidate in col.lower():
                    return col
        
        # Check for datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        logger.info("No date column found")
        return None
    
    def identify_sentiment_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the sentiment score column from the dataframe"""
        sentiment_column_candidates = [
            'sentiment_score', 'sentiment', 'score', 'polarity', 'compound'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in sentiment_column_candidates:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for candidate in sentiment_column_candidates:
                if candidate in col.lower():
                    return col
        
        logger.info("No sentiment column found")
        return None

    def get_company_name_from_ticker(self, ticker: str) -> str:
        """Map ticker symbols to proper company names"""
        ticker_to_company = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'BRK.A': 'Berkshire Hathaway Inc.',
            'BRK.B': 'Berkshire Hathaway Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'JNJ': 'Johnson & Johnson',
            'XOM': 'Exxon Mobil Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.',
            'HD': 'Home Depot Inc.',
            'CVX': 'Chevron Corporation',
            'LLY': 'Eli Lilly and Company',
            'ABBV': 'AbbVie Inc.',
            'BAC': 'Bank of America Corp.',
            'AVGO': 'Broadcom Inc.',
            'KO': 'Coca-Cola Co.',
            'WMT': 'Walmart Inc.',
            'COST': 'Costco Wholesale Corp.',
            'PEP': 'PepsiCo Inc.',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'MRK': 'Merck & Co. Inc.',
            'ADBE': 'Adobe Inc.',
            'NFLX': 'Netflix Inc.',
            'DIS': 'Walt Disney Co.',
            'ABT': 'Abbott Laboratories',
            'ACN': 'Accenture Plc',
            'CRM': 'Salesforce Inc.',
            'NKE': 'Nike Inc.',
            'TXN': 'Texas Instruments Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'VZ': 'Verizon Communications Inc.',
            'CMCSA': 'Comcast Corp.',
            'DHR': 'Danaher Corp.',
            'NEE': 'NextEra Energy Inc.',
            'INTC': 'Intel Corp.',
            'WFC': 'Wells Fargo & Co.',
            'IBM': 'International Business Machines Corp.',
            'AMD': 'Advanced Micro Devices Inc.',
            'T': 'AT&T Inc.',
            'COP': 'ConocoPhillips',
            'UNP': 'Union Pacific Corp.',
            'HON': 'Honeywell International Inc.',
            'RTX': 'RTX Corp.',
            'PM': 'Philip Morris International Inc.',
            'SPGI': 'S&P Global Inc.',
            'CAT': 'Caterpillar Inc.',
            'GS': 'Goldman Sachs Group Inc.',
            'SCHW': 'Charles Schwab Corp.',
            'AXP': 'American Express Co.',
            'NOW': 'ServiceNow Inc.',
            'ISRG': 'Intuitive Surgical Inc.',
            'BLK': 'BlackRock Inc.',
            'SYK': 'Stryker Corp.',
            'BKNG': 'Booking Holdings Inc.',
            'TJX': 'TJX Companies Inc.',
            'ADP': 'Automatic Data Processing Inc.',
            'GILD': 'Gilead Sciences Inc.',
            'MDLZ': 'Mondelez International Inc.',
            'VRTX': 'Vertex Pharmaceuticals Inc.',
            'MMC': 'Marsh & McLennan Companies Inc.',
            'C': 'Citigroup Inc.',
            'LRCX': 'Lam Research Corp.',
            'ZTS': 'Zoetis Inc.',
            'REGN': 'Regeneron Pharmaceuticals Inc.',
            'CB': 'Chubb Ltd.',
            'PGR': 'Progressive Corp.',
            'TMUS': 'T-Mobile US Inc.',
            'SO': 'Southern Co.',
            'BSX': 'Boston Scientific Corp.',
            'SHW': 'Sherwin-Williams Co.',
            'ETN': 'Eaton Corp. Plc',
            'MU': 'Micron Technology Inc.',
            'DUK': 'Duke Energy Corp.',
            'EQIX': 'Equinix Inc.',
            'AON': 'Aon Plc',
            'APD': 'Air Products and Chemicals Inc.',
            'ICE': 'Intercontinental Exchange Inc.',
            'CL': 'Colgate-Palmolive Co.',
            'CSX': 'CSX Corp.',
            'CME': 'CME Group Inc.',
            'USB': 'U.S. Bancorp',
            'ECL': 'Ecolab Inc.',
            'NSC': 'Norfolk Southern Corp.',
            'ITW': 'Illinois Tool Works Inc.',
            'FDX': 'FedEx Corp.',
            'WM': 'Waste Management Inc.',
            'GD': 'General Dynamics Corp.',
            'EOG': 'EOG Resources Inc.',
            'FCX': 'Freeport-McMoRan Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'PANW': 'Palo Alto Networks Inc.',
            'EL': 'Estee Lauder Companies Inc.',
            'PSA': 'Public Storage',
            'GM': 'General Motors Co.',
            'F': 'Ford Motor Co.',
            'RIVN': 'Rivian Automotive Inc.',
            'LCID': 'Lucid Group Inc.',
            'NIO': 'NIO Inc.',
            'XPEV': 'XPeng Inc.',
            'LI': 'Li Auto Inc.',
        }
        
        return ticker_to_company.get(ticker.upper(), '')

    def extract_source_from_url(self, url: str) -> str:
        """Extract and clean source name from URL"""
        if pd.isna(url) or not url:
            return ""
        
        url = str(url).strip()
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            domain_mapping = {
                # Financial News
                'reuters.com': 'Reuters',
                'bloomberg.com': 'Bloomberg',
                'wsj.com': 'Wall Street Journal',
                'ft.com': 'Financial Times',
                'cnbc.com': 'CNBC',
                'marketwatch.com': 'MarketWatch',
                'yahoo.com': 'Yahoo Finance',
                'finance.yahoo.com': 'Yahoo Finance',
                'fool.com': 'Motley Fool',
                'seekingalpha.com': 'Seeking Alpha',
                'benzinga.com': 'Benzinga',
                'zacks.com': 'Zacks',
                'morningstar.com': 'Morningstar',
                'barrons.com': 'Barrons',
                'investopedia.com': 'Investopedia',
                'thestreet.com': 'TheStreet',
                'forbes.com': 'Forbes',
                'fortune.com': 'Fortune',
                'businessinsider.com': 'Business Insider',
                
                # General News
                'cnn.com': 'CNN',
                'bbc.com': 'BBC',
                'npr.org': 'NPR',
                'apnews.com': 'Associated Press',
                'ap.org': 'Associated Press',
                'usatoday.com': 'USA Today',
                'nytimes.com': 'New York Times',
                'washingtonpost.com': 'Washington Post',
                'guardian.co.uk': 'The Guardian',
                'theguardian.com': 'The Guardian',
                'economist.com': 'The Economist',
                'axios.com': 'Axios',
                'politico.com': 'Politico',
                'abc.com': 'ABC News',
                'cbsnews.com': 'CBS News',
                'nbcnews.com': 'NBC News',
                'foxnews.com': 'Fox News',
                'reddit.com': 'Reddit',
                'slashdot.org': 'Slashdot',
                'zdnet.com': 'ZDNet',
                'biztoc.com': 'BizToc',
                'economictimes.indiatimes.com': 'Economic Times',
                'macobserver.com': 'Mac Observer',
                'ibtimes.com': 'International Business Times',
                'digitaljournal.com': 'Digital Journal',
                'japantoday.com': 'Japan Today',
                'globenewswire.com': 'GlobeNewswire',
                'timesofindia.indiatimes.com': 'Times of India',
                'variety.com': 'Variety',
                'finovate.com': 'Finovate',
                
                # Tech News
                'techcrunch.com': 'TechCrunch',
                'venturebeat.com': 'VentureBeat',
                'theverge.com': 'The Verge',
                'arstechnica.com': 'Ars Technica',
                'engadget.com': 'Engadget',
                'wired.com': 'Wired',
                'mashable.com': 'Mashable',
                'gizmodo.com': 'Gizmodo',
            }
            
            clean_name = domain_mapping.get(domain)
            if clean_name:
                return clean_name
            
            # Create clean name for unmapped domains
            name_parts = domain.split('.')
            if len(name_parts) >= 2:
                name_part = name_parts[0]
                
                if name_part in ['finance', 'money', 'news', 'business'] and len(name_parts) > 2:
                    name_part = name_parts[1]
                
                name_part = name_part.replace('-', ' ').replace('_', ' ')
                return name_part.title()
            
            return domain.replace('.com', '').replace('.org', '').replace('.net', '').title()
            
        except Exception as e:
            logger.warning(f"Failed to extract source from URL '{url}': {str(e)}")
            return "Unknown Source"

    def clean_source_name(self, source: str) -> str:
        """Clean and extract source name from URL or full name"""
        if pd.isna(source) or not source:
            return ""
        
        source = str(source).strip()
        
        if any(indicator in source.lower() for indicator in ['http://', 'https://', 'www.', '.com', '.org', '.net']):
            return self.extract_source_from_url(source)
        
        source_mapping = {
            'wsj': 'Wall Street Journal',
            'ft': 'Financial Times',
            'nyt': 'New York Times',
            'wapo': 'Washington Post',
            'wp': 'Washington Post',
            'lat': 'Los Angeles Times',
            'usat': 'USA Today',
            'ap': 'Associated Press',
            'reuters': 'Reuters',
            'bloomberg': 'Bloomberg',
            'cnbc': 'CNBC',
            'cnn': 'CNN',
            'bbc': 'BBC',
            'npr': 'NPR',
            'abc': 'ABC News',
            'cbs': 'CBS News',
            'nbc': 'NBC News',
            'fox': 'Fox News',
            'mw': 'MarketWatch',
            'yf': 'Yahoo Finance',
            'sa': 'Seeking Alpha',
            'tmf': 'Motley Fool',
            'bi': 'Business Insider',
            'tc': 'TechCrunch',
            'vb': 'VentureBeat'
        }
        
        source_lower = source.lower().strip()
        clean_name = source_mapping.get(source_lower)
        if clean_name:
            return clean_name
        
        return source.title()

    def extract_company_info(self, df: pd.DataFrame, text_column: str) -> Dict[str, str]:
        """Extract company name and ticker from the dataset"""
        company_info = {'company_name': '', 'ticker': ''}
        
        # Check column names for company/ticker info
        for col in df.columns:
            col_lower = col.lower()
            if 'company' in col_lower or 'firm' in col_lower or 'corp' in col_lower:
                if not df[col].isna().all():
                    company_info['company_name'] = str(df[col].iloc[0])
            elif 'ticker' in col_lower or 'symbol' in col_lower:
                if not df[col].isna().all():
                    company_info['ticker'] = str(df[col].iloc[0]).upper()
        
        # Check for 'Ticker' column specifically
        if 'Ticker' in df.columns and not df['Ticker'].isna().all():
            company_info['ticker'] = str(df['Ticker'].iloc[0]).upper()
        
        # If ticker found, get proper company name
        if company_info['ticker'] and not company_info['company_name']:
            mapped_name = self.get_company_name_from_ticker(company_info['ticker'])
            if mapped_name:
                company_info['company_name'] = mapped_name
        
        # If not found in columns, try to extract from text content
        if not company_info['company_name'] or not company_info['ticker']:
            sample_texts = df[text_column].dropna().head(3).tolist()
            if sample_texts:
                combined_sample = ' '.join([str(text)[:500] for text in sample_texts])
                extracted_info = self.extract_company_from_text(combined_sample)
                
                if not company_info['ticker'] and extracted_info['ticker']:
                    company_info['ticker'] = extracted_info['ticker'].upper()
                    mapped_name = self.get_company_name_from_ticker(company_info['ticker'])
                    if mapped_name:
                        company_info['company_name'] = mapped_name
                
                if not company_info['company_name'] and extracted_info['company_name']:
                    company_info['company_name'] = extracted_info['company_name']
        
        return company_info
    
    def extract_company_from_text(self, text: str) -> Dict[str, str]:
        """Use AI to extract company name and ticker from text"""
        prompt = f"""
        From the following text, extract the main company name and stock ticker symbol (if mentioned).
        Return only the company name and ticker, or "Not found" if not clearly identifiable.
        
        Text: {text}
        
        Please respond in this exact format:
        Company: [company name or "Not found"]
        Ticker: [ticker symbol or "Not found"]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting company information from financial text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            company_name = "Not found"
            ticker = "Not found"
            
            for line in content.split('\n'):
                if line.strip().startswith('Company:'):
                    company_name = line.split(':', 1)[1].strip()
                elif line.strip().startswith('Ticker:'):
                    ticker = line.split(':', 1)[1].strip()
            
            return {
                'company_name': company_name if company_name != "Not found" else "",
                'ticker': ticker if ticker != "Not found" else ""
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract company info from text: {str(e)}")
            return {'company_name': '', 'ticker': ''}
    
    def get_top_sources(self, df: pd.DataFrame, source_column: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get top N news sources from the dataset with cleaned names"""
        if source_column not in df.columns:
            return []
        
        cleaned_sources = df[source_column].dropna().apply(self.clean_source_name)
        cleaned_sources = cleaned_sources[cleaned_sources != ""]
        source_counts = cleaned_sources.value_counts().head(top_n)
        return [(source, count) for source, count in source_counts.items()]
    
    def get_date_range(self, df: pd.DataFrame, date_column: str) -> Dict[str, str]:
        """Get date range from the dataset"""
        if date_column not in df.columns:
            return {'start_date': '', 'end_date': ''}
        
        try:
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            
            if dates.empty:
                return {'start_date': '', 'end_date': ''}
            
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            
            return {'start_date': start_date, 'end_date': end_date}
            
        except Exception as e:
            logger.warning(f"Failed to extract date range: {str(e)}")
            return {'start_date': '', 'end_date': ''}
    
    def get_real_stock_data(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get real stock price data using yfinance if available"""
        if not YFINANCE_AVAILABLE:
            return self.generate_mock_price_data()
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.download(start=start_date, end=end_date, progress=False)
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return self.generate_mock_price_data()
            
            # Get current price (latest available)
            current_price = hist['Close'].iloc[-1]
            
            # Calculate average price for the period
            avg_price = hist['Close'].mean()
            
            # Calculate volatility (std deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            return {
                'current_price': round(current_price, 2),
                'avg_price_period': round(avg_price, 2),
                'volatility': volatility,
                'price_data_available': True
            }
            
        except Exception as e:
            logger.warning(f"Failed to get real price data for {ticker}: {str(e)}")
            return self.generate_mock_price_data()
    
    def generate_mock_price_data(self) -> Dict[str, Any]:
        """Generate mock price data when real data is not available"""
        mock_current_price = np.random.uniform(150, 200)
        return {
            'current_price': round(mock_current_price, 2),
            'avg_price_period': round(mock_current_price * np.random.uniform(0.98, 1.02), 2),
            'volatility': 0.25,  # 25% volatility assumption
            'price_data_available': False
        }
    
    def calculate_sentiment_metrics(self, df: pd.DataFrame, sentiment_column: str, source_column: str = None) -> Dict[str, Any]:
        """Calculate sentiment metrics for analysis"""
        if sentiment_column not in df.columns:
            return {
                'average_sentiment': 0,
                'total_mentions': len(df),
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'news_vs_social': {'news': {'count': 0, 'avg_sentiment': 0}, 'social': {'count': 0, 'avg_sentiment': 0}}
            }
        
        try:
            sentiment_scores = pd.to_numeric(df[sentiment_column], errors='coerce').dropna()
            
            if sentiment_scores.empty:
                return {
                    'average_sentiment': 0,
                    'total_mentions': len(df),
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                    'news_vs_social': {'news': {'count': 0, 'avg_sentiment': 0}, 'social': {'count': 0, 'avg_sentiment': 0}}
                }
            
            avg_sentiment = sentiment_scores.mean()
            total_mentions = len(sentiment_scores)
            
            # Sentiment distribution
            positive_count = len(sentiment_scores[sentiment_scores > 0.1])
            negative_count = len(sentiment_scores[sentiment_scores < -0.1])
            neutral_count = total_mentions - positive_count - negative_count
            
            sentiment_distribution = {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            }
            
            # News vs Social analysis
            news_vs_social = {'news': {'count': 0, 'avg_sentiment': 0}, 'social': {'count': 0, 'avg_sentiment': 0}}
            
            if source_column and source_column in df.columns:
                social_sources = ['reddit', 'twitter', 'facebook', 'social']
                
                df_with_sentiment = df[df[sentiment_column].notna()].copy()
                df_with_sentiment['clean_source'] = df_with_sentiment[source_column].apply(self.clean_source_name).str.lower()
                
                social_mask = df_with_sentiment['clean_source'].str.contains('|'.join(social_sources), case=False, na=False)
                
                social_data = df_with_sentiment[social_mask]
                news_data = df_with_sentiment[~social_mask]
                
                if len(social_data) > 0:
                    news_vs_social['social']['count'] = len(social_data)
                    news_vs_social['social']['avg_sentiment'] = pd.to_numeric(social_data[sentiment_column], errors='coerce').mean()
                
                if len(news_data) > 0:
                    news_vs_social['news']['count'] = len(news_data)
                    news_vs_social['news']['avg_sentiment'] = pd.to_numeric(news_data[sentiment_column], errors='coerce').mean()
            
            return {
                'average_sentiment': avg_sentiment,
                'total_mentions': total_mentions,
                'sentiment_distribution': sentiment_distribution,
                'news_vs_social': news_vs_social
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate sentiment metrics: {str(e)}")
            return {
                'average_sentiment': 0,
                'total_mentions': len(df),
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'news_vs_social': {'news': {'count': 0, 'avg_sentiment': 0}, 'social': {'count': 0, 'avg_sentiment': 0}}
            }
    
    def extract_keywords_and_themes(self, df: pd.DataFrame, text_column: str, company_info: Dict[str, str] = None, top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract top keywords and themes from the text data, excluding company names and tickers"""
        try:
            all_text = ' '.join(df[text_column].dropna().astype(str))
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            
            # Enhanced stop words list including company-related terms
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'had', 
                'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 
                'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 
                'let', 'put', 'say', 'she', 'too', 'use', 'will', 'said', 'each', 'make', 
                'most', 'over', 'such', 'time', 'very', 'when', 'have', 'from', 'they', 'know', 
                'want', 'been', 'good', 'much', 'some', 'than', 'call', 'come', 'could', 'find', 
                'first', 'look', 'made', 'many', 'may', 'people', 'these', 'think', 'this', 
                'well', 'work', 'would', 'year', 'years', 'about', 'after', 'before', 'down', 
                'here', 'just', 'like', 'long', 'more', 'only', 'other', 'right', 'since', 
                'still', 'such', 'take', 'through', 'under', 'where', 'while', 'with', 'without',
                # Common business terms
                'company', 'business', 'market', 'stock', 'price', 'share', 'investor', 'investment',
                'report', 'news', 'article', 'today', 'yesterday', 'week', 'month', 'quarter'
            }
            
            # Add company-specific terms to stop words if available
            if company_info:
                if company_info.get('company_name'):
                    # Split company name and add parts
                    company_parts = re.findall(r'\b[a-zA-Z]{3,}\b', company_info['company_name'].lower())
                    stop_words.update(company_parts)
                    # Also add common company suffixes
                    stop_words.update(['inc', 'corp', 'corporation', 'company', 'ltd', 'limited'])
                
                if company_info.get('ticker'):
                    stop_words.add(company_info['ticker'].lower())
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_counts = Counter(filtered_words)
            
            return word_counts.most_common(top_n)
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {str(e)}")
            return []

    def create_charts_for_pdf(self, sentiment_metrics: Dict[str, Any], top_sources: List[Tuple[str, int]], 
                             keywords: List[Tuple[str, int]], output_dir: str, is_news_only: bool = False) -> Dict[str, str]:
        """Create charts for PDF report"""
        chart_files = {}
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Set style for professional charts
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Sentiment Distribution Chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            distribution = sentiment_metrics.get('sentiment_distribution', {})
            labels = ['Positive', 'Neutral', 'Negative']
            values = [distribution.get('positive', 0), distribution.get('neutral', 0), distribution.get('negative', 0)]
            colors = ['#2E8B57', '#FFA500', '#DC143C']
            
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
            
            # Make percentage text bold and white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            plt.tight_layout()
            sentiment_chart_path = os.path.join(output_dir, 'sentiment_distribution.png')
            plt.savefig(sentiment_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files['sentiment_distribution'] = sentiment_chart_path
            
            # 2. News vs Social Chart (only if social data exists)
            news_social = sentiment_metrics.get('news_vs_social', {})
            social_count = news_social.get('social', {}).get('count', 0)
            
            if not is_news_only and social_count > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                categories = ['News', 'Social Media']
                news_sentiment = news_social.get('news', {}).get('avg_sentiment', 0)
                social_sentiment = news_social.get('social', {}).get('avg_sentiment', 0)
                sentiment_scores = [news_sentiment, social_sentiment]
                
                colors = ['#2E8B57' if score > 0 else '#DC143C' for score in sentiment_scores]
                bars = ax.bar(categories, sentiment_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, score in zip(bars, sentiment_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                            f'{score:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                            fontweight='bold', fontsize=11)
                
                ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
                ax.set_title('News vs Social Media Sentiment', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                news_social_chart_path = os.path.join(output_dir, 'news_vs_social.png')
                plt.savefig(news_social_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['news_vs_social'] = news_social_chart_path
            
            # 3. Top Sources Chart
            if top_sources:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sources = [source[0] for source in top_sources[:8]]  # Limit to top 8
                counts = [source[1] for source in top_sources[:8]]
                
                bars = ax.bar(sources, counts, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
                ax.set_title('Top News Sources', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                sources_chart_path = os.path.join(output_dir, 'top_sources.png')
                plt.savefig(sources_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['top_sources'] = sources_chart_path
            
            # 4. Keywords Chart
            if keywords:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                words = [kw[0].title() for kw in keywords[:10]]
                freqs = [kw[1] for kw in keywords[:10]]
                
                bars = ax.barh(words, freqs, color='#2e8b57', alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, freq in zip(bars, freqs):
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                            str(freq), ha='left', va='center', fontweight='bold', fontsize=10)
                
                ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title('Top Keywords & Themes', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                keywords_chart_path = os.path.join(output_dir, 'keywords.png')
                plt.savefig(keywords_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['keywords'] = keywords_chart_path
            
            return chart_files
            
        except Exception as e:
            logger.warning(f"Failed to create charts: {str(e)}")
            return {}

    def generate_individual_summary(self, text: str, company_info: Dict[str, str] = None) -> str:
        """Generate individual summary for a news item"""
        try:
            # Create context-aware prompt
            if company_info and company_info.get('company_name'):
                company_context = f"This content is related to {company_info['company_name']}"
                if company_info.get('ticker'):
                    company_context += f" ({company_info['ticker']})"
                company_context += ". "
            else:
                company_context = ""
            
            prompt = f"""
            {company_context}Please provide a concise summary of this news content in 2-3 sentences. 
            Focus on the key facts, main points, and any significant implications. 
            Make the summary informative and professional.
            
            Content: {text[:1000]}  # Limit to prevent token overflow
            
            Summary:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional news summarizer. Provide clear, concise, and factual summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate individual summary: {str(e)}")
            return f"Summary unavailable: {str(text)[:100]}..."

    def generate_overall_summary(self, df: pd.DataFrame, text_column: str, company_info: Dict[str, str], 
                                top_sources: List[Tuple[str, int]], keywords: List[Tuple[str, int]],
                                date_range: Dict[str, str]) -> str:
        """Generate comprehensive overall summary"""
        try:
            # Sample representative texts
            sample_texts = df[text_column].dropna().head(5).tolist()
            combined_sample = '\n\n'.join([f"Article {i+1}: {text[:300]}..." for i, text in enumerate(sample_texts)])
            
            # Prepare context information
            context_info = []
            
            if company_info.get('company_name'):
                context_info.append(f"Company: {company_info['company_name']}")
            if company_info.get('ticker'):
                context_info.append(f"Ticker: {company_info['ticker']}")
            if date_range.get('start_date'):
                if date_range['start_date'] == date_range.get('end_date'):
                    context_info.append(f"Date: {date_range['start_date']}")
                else:
                    context_info.append(f"Date Range: {date_range['start_date']} to {date_range.get('end_date', '')}")
            if top_sources:
                top_3_sources = [source[0] for source in top_sources[:3]]
                context_info.append(f"Top Sources: {', '.join(top_3_sources)}")
            if keywords:
                top_keywords = [kw[0] for kw in keywords[:5]]
                context_info.append(f"Key Themes: {', '.join(top_keywords)}")
            
            context_str = '\n'.join(context_info) if context_info else "General news analysis"
            
            prompt = f"""
            Based on the following news data analysis, provide a comprehensive summary that covers:

            1. Main themes and topics discussed
            2. Key developments and trends
            3. Overall sentiment or tone
            4. Significant events or announcements
            5. Potential implications or impact

            Analysis Context:
            {context_str}
            Total articles processed: {len(df)}

            Sample Articles:
            {combined_sample}

            Please provide a professional, informative summary that would be valuable for business intelligence purposes.
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior business analyst providing executive summaries of news analysis. Be thorough, professional, and insightful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate overall summary: {str(e)}")
            return f"Unable to generate comprehensive summary. Processed {len(df)} articles covering various topics and themes."

    def generate_equity_research_report(self, df: pd.DataFrame, company_info: Dict[str, str], 
                                       sentiment_metrics: Dict[str, Any], price_data: Dict[str, Any],
                                       keywords: List[Tuple[str, int]], top_sources: List[Tuple[str, int]],
                                       date_range: Dict[str, str]) -> str:
        """Generate comprehensive equity research report"""
        try:
            # Calculate key metrics
            avg_sentiment = sentiment_metrics.get('average_sentiment', 0)
            total_mentions = sentiment_metrics.get('total_mentions', 0)
            current_price = price_data.get('current_price', 0)
            volatility = price_data.get('volatility', 0.25)
            
            # Generate recommendation
            if avg_sentiment > 0.3:
                recommendation = "BUY"
                outlook = "Positive"
            elif avg_sentiment < -0.3:
                recommendation = "SELL"
                outlook = "Negative"
            else:
                recommendation = "HOLD"
                outlook = "Neutral"
            
            # Calculate target price (simplified model)
            sentiment_multiplier = 1 + (avg_sentiment * 0.05)  # 5% impact per sentiment unit
            target_price = current_price * sentiment_multiplier
            
            # Generate report
            report = f"""
EQUITY RESEARCH REPORT
{'='*50}

COMPANY OVERVIEW
Company: {company_info.get('company_name', 'N/A')}
Ticker: {company_info.get('ticker', 'N/A')}
Report Date: {datetime.now().strftime('%Y-%m-%d')}
Analysis Period: {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}

PRICE ANALYSIS
Current Price: ${current_price:.2f}
Target Price (1-Month): ${target_price:.2f}
Potential Return: {((target_price - current_price) / current_price * 100):+.1f}%
Volatility: {volatility:.1%}

RECOMMENDATION: {recommendation}
Investment Outlook: {outlook}

SENTIMENT ANALYSIS
Overall Sentiment Score: {avg_sentiment:+.3f}
Total News Mentions: {total_mentions}
Positive Coverage: {sentiment_metrics.get('sentiment_distribution', {}).get('positive', 0)} articles
Negative Coverage: {sentiment_metrics.get('sentiment_distribution', {}).get('negative', 0)} articles
Neutral Coverage: {sentiment_metrics.get('sentiment_distribution', {}).get('neutral', 0)} articles

NEWS vs SOCIAL MEDIA
News Sentiment: {sentiment_metrics.get('news_vs_social', {}).get('news', {}).get('avg_sentiment', 0):+.3f}
Social Media Sentiment: {sentiment_metrics.get('news_vs_social', {}).get('social', {}).get('avg_sentiment', 0):+.3f}

TOP NEWS SOURCES
{chr(10).join([f'{i+1}. {source[0]}: {source[1]} articles' for i, source in enumerate(top_sources[:5])])}

KEY THEMES & CATALYSTS
{chr(10).join([f'• {keyword[0].title()} (mentioned {keyword[1]} times)' for keyword in keywords[:8]])}

RISK FACTORS
• Market volatility may impact price targets
• Sentiment analysis based on limited time period
• External market factors not considered in price model
• News sentiment may not reflect fundamental performance

ANALYST NOTES
This analysis is based on sentiment data from {total_mentions} news articles and social media mentions
over the period from {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}.
The price target is calculated using a sentiment-momentum model and should be considered
a tactical 1-month target rather than a long-term valuation.

{'Real market data used' if price_data.get('price_data_available') else 'Estimated price data used'} for price analysis.

DISCLAIMER
This report is for informational purposes only and should not be considered as investment advice.
Past performance does not guarantee future results. Please consult with a qualified financial
advisor before making investment decisions.

Report generated by News And Social Media Analysis Bot
Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate equity research report: {str(e)}")
            return f"Error generating report: {str(e)}"

    def process_file(self, file_path: str, analysis_type: str = "auto", output_dir: str = "output") -> Dict[str, Any]:
        """Main processing function that handles both summarization and equity research"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load file
            df = self.load_file(file_path)
            if df is None:
                return {"success": False, "error": "Failed to load file"}
            
            # Identify columns
            text_column = self.identify_text_column(df)
            if not text_column:
                return {"success": False, "error": "Could not identify text column in the dataset"}
            
            source_column = self.identify_source_column(df)
            date_column = self.identify_date_column(df)
            sentiment_column = self.identify_sentiment_column(df)
            
            # Extract company information
            company_info = self.extract_company_info(df, text_column)
            
            # Determine analysis type automatically if needed
            if analysis_type == "auto":
                # Use equity research if sentiment column exists, otherwise use summarization
                analysis_type = "equity_research" if sentiment_column else "summarization"
            
            # Get metadata
            top_sources = self.get_top_sources(df, source_column) if source_column else []
            date_range = self.get_date_range(df, date_column) if date_column else {}
            keywords = self.extract_keywords_and_themes(df, text_column, company_info)
            
            # Initialize results
            results = {
                "success": True,
                "analysis_type": analysis_type,
                "processed_items": len(df),
                "company_info": company_info,
                "top_sources": top_sources,
                "date_range": date_range,
                "keywords": keywords,
                "text_column_used": text_column,
                "source_column_used": source_column,
                "has_sentiment_data": sentiment_column is not None
            }
            
            if analysis_type == "equity_research":
                return self._process_equity_research(df, text_column, sentiment_column, source_column, 
                                                   company_info, top_sources, date_range, keywords, 
                                                   output_dir, results)
            else:
                return self._process_summarization(df, text_column, source_column, company_info, 
                                                 top_sources, date_range, keywords, output_dir, results)
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _process_equity_research(self, df: pd.DataFrame, text_column: str, sentiment_column: str,
                               source_column: str, company_info: Dict[str, str], 
                               top_sources: List[Tuple[str, int]], date_range: Dict[str, str],
                               keywords: List[Tuple[str, int]], output_dir: str, 
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """Process equity research analysis"""
        
        # Calculate sentiment metrics
        sentiment_metrics = self.calculate_sentiment_metrics(df, sentiment_column, source_column)
        
        # Get price data
        ticker = company_info.get('ticker', '')
        if ticker and date_range.get('start_date'):
            price_data = self.get_real_stock_data(ticker, date_range['start_date'], 
                                                date_range.get('end_date', date_range['start_date']))
        else:
            price_data = self.generate_mock_price_data()
        
        price_data_source = 'Real Market Data' if price_data.get('price_data_available') else 'Estimated Data'
        
        # Generate equity research report
        equity_report = self.generate_equity_research_report(
            df, company_info, sentiment_metrics, price_data, keywords, top_sources, date_range
        )
        
        # Get top positive and negative news
        top_positive_news = []
        top_negative_news = []
        
        if sentiment_column:
            df_with_sentiment = df[df[sentiment_column].notna()].copy()
            df_with_sentiment[sentiment_column] = pd.to_numeric(df_with_sentiment[sentiment_column], errors='coerce')
            df_with_sentiment = df_with_sentiment.dropna(subset=[sentiment_column])
            
            if len(df_with_sentiment) > 0:
                # Top positive news
                positive_df = df_with_sentiment[df_with_sentiment[sentiment_column] > 0].nlargest(5, sentiment_column)
                for _, row in positive_df.iterrows():
                    top_positive_news.append({
                        'text': str(row[text_column])[:100],
                        'sentiment_score': row[sentiment_column],
                        'date': row.get(self.identify_date_column(df), 'N/A') if self.identify_date_column(df) else 'N/A',
                        'url': row.get(source_column, 'N/A') if source_column else 'N/A'
                    })
                
                # Top negative news
                negative_df = df_with_sentiment[df_with_sentiment[sentiment_column] < 0].nsmallest(5, sentiment_column)
                for _, row in negative_df.iterrows():
                    top_negative_news.append({
                        'text': str(row[text_column])[:100],
                        'sentiment_score': row[sentiment_column],
                        'date': row.get(self.identify_date_column(df), 'N/A') if self.identify_date_column(df) else 'N/A',
                        'url': row.get(source_column, 'N/A') if source_column else 'N/A'
                    })
        
        # Create PDF report
        pdf_path = self.create_pdf_report(
            {
                'company_name': company_info.get('company_name', ''),
                'ticker': company_info.get('ticker', ''),
                'date_range': date_range,
                'top_sources': top_sources,
                'processed_items': len(df)
            },
            sentiment_metrics, keywords, top_positive_news, top_negative_news, 
            price_data, output_dir, 'equity_research'
        )
        
        # Save equity report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_name = company_info.get('company_name', 'Company').replace(' ', '_')
        ticker = company_info.get('ticker', '')
        
        equity_filename = f"equity_research_{company_name}_{ticker}_{timestamp}.txt" if ticker else f"equity_research_{company_name}_{timestamp}.txt"
        equity_report_path = os.path.join(output_dir, equity_filename)
        
        with open(equity_report_path, 'w', encoding='utf-8') as f:
            f.write(equity_report)
        
        # Update results
        results.update({
            "sentiment_metrics": sentiment_metrics,
            "price_data": price_data,
            "price_data_source": price_data_source,
            "equity_report": equity_report,
            "equity_report_file": equity_report_path,
            "pdf_report_file": pdf_path,
            "top_positive_news": top_positive_news,
            "top_negative_news": top_negative_news
        })
        
        return results

    def _process_summarization(self, df: pd.DataFrame, text_column: str, source_column: str,
                              company_info: Dict[str, str], top_sources: List[Tuple[str, int]],
                              date_range: Dict[str, str], keywords: List[Tuple[str, int]],
                              output_dir: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process news summarization analysis"""
        
        # Generate individual summaries
        individual_summaries = []
        summary_texts = []
        
        logger.info("Generating individual summaries...")
        for idx, row in df.iterrows():
            text = str(row[text_column])
            if len(text.strip()) > 50:  # Only summarize substantial content
                summary = self.generate_individual_summary(text, company_info)
                individual_summaries.append(summary)
                summary_texts.append(summary)
            else:
                summary_texts.append("Content too short for meaningful summary")
                individual_summaries.append("Content too short for meaningful summary")
        
        # Generate overall summary
        overall_summary = self.generate_overall_summary(
            df, text_column, company_info, top_sources, keywords, date_range
        )
        
        # Create document title
        if company_info.get('company_name'):
            document_title = f"News Summary Report - {company_info['company_name']}"
        else:
            document_title = "News Summary Report"
        
        # Save enhanced CSV with summaries
        df_output = df.copy()
        df_output['AI_Summary'] = summary_texts
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"news_summary_detailed_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        
        # Save overall summary to file
        summary_filename = f"news_summary_overall_{timestamp}.txt"
        summary_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{document_title}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Items Processed: {len(df)}\n")
            
            if company_info.get('company_name'):
                f.write(f"Company: {company_info['company_name']}\n")
            if company_info.get('ticker'):
                f.write(f"Ticker: {company_info['ticker']}\n")
            if date_range.get('start_date'):
                f.write(f"Date Range: {date_range['start_date']} to {date_range.get('end_date', '')}\n")
            
            f.write("\nTOP NEWS SOURCES:\n")
            for i, (source, count) in enumerate(top_sources[:5], 1):
                f.write(f"{i}. {source}: {count} articles\n")
            
            f.write("\nKEY THEMES:\n")
            for keyword, freq in keywords[:10]:
                f.write(f"• {keyword.title()} ({freq} mentions)\n")
            
            f.write(f"\n\nOVERALL SUMMARY:\n{overall_summary}\n")
        
        # Create PDF report
        pdf_path = self.create_pdf_report
        (
            {
                'company_name': company_info.get('company_name', ''),
                'ticker': company_info.get('ticker', ''),
                'date_range': date_range,
                'top_sources': top_sources,
                'processed_items': len(df)
            },
            # This is the completion of the _process_summarization method and any other missing parts
        )
        # Create PDF report (continuing from where it was cut off)
        pdf_path = self.create_pdf_report(
            {
                'company_name': company_info.get('company_name', ''),
                'ticker': company_info.get('ticker', ''),
                'date_range': date_range,
                'top_sources': top_sources,
                'processed_items': len(df)
            },
            {}, keywords, [], [], {}, output_dir, 'summarization', overall_summary
        )
        
        # Update results
        results.update({
            "individual_summaries": individual_summaries,
            "overall_summary": overall_summary,
            "detailed_output_file": output_path,
            "summary_file": summary_path,
            "pdf_report_file": pdf_path,
            "document_title": document_title
        })
        
        return results

    def create_charts_for_pdf(self, sentiment_metrics: Dict[str, Any], top_sources: List[Tuple[str, int]], 
                             keywords: List[Tuple[str, int]], output_dir: str, is_news_only: bool = False) -> Dict[str, str]:
        """Create charts for PDF report"""
        chart_files = {}
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Set style for professional charts
            plt.style.use('default')  # Changed from seaborn-v0_8 for better compatibility
            
            # 1. Sentiment Distribution Chart (only if sentiment data exists)
            if sentiment_metrics.get('sentiment_distribution'):
                fig, ax = plt.subplots(figsize=(8, 6))
                
                distribution = sentiment_metrics.get('sentiment_distribution', {})
                labels = ['Positive', 'Neutral', 'Negative']
                values = [distribution.get('positive', 0), distribution.get('neutral', 0), distribution.get('negative', 0)]
                colors = ['#2E8B57', '#FFA500', '#DC143C']
                
                # Only create pie chart if there's data
                if sum(values) > 0:
                    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
                    
                    # Make percentage text bold and white
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(10)
                    
                    plt.tight_layout()
                    sentiment_chart_path = os.path.join(output_dir, 'sentiment_distribution.png')
                    plt.savefig(sentiment_chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['sentiment_distribution'] = sentiment_chart_path
            
            # 2. News vs Social Chart (only if social data exists and not news-only)
            news_social = sentiment_metrics.get('news_vs_social', {})
            social_count = news_social.get('social', {}).get('count', 0)
            
            if not is_news_only and social_count > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                categories = ['News', 'Social Media']
                news_sentiment = news_social.get('news', {}).get('avg_sentiment', 0)
                social_sentiment = news_social.get('social', {}).get('avg_sentiment', 0)
                sentiment_scores = [news_sentiment, social_sentiment]
                
                colors = ['#2E8B57' if score > 0 else '#DC143C' for score in sentiment_scores]
                bars = ax.bar(categories, sentiment_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, score in zip(bars, sentiment_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                            f'{score:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                            fontweight='bold', fontsize=11)
                
                ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
                ax.set_title('News vs Social Media Sentiment', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                news_social_chart_path = os.path.join(output_dir, 'news_vs_social.png')
                plt.savefig(news_social_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['news_vs_social'] = news_social_chart_path
            
            # 3. Top Sources Chart
            if top_sources:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sources = [source[0] for source in top_sources[:8]]  # Limit to top 8
                counts = [source[1] for source in top_sources[:8]]
                
                bars = ax.bar(sources, counts, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
                ax.set_title('Top News Sources', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                sources_chart_path = os.path.join(output_dir, 'top_sources.png')
                plt.savefig(sources_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['top_sources'] = sources_chart_path
            
            # 4. Keywords Chart
            if keywords:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                words = [kw[0].title() for kw in keywords[:10]]
                freqs = [kw[1] for kw in keywords[:10]]
                
                bars = ax.barh(words, freqs, color='#2e8b57', alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, freq in zip(bars, freqs):
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                            str(freq), ha='left', va='center', fontweight='bold', fontsize=10)
                
                ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title('Top Keywords & Themes', fontsize=16, fontweight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                keywords_chart_path = os.path.join(output_dir, 'keywords.png')
                plt.savefig(keywords_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['keywords'] = keywords_chart_path
            
            return chart_files
            
        except Exception as e:
            logger.warning(f"Failed to create charts: {str(e)}")
            return {}

    def create_pdf_report(self, metadata: Dict[str, Any], sentiment_metrics: Dict[str, Any], 
                          keywords: List[Tuple[str, int]], top_positive_news: List[Dict[str, Any]], 
                          top_negative_news: List[Dict[str, Any]], price_data: Dict[str, Any],
                          output_dir: str, analysis_type: str, overall_summary: str = None) -> str:
        """Create a comprehensive PDF report with charts and analysis"""
        
        # Check if this is news-only data
        news_social = sentiment_metrics.get('news_vs_social', {})
        social_count = news_social.get('social', {}).get('count', 0)
        is_news_only = social_count == 0
        
        # Create charts with news-only flag
        chart_files = self.create_charts_for_pdf(sentiment_metrics, metadata.get('top_sources', []), keywords, output_dir, is_news_only)
        
        # Setup PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_name = metadata.get('company_name', 'Analysis')
        ticker = metadata.get('ticker', '')
        
        # Create filename
        filename_parts = [analysis_type.replace('_', '_')]
        if company_name and company_name != 'Analysis':
            clean_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename_parts.append(clean_name.replace(' ', '_'))
        if ticker:
            filename_parts.append(ticker)
        filename_parts.append(timestamp)
        
        pdf_filename = f"{'_'.join(filename_parts)}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor('#1f77b4')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2e8b57')
        )
        
        # Add title
        if analysis_type == 'equity_research':
            title_text = f"Equity Research Report<br/>{company_name}"
            if ticker:
                title_text += f" ({ticker})"
        else:
            title_text = f"News Analysis Report<br/>{company_name if company_name != 'Analysis' else 'News Summary'}"
        
        elements.append(Paragraph(title_text, title_style))
        elements.append(Spacer(1, 20))
        
        # Add generation info
        date_range = metadata.get('date_range', {})
        gen_info = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
        if date_range.get('start_date'):
            if date_range['start_date'] == date_range.get('end_date', ''):
                gen_info += f"Analysis Date: {date_range['start_date']}<br/>"
            else:
                gen_info += f"Analysis Period: {date_range['start_date']} to {date_range.get('end_date', '')}<br/>"
        gen_info += f"Total Items Processed: {metadata.get('processed_items', 0)}"
        
        elements.append(Paragraph(gen_info, styles['Normal']))
        elements.append(Spacer(1, 30))
        
        # Executive Summary
        elements.append(Paragraph("1. Executive Summary", heading_style))
        
        if analysis_type == 'equity_research':
            exec_summary = f"""
            <b>Company:</b> {company_name}<br/>
            <b>Ticker:</b> {ticker}<br/>
            <b>Overall Sentiment:</b> {sentiment_metrics.get('average_sentiment', 0):+.3f}<br/>
            <b>Total Mentions:</b> {sentiment_metrics.get('total_mentions', 0)}<br/>
            <b>Current Price:</b> ${price_data.get('current_price', 0):.2f}<br/>
            <b>Target Period:</b> 1-Month Tactical Recommendation<br/>
            <b>Price Data Source:</b> {price_data.get('price_data_available', False) and 'Real Market Data' or 'Estimated Data'}<br/>
            """
        else:
            exec_summary = f"""
            <b>Analysis Type:</b> News Summarization<br/>
            <b>Items Processed:</b> {metadata.get('processed_items', 0)}<br/>
            """
            if company_name != 'Analysis':
                exec_summary += f"<b>Company Focus:</b> {company_name}<br/>"
            if ticker:
                exec_summary += f"<b>Ticker:</b> {ticker}<br/>"
        
        elements.append(Paragraph(exec_summary, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Add charts
        elements.append(Paragraph("2. Visual Analysis", heading_style))
        
        # Sentiment Distribution Chart
        if 'sentiment_distribution' in chart_files:
            try:
                img = Image(chart_files['sentiment_distribution'])
                img.drawHeight = 4*inch
                img.drawWidth = 6*inch
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                logger.warning(f"Failed to add sentiment distribution chart: {str(e)}")
        
        # News vs Social Chart (only if social data exists)
        if 'news_vs_social' in chart_files:
            try:
                img = Image(chart_files['news_vs_social'])
                img.drawHeight = 4*inch
                img.drawWidth = 6*inch
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                logger.warning(f"Failed to add news vs social chart: {str(e)}")
        
        # Top Sources Chart
        if 'top_sources' in chart_files:
            try:
                img = Image(chart_files['top_sources'])
                img.drawHeight = 4*inch
                img.drawWidth = 7*inch
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                logger.warning(f"Failed to add top sources chart: {str(e)}")
        
        # Keywords Chart
        if 'keywords' in chart_files:
            try:
                img = Image(chart_files['keywords'])
                img.drawHeight = 4*inch
                img.drawWidth = 7*inch
                elements.append(img)
                elements.append(Spacer(1, 20))
            except Exception as e:
                logger.warning(f"Failed to add keywords chart: {str(e)}")
        
        # Add page break before detailed analysis
        elements.append(PageBreak())
        
        if analysis_type == 'equity_research':
            # Price Analysis Table
            elements.append(Paragraph("3. Price Analysis & Recommendation", heading_style))
            
            # Calculate recommendation
            avg_sentiment = sentiment_metrics.get('average_sentiment', 0)
            current_price = price_data['current_price']
            avg_price = price_data.get('avg_price_period', current_price)
            volatility = price_data.get('volatility', 0.25)
            
            sentiment_multiplier = 1 + (avg_sentiment * 0.05)
            target_price = current_price * sentiment_multiplier
            
            price_momentum = (current_price - avg_price) / avg_price if avg_price != 0 else 0
            
            if avg_sentiment > 0.3 or (avg_sentiment > 0.1 and price_momentum > 0.02):
                recommendation = "BUY"
            elif avg_sentiment < -0.3 or (avg_sentiment < -0.1 and price_momentum < -0.02):
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            upper_range = target_price * (1 + volatility * 0.5)
            lower_range = target_price * (1 - volatility * 0.5)
            
            price_table_data = [
                ['Metric', 'Value'],
                ['Current Price', f'${current_price:.2f}'],
                ['Average Price (Period)', f'${avg_price:.2f}'],
                ['Target Price (1-Month)', f'${target_price:.2f}'],
                ['Price Range (1σ)', f'${lower_range:.2f} - ${upper_range:.2f}'],
                ['Recommendation', recommendation],
                ['Volatility', f'{volatility:.1%}']
            ]
            
            price_table = Table(price_table_data)
            price_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(price_table)
            elements.append(Spacer(1, 20))
        
        # Keywords Table (numbered section)
        section_num = "4" if analysis_type == 'equity_research' else "3"
        if keywords:
            elements.append(Paragraph(f"{section_num}. Top Keywords & Themes", heading_style))
            
            keywords_table_data = [['Rank', 'Keyword', 'Frequency']]
            for i, (keyword, freq) in enumerate(keywords[:10], 1):
                keywords_table_data.append([str(i), keyword.title(), str(freq)])
            
            keywords_table = Table(keywords_table_data)
            keywords_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e8b57')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(keywords_table)
            elements.append(Spacer(1, 20))
        
        # Top Sources Table
        section_num = "5" if analysis_type == 'equity_research' else "4"
        if metadata.get('top_sources'):
            elements.append(Paragraph(f"{section_num}. Top News Sources", heading_style))
            
            sources_table_data = [['Rank', 'Source', 'Article Count']]
            for i, (source, count) in enumerate(metadata['top_sources'], 1):
                sources_table_data.append([str(i), source, str(count)])
            
            sources_table = Table(sources_table_data)
            sources_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            elements.append(sources_table)
            elements.append(Spacer(1, 20))
        
        # Add overall summary if available
        if overall_summary:
            section_num = "6" if analysis_type == 'equity_research' else "5"
            elements.append(Paragraph(f"{section_num}. Overall Analysis Summary", heading_style))
            # Clean the summary text for PDF
            clean_summary = overall_summary.replace('\n', '<br/>')
            elements.append(Paragraph(clean_summary, styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Top News Items with URLs (only for equity research)
        if analysis_type == 'equity_research':
            # Top Positive News
            section_num = "7" if overall_summary else "6"
            if top_positive_news:
                elements.append(Paragraph(f"{section_num}. Top Positive News", heading_style))
                
                pos_news_data = [['Rank', 'Date', 'Headline', 'Sentiment', 'URL']]
                for i, news in enumerate(top_positive_news[:5], 1):
                    headline = news['text'][:50] + "..." if len(news['text']) > 50 else news['text']
                    url = news.get('url', 'N/A')[:30] + "..." if news.get('url') and len(news.get('url', '')) > 30 else news.get('url', 'N/A')
                    pos_news_data.append([
                        str(i),
                        news['date'], 
                        headline, 
                        f"{news['sentiment_score']:+.3f}",
                        url
                    ])
                
                pos_news_table = Table(pos_news_data, colWidths=[0.5*inch, 1.3*inch, 2.7*inch, 0.8*inch, 2.0*inch])
                pos_news_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e8b57')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                
                elements.append(pos_news_table)
                elements.append(Spacer(1, 15))
            
            # Top Negative News
            section_num = "8" if overall_summary else "7"
            if top_negative_news:
                elements.append(Paragraph(f"{section_num}. Top Negative News", heading_style))
                
                neg_news_data = [['Rank', 'Date', 'Headline', 'Sentiment', 'URL']]
                for i, news in enumerate(top_negative_news[:5], 1):
                    headline = news['text'][:50] + "..." if len(news['text']) > 50 else news['text']
                    url = news.get('url', 'N/A')[:30] + "..." if news.get('url') and len(news.get('url', '')) > 30 else news.get('url', 'N/A')
                    neg_news_data.append([
                        str(i),
                        news['date'], 
                        headline, 
                        f"{news['sentiment_score']:+.3f}",
                        url
                    ])
                
                neg_news_table = Table(neg_news_data, colWidths=[0.5*inch, 1.4*inch, 2.7*inch, 0.8*inch, 2.0*inch])
                neg_news_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC143C')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                
                elements.append(neg_news_table)
                elements.append(Spacer(1, 15))
        
        # Add disclaimer
        elements.append(Spacer(1, 30))
        final_section = "9" if analysis_type == 'equity_research' and overall_summary else "8" if analysis_type == 'equity_research' else "6"
        elements.append(Paragraph(f"{final_section}. Disclaimer", heading_style))
        disclaimer_text = """
        This report is for informational purposes only and is not intended as investment advice. 
        The analysis is based on news sentiment and historical data, which may not be indicative of future performance. 
        Price targets are tactical 1-month recommendations based on sentiment momentum and should not be considered as long-term investment guidance.
        Always consult with a qualified financial advisor before making investment decisions.
        """
        elements.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Build PDF
        try:
            doc.build(elements)
            logger.info(f"PDF report created successfully: {pdf_path}")
            
            # Clean up chart files
            for chart_file in chart_files.values():
                try:
                    if os.path.exists(chart_file):
                        os.remove(chart_file)
                except:
                    pass
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to create PDF: {str(e)}")
            return ""


# Main execution helper function
def main():
    """Example usage of the NewsAndSocialMediaAnalysisBot"""
    # Example API key (replace with actual key)
    api_key = "your_openai_api_key_here"
    
    # Initialize the bot
    bot = NewsAndSocialMediaAnalysisBot(api_key)
    
    # Process a file
    file_path = "path_to_your_data_file.csv"  # Replace with actual file path
    
    # Run analysis (auto-detects whether to do equity research or summarization)
    results = bot.process_file(file_path, analysis_type="auto", output_dir="output")
    
    if results["success"]:
        print(f"Analysis completed successfully!")
        print(f"Analysis type: {results['analysis_type']}")
        print(f"Processed {results['processed_items']} items")
        
        if results.get("pdf_report_file"):
            print(f"PDF report created: {results['pdf_report_file']}")
        
        if results.get("equity_report_file"):
            print(f"Equity research report: {results['equity_report_file']}")
        
        if results.get("summary_file"):
            print(f"Summary report: {results['summary_file']}")
            
    else:
        print(f"Analysis failed: {results['error']}")


if __name__ == "__main__":
    main()