import streamlit as st
import pandas as pd
import json
import os
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
warnings.filterwarnings('ignore')

# Import the updated backend
try:
    from backend import NewsAndSocialMediaAnalysisBot, get_available_columns
except ImportError:
    st.error("Please make sure 'backend.py' is in the same directory as this file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="News And Social Media Analysis Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .report-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        margin-top: 1rem;
        max-height: 600px;
        overflow-y: auto;
    }
    .analysis-type-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .analysis-type-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
    }
    .analysis-type-card.selected {
        border-color: #1f77b4;
        background: linear-gradient(135deg, #e3f2fd, #f0f8ff);
    }
    .tab-content {
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    .pdf-download-card {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None
    if 'uploaded_file_processed' not in st.session_state:
        st.session_state.uploaded_file_processed = False
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "auto"
    if 'email_sent' not in st.session_state:
        st.session_state.email_sent = False

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key.startswith('sk-') and len(api_key) > 20

def send_results_via_email(email_address: str, results: dict, sender_email: str, sender_password: str) -> bool:
    """Send analysis results via email with PDF attachment"""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email_address
        
        # Determine subject based on analysis type
        analysis_type = results.get('analysis_type', 'analysis')
        if analysis_type == 'equity_research':
            subject = f"Equity Research Report - {results.get('company_info', {}).get('company_name', 'Analysis')}"
        else:
            subject = f"News Summary Report - {results.get('document_title', 'Analysis')}"
        
        msg['Subject'] = subject
        
        # Create email body based on analysis type
        if analysis_type == 'equity_research':
            html_body = create_equity_email_body(results)
        else:
            html_body = create_summary_email_body(results)
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach PDF file (priority attachment)
        if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
            with open(results["pdf_report_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="pdf")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["pdf_report_file"]))
                msg.attach(attachment)
        
        # Attach other files if they exist
        if "equity_report_file" in results and os.path.exists(results["equity_report_file"]):
            with open(results["equity_report_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="txt")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["equity_report_file"]))
                msg.attach(attachment)
        
        if "output_file" in results and os.path.exists(results["output_file"]):
            with open(results["output_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="csv")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["output_file"]))
                msg.attach(attachment)
        
        if "overall_summary_file" in results and os.path.exists(results["overall_summary_file"]):
            with open(results["overall_summary_file"], "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="txt")
                attachment.add_header('Content-Disposition', 'attachment', 
                                    filename=os.path.basename(results["overall_summary_file"]))
                msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def create_equity_email_body(results: dict) -> str:
    """Create HTML email body for equity research report"""
    company_info = results.get("company_info", {})
    sentiment_metrics = results.get("sentiment_metrics", {})
    
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(90deg, #1f77b4, #2e8b57); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; background-color: #f8f9fa; }}
            .metric {{ background: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .pdf-highlight {{ background: linear-gradient(90deg, #ff6b6b, #ffa500); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Equity Research Report</h1>
            <h2>{company_info.get('company_name', 'Company Analysis')} ({company_info.get('ticker', 'N/A')})</h2>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="pdf-highlight">
            <h3>📊 Professional PDF Report with Charts & Graphs Attached!</h3>
            <p>Your comprehensive analysis includes visual charts, tables, and detailed insights.</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric"><strong>Overall Sentiment:</strong> {sentiment_metrics.get('average_sentiment', 0):+.2f}</div>
            <div class="metric"><strong>Total Mentions:</strong> {sentiment_metrics.get('total_mentions', 0)}</div>
            <div class="metric"><strong>Items Processed:</strong> {results.get('processed_items', 0)}</div>
        </div>
        
        <div class="section">
            <h2>Analysis Details</h2>
            <p><strong>Data Source:</strong> {results.get('price_data_source', 'N/A')} price data</p>
            <p><strong>Sentiment Data:</strong> {'Available' if results.get('has_sentiment_data') else 'Not Available'}</p>
            <p><strong>Report Format:</strong> PDF with Interactive Charts and Visual Analysis</p>
        </div>
        
        <div class="section" style="text-align: center; color: #666;">
            <p>This report was generated automatically by the News And Social Media Analysis Bot</p>
            <p>Powered by AI | Professional PDF Reports | Delivered to your inbox</p>
        </div>
    </body>
    </html>
    """

def create_summary_email_body(results: dict) -> str:
    """Create HTML email body for news summary report"""
    company_info = results.get("company_info", {})
    date_range = results.get("date_range", {})
    
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(90deg, #1f77b4, #2e8b57); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; background-color: #f8f9fa; }}
            .summary-box {{ background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 15px 0; }}
            .pdf-highlight {{ background: linear-gradient(90deg, #ff6b6b, #ffa500); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{results.get('document_title', 'News Summary Report')}</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="pdf-highlight">
            <h3>📋 Professional PDF Report with Visual Analysis Attached!</h3>
            <p>Your news summary includes charts, keyword analysis, and comprehensive insights.</p>
        </div>
        
        <div class="section">
            <h2>Summary Details</h2>
            {'<p><strong>Company:</strong> ' + company_info.get('company_name', '') + '</p>' if company_info.get('company_name') else ''}
            {'<p><strong>Ticker:</strong> ' + company_info.get('ticker', '') + '</p>' if company_info.get('ticker') else ''}
            <p><strong>Items Processed:</strong> {results.get('processed_items', 0)}</p>
            <p><strong>Report Format:</strong> PDF with Charts and Visual Insights</p>
        </div>
        
        <div class="section">
            <h2>Overall Summary</h2>
            <div class="summary-box">
                {results.get('overall_summary', 'No summary available').replace(chr(10), '<br>')}
            </div>
        </div>
        
        <div class="section" style="text-align: center; color: #666;">
            <p>This summary was generated automatically by the News And Social Media Analysis Bot</p>
            <p>Powered by AI | Professional PDF Reports | Delivered to your inbox</p>
        </div>
    </body>
    </html>
    """

def create_sentiment_charts(result_data):
    """Create sentiment visualization charts for equity research"""
    if not result_data or 'success' not in result_data or not result_data['success']:
        return None, None, None
    
    sentiment_metrics = result_data.get('sentiment_metrics', {})
    
    # Chart 1: Overall Sentiment - News vs Social
    fig1 = go.Figure()
    
    news_sentiment = sentiment_metrics.get('news_vs_social', {}).get('news', {}).get('avg_sentiment', 0)
    social_sentiment = sentiment_metrics.get('news_vs_social', {}).get('social', {}).get('avg_sentiment', 0)
    overall_sentiment = sentiment_metrics.get('average_sentiment', 0)
    
    categories = ['News', 'Social Media', 'Overall']
    sentiment_scores = [news_sentiment, social_sentiment, overall_sentiment]
    
    colors = ['#2E8B57' if score > 0 else '#DC143C' for score in sentiment_scores]
    
    fig1.add_trace(go.Bar(
        x=categories,
        y=sentiment_scores,
        marker_color=colors,
        text=[f'{score:+.2f}' for score in sentiment_scores],
        textposition='outside'
    ))
    
    fig1.update_layout(
        title="Overall Sentiment - News vs Social + Equal-weight Overall",
        yaxis_title="Sentiment Score",
        xaxis_title="Source Type",
        showlegend=False,
        height=400
    )
    
    # Chart 2: Sentiment Distribution
    fig2 = go.Figure()
    
    distribution = sentiment_metrics.get('sentiment_distribution', {})
    labels = ['Positive', 'Neutral', 'Negative']
    values = [distribution.get('positive', 0), distribution.get('neutral', 0), distribution.get('negative', 0)]
    colors = ['#2E8B57', '#FFA500', '#DC143C']
    
    fig2.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent'
    ))
    
    fig2.update_layout(
        title="Sentiment Distribution",
        height=400
    )
    
    # Chart 3: Top Sources (if available)
    fig3 = go.Figure()
    
    top_sources = result_data.get('top_sources', [])
    if top_sources:
        sources = [source[0] for source in top_sources]
        counts = [source[1] for source in top_sources]
        
        fig3.add_trace(go.Bar(
            x=sources,
            y=counts,
            marker_color='#1f77b4',
            text=counts,
            textposition='outside'
        ))
        
        fig3.update_layout(
            title="Top News Sources",
            yaxis_title="Mention Count",
            xaxis_title="Source",
            height=400,
            xaxis_tickangle=-45
        )
    else:
        fig3.add_annotation(
            text="No source data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        fig3.update_layout(title="Top News Sources", height=400)
    
    return fig1, fig2, fig3

def display_file_preview(df, max_rows=5):
    """Display a preview of the uploaded file"""
    st.markdown('<div class="sub-header">File Preview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    st.markdown("**Column Information:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes],
        'Non-Null Count': [df[col].notna().sum() for col in df.columns],
        'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 and df[col].notna().any() else 'N/A' for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.markdown("**Data Preview:**")
    st.dataframe(df.head(max_rows), use_container_width=True)
def display_analysis_type_selector():
    """Display analysis type selection interface"""
    st.markdown('<div class="sub-header">Analysis Type</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_selected = st.session_state.analysis_type == "auto"
        if st.button("Auto-Detect", key="auto_btn", 
                    help="Automatically determine the best analysis type based on your data",
                    use_container_width=True):
            st.session_state.analysis_type = "auto"
        
        if auto_selected:
            st.markdown("""
            <div class="analysis-type-card selected" style="color: black;">
                <h4>Auto-Detect</h4>
                <p>Automatically chooses between summarization and equity research based on your data structure.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        summary_selected = st.session_state.analysis_type == "summarization"
        if st.button("News Summarization", key="summary_btn",
                    help="Generate individual and overall summaries of news articles",
                    use_container_width=True):
            st.session_state.analysis_type = "summarization"
        
        if summary_selected:
            st.markdown("""
            <div class="analysis-type-card selected" style="color: black;">
                <h4>News Summarization</h4>
                <p>Creates individual summaries and overall analysis of news articles with company insights.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        equity_selected = st.session_state.analysis_type == "equity_research"
        if st.button("Equity Research", key="equity_btn",
                    help="Generate comprehensive equity research reports with sentiment analysis",
                    use_container_width=True):
            st.session_state.analysis_type = "equity_research"
        
        if equity_selected:
            st.markdown("""
            <div class="analysis-type-card selected" style="color: black;">
                <h4>Equity Research</h4>
                <p>Professional equity research reports with sentiment analysis, price targets, and recommendations.</p>
            </div>
            """, unsafe_allow_html=True)

def display_results(results: dict):
    """Display processing results based on analysis type"""
    if results.get("error"):
        st.error(f"Error: {results['error']}")
        return
    
    analysis_type = results.get('analysis_type', 'analysis')
    
    st.markdown("""
    <div class="success-message">
        <strong>Processing Complete!</strong><br>
        Your analysis has been generated successfully with PDF report and visual charts.
    </div>
    """, unsafe_allow_html=True)
    
    # Highlight PDF availability
    if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
        st.markdown("""
        <div class="pdf-download-card">
            <h3>📊 Professional PDF Report Ready!</h3>
            <p>Your analysis includes interactive charts, tables, and comprehensive insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results based on analysis type
    if analysis_type == 'equity_research':
        display_equity_research_results(results)
    else:
        display_summarization_results(results)
    
    # Email delivery section (common to both)
    display_email_section(results)
    
    # Reset button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Process Another File", type="secondary", use_container_width=True):
            st.session_state.processing_complete = False
            st.session_state.report_data = None
            st.session_state.uploaded_file_processed = False
            st.session_state.email_sent = False
            st.rerun()
    
    with col2:
        if st.button("Share Results", type="secondary", use_container_width=True):
            st.info("Configure email settings below to receive results in your inbox!")

def display_equity_research_results(results: dict):
    """Display equity research specific results"""
    # Display summary metrics
    st.markdown('<div class="sub-header">Analysis Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    company_info = results.get('company_info', {})
    sentiment_metrics = results.get('sentiment_metrics', {})
    
    with col1:
        st.metric("Items Processed", results.get("processed_items", 0))
    
    with col2:
        st.metric("Company Ticker", company_info.get('ticker', 'N/A'))
    
    with col3:
        st.metric("Avg Sentiment", f"{sentiment_metrics.get('average_sentiment', 0):+.3f}")
    
    with col4:
        st.metric("Price Data", results.get('price_data_source', 'N/A'))
    
    # Company Information
    if company_info:
        st.markdown('<div class="sub-header">Company Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company Name:** {company_info.get('company_name', 'Not identified')}")
            st.write(f"**Ticker Symbol:** {company_info.get('ticker', 'Not identified')}")
        
        with col2:
            date_range = results.get('date_range', {})
            st.write(f"**Date Range:** {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}")
            st.write(f"**Sentiment Available:** {'Yes' if results.get('has_sentiment_data') else 'No'}")
    
    # Visualization Charts
    st.markdown('<div class="sub-header">Sentiment Analysis Charts</div>', unsafe_allow_html=True)
    
    fig1, fig2, fig3 = create_sentiment_charts(results)
    
    if fig1 and fig2 and fig3:
        tab1, tab2, tab3 = st.tabs(["Overall Sentiment", "Sentiment Distribution", "Top Sources"])
        
        with tab1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.plotly_chart(fig3, use_container_width=True)
    
    # PDF Download Section
    st.markdown('<div class="sub-header">Download Professional Report</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
            with open(results["pdf_report_file"], "rb") as file:
                filename = os.path.basename(results["pdf_report_file"])
                st.download_button(
                    label="📊 Download PDF Report",
                    data=file.read(),
                    file_name=filename,
                    mime="application/pdf",
                    help="Professional PDF with charts, analysis, and insights",
                    use_container_width=True,
                    type="primary"
                )
    
    with col2:
        if 'equity_report' in results:
            report_content = results['equity_report']
            st.download_button(
                label="📄 Download Text Report",
                data=report_content,
                file_name=f"equity_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col3:
        if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
            file_size = os.path.getsize(results["pdf_report_file"]) / 1024
            st.metric("PDF Size", f"{file_size:.1f} KB")
    
    # Full Report Display
    st.markdown('<div class="sub-header">Complete Equity Research Report</div>', unsafe_allow_html=True)
    
    if 'equity_report' in results:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.text(results['equity_report'])
        st.markdown('</div>', unsafe_allow_html=True)

def display_summarization_results(results: dict):
    """Display news summarization specific results"""
    # Display document title prominently
    document_title = results.get("document_title", "News Summary Report")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1f77b4, #2e8b57);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        ">
            <h2 style="margin: 0; color: white;">{document_title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Processed", results.get("processed_items", 0))
    
    with col2:
        st.metric("Text Column Used", results.get("text_column_used", "N/A"))
    
    with col3:
        source_col = results.get("source_column_used", "Not found")
        st.metric("Source Column", source_col if source_col else "Not found")
    
    with col4:
        if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
            file_size = os.path.getsize(results["pdf_report_file"]) / 1024
            st.metric("PDF Report Size", f"{file_size:.1f} KB")
    
    st.divider()
    
    # Display company information
    company_info = results.get("company_info", {})
    if company_info and (company_info.get('company_name') or company_info.get('ticker')):
        st.markdown('<div class="sub-header">Company Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if company_info.get('company_name'):
                st.info(f"**Company:** {company_info['company_name']}")
            else:
                st.info("**Company:** Not identified")
        
        with col2:
            if company_info.get('ticker'):
                st.info(f"**Ticker:** {company_info['ticker']}")
            else:
                st.info("**Ticker:** Not identified")
        
        with col3:
            date_range = results.get('date_range', {})
            if date_range.get('start_date'):
                if date_range['start_date'] == date_range.get('end_date', ''):
                    st.info(f"**Date:** {date_range['start_date']}")
                else:
                    st.info(f"**Date Range:** {date_range['start_date']} to {date_range.get('end_date', '')}")
            else:
                st.info("**Date:** Not identified")
    
    # Display top sources
    top_sources = results.get("top_sources", [])
    if top_sources:
        st.markdown('<div class="sub-header">Top News Sources</div>', unsafe_allow_html=True)
        
        for i, (source, count) in enumerate(top_sources, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {source}**")
            with col2:
                st.metric("Articles", count)
    
    # Display overall summary
    st.markdown('<div class="sub-header">Overall News Summary</div>', unsafe_allow_html=True)
    overall_summary = results.get("overall_summary", "No summary available")
    
    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="line-height: 1.6; color: #333;">
                {overall_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Download section
    st.markdown('<div class="sub-header">Download Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "pdf_report_file" in results and os.path.exists(results["pdf_report_file"]):
            with open(results["pdf_report_file"], "rb") as file:
                filename = os.path.basename(results["pdf_report_file"])
                st.download_button(
                    label="📊 Download PDF Report",
                    data=file.read(),
                    file_name=filename,
                    mime="application/pdf",
                    help="Professional PDF with charts and visual analysis",
                    use_container_width=True,
                    type="primary"
                )
    
    with col2:
        if "output_file" in results and os.path.exists(results["output_file"]):
            with open(results["output_file"], "rb") as file:
                filename = os.path.basename(results["output_file"])
                st.download_button(
                    label="📄 Download Detailed CSV",
                    data=file.read(),
                    file_name=filename,
                    mime="text/csv",
                    help="Contains original data with individual summaries",
                    use_container_width=True
                )
    
    with col3:
        if "overall_summary_file" in results and os.path.exists(results["overall_summary_file"]):
            with open(results["overall_summary_file"], "rb") as file:
                filename = os.path.basename(results["overall_summary_file"])
                st.download_button(
                    label="📋 Download Text Summary",
                    data=file.read(),
                    file_name=filename,
                    mime="text/plain",
                    help="Complete report with metadata, sources, and summary",
                    use_container_width=True
                )
    
    # Display individual summaries
    with st.expander("View Individual Summaries", expanded=False):
        individual_summaries = results.get("individual_summaries", [])
        
        if individual_summaries:
            search_term = st.text_input("Search summaries:", placeholder="Enter keywords to filter summaries...")
            
            filtered_summaries = individual_summaries
            if search_term:
                filtered_summaries = [s for s in individual_summaries if search_term.lower() in s.lower()]
                st.info(f"Found {len(filtered_summaries)} summaries matching '{search_term}'")
            
            for i, summary in enumerate(filtered_summaries, 1):
                with st.container():
                    st.markdown(f"**Item {i}:**")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #fafafa;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 3px solid #28a745;
                            margin: 10px 0;
                        ">
                            {summary}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.write("No individual summaries available.")

def display_email_section(results: dict):
    """Display email delivery section"""
    st.markdown('<div class="sub-header">Email Delivery</div>', unsafe_allow_html=True)
    
    with st.expander("📧 Send Results via Email", expanded=False):
        st.info("💡 **Tip:** Get your results delivered directly to your inbox with all attachments!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            recipient_email = st.text_input(
                "Recipient Email Address",
                placeholder="your.email@example.com",
                help="Enter the email address where you want to receive the results"
            )
            
            sender_email = st.text_input(
                "Gmail Sender Address",
                placeholder="sender@gmail.com",
                help="Gmail account to send from (must have app password enabled)"
            )
        
        with col2:
            sender_password = st.text_input(
                "Gmail App Password",
                type="password",
                placeholder="App Password (not regular password)",
                help="Generate an App Password in Gmail settings for secure access"
            )
            
            st.markdown("""
            **Email Setup Help:**
            1. Enable 2-factor authentication on Gmail
            2. Generate an App Password (not your regular password)
            3. Use that App Password here
            """)
        
        # Email validation and send button
        email_valid = validate_email(recipient_email) if recipient_email else False
        sender_valid = validate_email(sender_email) if sender_email else False
        password_valid = len(sender_password) > 0 if sender_password else False
        
        if st.button(
            "📨 Send Results via Email",
            disabled=not (email_valid and sender_valid and password_valid),
            use_container_width=True,
            type="primary"
        ):
            if not st.session_state.email_sent:
                with st.spinner("Sending email with attachments..."):
                    success = send_results_via_email(
                        recipient_email, 
                        results, 
                        sender_email, 
                        sender_password
                    )
                    
                    if success:
                        st.session_state.email_sent = True
                        st.success(f"✅ Results sent successfully to {recipient_email}!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to send email. Please check your credentials and try again.")
            else:
                st.info("Email already sent! Check your inbox.")
        
        # Validation messages
        if recipient_email and not email_valid:
            st.error("Please enter a valid recipient email address")
        if sender_email and not sender_valid:
            st.error("Please enter a valid Gmail sender address")
        if not password_valid and sender_password == "":
            st.warning("Gmail App Password is required")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<div class="main-header">📊 News And Social Media Analysis Bot</div>', unsafe_allow_html=True)
    
    # Check if processing is complete
    if st.session_state.processing_complete and st.session_state.report_data:
        display_results(st.session_state.report_data)
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to enable AI-powered analysis",
            key="openai_api_key"
        )
        
        api_key_valid = validate_api_key(api_key) if api_key else False
        
        if api_key and not api_key_valid:
            st.error("Please enter a valid OpenAI API key (starts with 'sk-')")
        elif api_key_valid:
            st.success("✅ Valid API key")
        
        st.divider()
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel, JSON",
            key="data_file"
        )
        
        if uploaded_file and not st.session_state.uploaded_file_processed:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            output_dir = st.text_input("Output Directory", value="output", help="Directory to save generated files")
            
            st.info("💡 **Tip:** All analysis types generate professional PDF reports with charts and visual insights!")
    
    # Main content area
    if not uploaded_file:
        # Landing page
        st.markdown("""
        ## Welcome to AI-Powered News Analysis
        
        This tool provides two types of comprehensive analysis:
        
        ### 📝 **News Summarization**
        - Generate individual summaries for each news item
        - Create comprehensive overall analysis
        - Extract key themes and insights
        - Professional PDF reports with visualizations
        
        ### 📈 **Equity Research Reports**
        - Sentiment analysis and scoring
        - Price targets and recommendations
        - Risk and catalyst analysis
        - Professional PDF reports with charts and graphs
        
        ### 🚀 **Getting Started**
        1. Enter your OpenAI API key in the sidebar
        2. Upload your data file (CSV, Excel, or JSON)
        3. Choose your analysis type (or use Auto-Detect)
        4. Get professional reports delivered to your inbox!
        
        ---
        
        ### 📊 **What You'll Get:**
        - **PDF Report**: Professional document with charts, tables, and insights
        - **Text Reports**: Detailed analysis in readable format
        - **Data Files**: Enhanced datasets with analysis results
        - **Email Delivery**: All results sent directly to your inbox
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card"style="color: black;">
                <h4>🤖 AI-Powered</h4>
                <p>Uses advanced GPT-4 for intelligent summarization and analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card"style="color: black;">
                <h4>📈 Professional Reports</h4>
                <p>Generate equity research reports with price targets and recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card"style="color: black;">
                <h4>📧 Email Integration</h4>
                <p>Get results delivered directly to your inbox with all attachments</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # File uploaded - show processing interface
    if not api_key_valid:
        st.error("⚠️ Please enter a valid OpenAI API key in the sidebar to continue")
        return
    
    # Load and preview file
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Load file for preview
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df_preview = pd.read_csv(temp_file_path)
        elif file_extension in ['xlsx', 'xls']:
            df_preview = pd.read_excel(temp_file_path)
        elif file_extension == 'json':
            df_preview = pd.read_json(temp_file_path)
        
        # Display file preview
        display_file_preview(df_preview)
        
        # Display analysis type selector
        display_analysis_type_selector()
        
        # Processing section
        st.markdown('<div class="sub-header">Start Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"""
            **Ready to analyze:**
            - 📄 File: {uploaded_file.name} ({len(df_preview)} rows, {len(df_preview.columns)} columns)
            - 🤖 Analysis Type: {st.session_state.analysis_type.replace('_', ' ').title()}
            - 🎯 AI Model: GPT-4 with professional analysis
            """)
        
        with col2:
            if st.button(
                "🚀 Start Analysis",
                type="primary",
                use_container_width=True,
                help="Begin AI-powered analysis of your data"
            ):
                # Validate inputs
                if not api_key_valid:
                    st.error("Please provide a valid OpenAI API key")
                    return
                
                # Start processing
                try:
                    with st.spinner("🤖 AI Analysis in Progress... This may take a few minutes"):
                        # Initialize bot
                        bot = NewsAndSocialMediaAnalysisBot(api_key)
                        
                        # Process file
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔍 Analyzing file structure...")
                        progress_bar.progress(10)
                        
                        status_text.text("📊 Processing data with AI...")
                        progress_bar.progress(30)
                        
                        results = bot.process_file(
                            temp_file_path,
                            st.session_state.analysis_type,
                            output_dir
                        )
                        
                        status_text.text("📈 Generating charts and reports...")
                        progress_bar.progress(80)
                        
                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)
                        
                        # Store results and mark as complete
                        st.session_state.report_data = results
                        st.session_state.processing_complete = True
                        st.session_state.uploaded_file_processed = True
                        
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                        
                        # Refresh to show results
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
        
        # Sample data format help
        with st.expander("📋 Expected Data Format", expanded=False):
            st.markdown("""
            ### For News Summarization:
            - **Required**: Text column (content, news, article, description, etc.)
            - **Optional**: Source column, Date column
            
            ### For Equity Research:
            - **Required**: Text column, Sentiment score column
            - **Recommended**: Company/Ticker column, Date column, Source column
            
            ### Column Auto-Detection:
            The system automatically identifies columns based on common naming patterns:
            - **Text**: text, content, news, article, description, summary, body, message, title, headline
            - **Source**: source, publisher, publication, outlet, provider, url, link
            - **Date**: date, time, timestamp, published, created, pub_date, datetime
            - **Sentiment**: sentiment_score, sentiment, score, polarity, compound
            - **Company**: company, firm, corp, ticker, symbol
            """)
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        # Clean up temporary file if it exists
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass

if __name__ == "__main__":
    main()