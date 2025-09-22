"""
FastAPI Application for Financial Analysis
Separated from backend.py for better architecture
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import shutil
import uuid
import os
import logging
from datetime import datetime

# Import the analysis bot from backend
from backend import EnhancedDualFileAnalysisBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analysis API",
    description="AI-powered financial sentiment analysis and reporting system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
BOT = None
FILE_REGISTRY = {}

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    send_email: bool = False
    recipient_email: Optional[str] = None
    email_password: Optional[str] = None
    create_download_package: bool = True

class AnalysisResponse(BaseModel):
    success: bool
    company_info: dict
    pdf_reports: List[str]
    sentiment_data: dict
    email_results: dict
    download_package: str
    downloadables: dict
    error_message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    bot_initialized: bool

def register_file(path: str) -> str:
    """Register a file for download and return a unique ID"""
    file_id = str(uuid.uuid4())
    FILE_REGISTRY[file_id] = path
    logger.info(f"Registered file {path} with ID {file_id}")
    return file_id

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.info(f"Cleaned up temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the analysis bot when API starts"""
    global BOT
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        
        # Initialize email configuration if available
        email_config = {}
        if os.getenv("EMAIL_ADDRESS") and os.getenv("EMAIL_APP_PASSWORD"):
            email_config = {
                'email_address': os.getenv("EMAIL_ADDRESS"),
                'app_password': os.getenv("EMAIL_APP_PASSWORD")
            }
            logger.info("Email configuration loaded")
        
        BOT = EnhancedDualFileAnalysisBot(
            api_key=api_key,
            email_config=email_config if email_config else None
        )
        
        logger.info("Financial analysis bot initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API and cleaning up resources")
    # Clean up registered files
    for file_path in FILE_REGISTRY.values():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup file on shutdown: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if BOT is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        bot_initialized=BOT is not None
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Financial Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "download": "/download/{file_id}",
            "docs": "/docs"
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_files(
    news_file: UploadFile = File(..., description="News articles CSV file"),
    reddit_file: UploadFile = File(..., description="Reddit/social media CSV file"),
    send_email: bool = Form(False, description="Whether to send results via email"),
    recipient_email: Optional[str] = Form(None, description="Email recipient"),
    create_download_package: bool = Form(True, description="Create downloadable package")
):
    """
    Main analysis endpoint
    
    Analyzes uploaded news and social media files to generate:
    - Sentiment analysis
    - PDF reports
    - Charts and visualizations
    - Email distribution (optional)
    """
    temp_files = []
    
    try:
        if BOT is None:
            raise HTTPException(
                status_code=503, 
                detail="Analysis bot not initialized. Check server logs."
            )
        
        # Validate file types
        if not news_file.filename.endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(
                status_code=400,
                detail="News file must be CSV, Excel, or JSON format"
            )
        
        if not reddit_file.filename.endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Reddit file must be CSV, Excel, or JSON format"
            )
        
        # Validate email parameters
        if send_email and not recipient_email:
            raise HTTPException(
                status_code=400,
                detail="Recipient email required when send_email is True"
            )
        
        logger.info(f"Starting analysis for files: {news_file.filename}, {reddit_file.filename}")
        
        # Save uploaded files to temporary location
        news_path = os.path.join(tempfile.gettempdir(), f"news_{uuid.uuid4()}_{news_file.filename}")
        reddit_path = os.path.join(tempfile.gettempdir(), f"reddit_{uuid.uuid4()}_{reddit_file.filename}")
        temp_files.extend([news_path, reddit_path])
        
        # Write uploaded files
        with open(news_path, "wb") as f:
            shutil.copyfileobj(news_file.file, f)
        
        with open(reddit_path, "wb") as f:
            shutil.copyfileobj(reddit_file.file, f)
        
        logger.info("Files saved, starting analysis...")
        
        # Run comprehensive analysis
        email_recipients = [recipient_email] if send_email and recipient_email else None
        
        results = BOT.run_comprehensive_analysis_with_distribution(
            news_file=news_path,
            reddit_file=reddit_path,
            email_recipients=email_recipients,
            create_download_package=create_download_package
        )
        
        # Register files for download
        downloadables = {}
        
        if results.get("pdf_reports"):
            downloadables["pdf_reports"] = []
            for pdf_path in results["pdf_reports"]:
                if pdf_path and os.path.exists(pdf_path):
                    downloadables["pdf_reports"].append({
                        "id": register_file(pdf_path),
                        "name": os.path.basename(pdf_path),
                        "type": "pdf"
                    })
        
        if results.get("generated_charts"):
            downloadables["charts"] = []
            for chart_path in results["generated_charts"]:
                if chart_path and os.path.exists(chart_path):
                    downloadables["charts"].append({
                        "id": register_file(chart_path),
                        "name": os.path.basename(chart_path),
                        "type": "image"
                    })
        
        if results.get("download_package") and os.path.exists(results["download_package"]):
            downloadables["package"] = {
                "id": register_file(results["download_package"]),
                "name": os.path.basename(results["download_package"]),
                "type": "zip"
            }
        
        # Add downloadables to results
        results["downloadables"] = downloadables
        
        # Clean up uploaded temp files
        cleanup_temp_files(temp_files)
        
        logger.info("Analysis completed successfully")
        return AnalysisResponse(**results)
        
    except HTTPException:
        cleanup_temp_files(temp_files)
        raise
    
    except Exception as e:
        cleanup_temp_files(temp_files)
        logger.error(f"Error in analysis endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download endpoint for generated files
    
    Args:
        file_id: Unique file identifier returned by analysis endpoint
    """
    if file_id not in FILE_REGISTRY:
        logger.warning(f"Download requested for unknown file ID: {file_id}")
        raise HTTPException(
            status_code=404,
            detail="File not found. File may have expired or never existed."
        )
    
    file_path = FILE_REGISTRY[file_id]
    
    if not os.path.exists(file_path):
        logger.warning(f"Download requested for missing file: {file_path}")
        # Clean up registry
        del FILE_REGISTRY[file_id]
        raise HTTPException(
            status_code=410,
            detail="File no longer available on server"
        )
    
    filename = os.path.basename(file_path)
    logger.info(f"Serving download for file: {filename}")
    
    return FileResponse(
        file_path, 
        filename=filename,
        headers={"Cache-Control": "no-cache"}
    )
from backend import EmailHandler

@app.post("/send-email")
async def send_email_endpoint(
    recipient_email: str = Form(..., description="Email recipient"),
    file_ids: List[str] = Form(..., description="List of file IDs to send")
):
    """
    Send selected reports/files to a recipient via email.
    """
    try:
        if not recipient_email:
            raise HTTPException(status_code=400, detail="Recipient email required")

        # Check email config
        if not os.getenv("EMAIL_ADDRESS") or not os.getenv("EMAIL_APP_PASSWORD"):
            raise HTTPException(status_code=500, detail="Email not configured on server")

        email_handler = EmailHandler(
            os.getenv("EMAIL_ADDRESS"),
            os.getenv("EMAIL_APP_PASSWORD")
        )

        # Collect file paths
        attachments = []
        for fid in file_ids:
            if fid not in FILE_REGISTRY:
                continue
            file_path = FILE_REGISTRY[fid]
            if os.path.exists(file_path):
                attachments.append({
                    "path": file_path,
                    "name": os.path.basename(file_path)
                })

        if not attachments:
            raise HTTPException(status_code=400, detail="No valid files found to send")

        # Send email
        success = email_handler.send_email_with_attachments(
            recipient_email=recipient_email,
            subject="Your Financial Analysis Reports",
            body="Attached are the analysis reports you requested.",
            attachments=attachments
        )

        if success:
            return {"message": f"Email sent successfully to {recipient_email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/cleanup")
async def cleanup_files():
    """Administrative endpoint to clean up old files"""
    cleaned_count = 0
    
    # Clean up registered files
    files_to_remove = []
    for file_id, file_path in FILE_REGISTRY.items():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                cleaned_count += 1
            files_to_remove.append(file_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    # Remove from registry
    for file_id in files_to_remove:
        del FILE_REGISTRY[file_id]
    
    logger.info(f"Cleaned up {cleaned_count} files")
    return {"message": f"Cleaned up {cleaned_count} files", "cleaned_count": cleaned_count}

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "registered_files": len(FILE_REGISTRY),
        "bot_status": "initialized" if BOT else "not_initialized",
        "temp_directory": tempfile.gettempdir(),
        "available_endpoints": [
            "/health", "/analyze", "/download/{file_id}", 
            "/cleanup", "/stats", "/docs"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Check server logs"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", "5050"))
    
    # SSL configuration (optional)
    ssl_keyfile = os.getenv("SSL_KEYFILE", "key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "cert.pem")
    
    ssl_config = {}
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        ssl_config = {
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        }
        logger.info("SSL configuration loaded")
    else:
        logger.info("Running without SSL (HTTP only)")
    
    logger.info(f"Starting Financial Analysis API on {HOST}:{PORT}")
    
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=True,
        **ssl_config
    )