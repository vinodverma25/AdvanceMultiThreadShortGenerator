import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///youtube_shorts_generator.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure API keys
app.config["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")
app.config["YOUTUBE_CLIENT_ID"] = os.environ.get("YOUTUBE_CLIENT_ID")
app.config["YOUTUBE_CLIENT_SECRET"] = os.environ.get("YOUTUBE_CLIENT_SECRET")
# Configure YouTube redirect URI for Replit environment
replit_domain = os.environ.get("REPLIT_DEV_DOMAIN")
if replit_domain:
    app.config["YOUTUBE_REDIRECT_URI"] = f"https://{replit_domain}/youtube/callback"
else:
    app.config["YOUTUBE_REDIRECT_URI"] = os.environ.get("YOUTUBE_REDIRECT_URI", "http://localhost:5000/youtube/callback")

# Initialize the app with the extension
db.init_app(app)

# Create upload and output directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('temp', exist_ok=True)

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

# Import routes after app configuration
import routes

# Print setup instructions
if not os.environ.get("GEMINI_API_KEY"):
    print("""
    Gemini AI Setup Required:
    1. Go to https://makersuite.google.com/app/apikey
    2. Create API keys (primary + backups for fallback)
    3. Set environment variables:
       - GEMINI_API_KEY=your_primary_key (required)
       - GEMINI_API_KEY_1=your_backup_key_1 (optional)
       - GEMINI_API_KEY_2=your_backup_key_2 (optional)
       - GEMINI_API_KEY_3=your_backup_key_3 (optional)
       - GEMINI_API_KEY_4=your_backup_key_4 (optional)
    """)

if not os.environ.get("YOUTUBE_CLIENT_ID"):
    current_domain = os.environ.get("REPLIT_DEV_DOMAIN", "localhost:5000")
    redirect_uri = f"https://{current_domain}/youtube/callback" if current_domain != "localhost:5000" else "http://localhost:5000/youtube/callback"
    print(f"""
    YouTube Data API Setup Required:
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create a new OAuth 2.0 Client ID
    3. Add {redirect_uri} to Authorized redirect URIs
    4. Set environment variables:
       - YOUTUBE_CLIENT_ID=your_client_id
       - YOUTUBE_CLIENT_SECRET=your_client_secret
    
    IMPORTANT: Your current redirect URI is: {redirect_uri}
    """)
