# YouTube Shorts Generator - Deployment Guide

## Requirements File Content

Create a `requirements.txt` file with these dependencies:

```
Flask==3.0.0
Flask-SQLAlchemy==3.1.1
gunicorn==21.2.0
psycopg2-binary==2.9.9
yt-dlp==2023.12.30
moviepy==1.0.3
google-generativeai==0.3.2
google-api-python-client==2.111.0
google-auth==2.25.2
google-auth-oauthlib==1.2.0
google-auth-httplib2==0.2.0
requests==2.31.0
Pillow==10.1.0
pydantic==2.5.2
python-dotenv==1.0.0
```

## Important: Streamlit Cloud Compatibility

**This Flask application cannot be deployed on Streamlit Cloud** because:
- Streamlit Cloud only supports Streamlit apps, not Flask applications
- This app requires background processing, file uploads, and database operations
- Flask apps need a different deployment architecture

## Recommended Deployment Options

### Option 1: Replit Deployments (Recommended)
- **Best choice** - Already configured and running
- Click the "Deploy" button in Replit
- Automatic HTTPS, database, and domain setup
- No additional configuration needed

### Option 2: Railway
- Connect your GitHub repository
- Add environment variables (API keys)
- Automatic deployment from main branch
- Built-in PostgreSQL database

### Option 3: Render
- Connect GitHub repository
- Add environment variables
- Choose "Web Service" type
- Built-in PostgreSQL database

### Option 4: Heroku
- Install Heroku CLI
- Create `Procfile`: `web: gunicorn --bind 0.0.0.0:$PORT main:app`
- Add Heroku PostgreSQL addon
- Deploy via Git

## Environment Variables Required

All deployment platforms need these environment variables:
```
DATABASE_URL=your_postgresql_connection_string
GEMINI_API_KEY=your_gemini_api_key
YOUTUBE_CLIENT_ID=your_youtube_client_id
YOUTUBE_CLIENT_SECRET=your_youtube_client_secret
SESSION_SECRET=your_random_secret_key
```

## Deployment Commands

For platforms that support it, use:
```bash
# Start command
gunicorn --bind 0.0.0.0:$PORT main:app

# Or with workers
gunicorn --bind 0.0.0.0:$PORT --workers 2 main:app
```

## File Structure Required

Ensure these files are in your project root:
- `main.py` (entry point)
- `app.py` (Flask app setup)
- `requirements.txt` (dependencies)
- All Python modules (routes.py, models.py, etc.)
- `templates/` folder (HTML templates)
- `static/` folder (CSS/JS files)

## Recommendation

Since your app is already running perfectly on Replit, the easiest deployment option is **Replit Deployments**. Just click the Deploy button in your Replit project for a production-ready deployment with automatic scaling and monitoring.