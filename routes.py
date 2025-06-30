from flask import render_template, request, redirect, url_for, flash, jsonify, send_file, session
from app import app, db
from models import VideoJob, VideoShort, TranscriptSegment, YouTubeCredentials, ProcessingStatus, UploadStatus
from oauth_handler import OAuthHandler
from youtube_uploader import YouTubeUploader
import threading
import os
import re
import shutil
import subprocess
from video_processor import VideoProcessor

def is_valid_youtube_url(url):
    """Validate if the URL is a valid YouTube URL"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return youtube_regex.match(url) is not None

@app.route('/')
def index():
    """Home page with URL input form"""
    recent_jobs = VideoJob.query.order_by(VideoJob.created_at.desc()).limit(5).all()
    # Check if user has YouTube credentials
    user_email = session.get('user_email')
    youtube_connected = False
    if user_email:
        creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()
        youtube_connected = creds is not None
    
    return render_template('index.html', 
                         recent_jobs=recent_jobs,
                         youtube_connected=youtube_connected,
                         user_email=user_email)

@app.route('/submit', methods=['POST'])
def submit_video():
    """Submit a YouTube URL for processing"""
    youtube_url = request.form.get('youtube_url', '').strip()
    video_quality = request.form.get('video_quality', '1080p')
    aspect_ratio = request.form.get('aspect_ratio', '9:16')
    num_shorts = int(request.form.get('num_shorts', 3))
    short_length = int(request.form.get('short_length', 30))
    language = request.form.get('language', 'hinglish')
    auto_upload = request.form.get('auto_upload') == '1'
    
    if not youtube_url:
        flash('Please enter a YouTube URL', 'error')
        return redirect(url_for('index'))
    
    if not is_valid_youtube_url(youtube_url):
        flash('Please enter a valid YouTube URL', 'error')
        return redirect(url_for('index'))
    
    # Check if URL is already being processed
    existing_job = VideoJob.query.filter_by(youtube_url=youtube_url).filter(
        VideoJob.status.in_([ProcessingStatus.PENDING, ProcessingStatus.DOWNLOADING, 
                           ProcessingStatus.TRANSCRIBING, ProcessingStatus.ANALYZING, 
                           ProcessingStatus.EDITING, ProcessingStatus.UPLOADING])
    ).first()
    
    if existing_job:
        flash('This video is already being processed', 'info')
        return redirect(url_for('process', job_id=existing_job.id))
    
    # Create new job
    job = VideoJob()
    job.youtube_url = youtube_url
    job.video_quality = video_quality
    job.aspect_ratio = aspect_ratio
    job.num_shorts = num_shorts
    job.short_length = short_length
    job.language = language
    job.auto_upload = auto_upload
    job.user_email = session.get('user_email')
    db.session.add(job)
    db.session.commit()
    
    # Start processing in background thread
    processor = VideoProcessor()
    thread = threading.Thread(target=processor.process_video, args=(job.id,))
    thread.daemon = True
    thread.start()
    
    flash('Video processing started! This may take several minutes.', 'success')
    return redirect(url_for('process', job_id=job.id))

@app.route('/process/<int:job_id>')
def process(job_id):
    """Show processing status page"""
    job = VideoJob.query.get_or_404(job_id)
    return render_template('process.html', job=job)

@app.route('/api/status/<int:job_id>')
def api_status(job_id):
    """API endpoint to get job status"""
    job = VideoJob.query.get_or_404(job_id)
    return jsonify({
        'status': job.status.value,
        'progress': job.progress,
        'error_message': job.error_message,
        'title': job.title,
        'shorts_count': len(job.shorts) if job.shorts else 0,
        'current_status_text': get_status_text(job.status)
    })

def get_status_text(status):
    """Get human-readable status text"""
    status_texts = {
        ProcessingStatus.PENDING: "Initializing...",
        ProcessingStatus.DOWNLOADING: "Downloading video in high quality...",
        ProcessingStatus.TRANSCRIBING: "Extracting audio and transcribing...",
        ProcessingStatus.ANALYZING: "Analyzing content with Gemini AI...",
        ProcessingStatus.EDITING: "Generating vertical shorts...",
        ProcessingStatus.UPLOADING: "Uploading to YouTube...",
        ProcessingStatus.COMPLETED: "Completed successfully!",
        ProcessingStatus.FAILED: "Processing failed"
    }
    return status_texts.get(status, "Unknown status")

@app.route('/results/<int:job_id>')
def results(job_id):
    """Show results page with generated shorts"""
    job = VideoJob.query.get_or_404(job_id)
    
    if job.status != ProcessingStatus.COMPLETED:
        flash('Video processing is not yet complete', 'warning')
        return redirect(url_for('process', job_id=job_id))
    
    shorts = VideoShort.query.filter_by(job_id=job_id).order_by(VideoShort.engagement_score.desc()).all()
    
    # Check YouTube connection
    user_email = session.get('user_email', job.user_email)
    youtube_connected = False
    if user_email:
        creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()
        youtube_connected = creds is not None
    
    return render_template('results.html', 
                         job=job, 
                         shorts=shorts,
                         youtube_connected=youtube_connected)

@app.route('/download/<int:short_id>')
def download_short(short_id):
    """Download a generated short video"""
    short = VideoShort.query.get_or_404(short_id)
    
    if not short.output_path or not os.path.exists(short.output_path):
        flash('Short video file not found', 'error')
        return redirect(url_for('results', job_id=short.job_id))
    
    return send_file(
        short.output_path,
        as_attachment=True,
        download_name=f"{short.title or 'short'}_{short.id}.mp4",
        mimetype='video/mp4'
    )

@app.route('/upload_short/<int:short_id>', methods=['POST'])
def upload_short(short_id):
    """Upload a short to YouTube"""
    short = VideoShort.query.get_or_404(short_id)
    user_email = session.get('user_email', short.job.user_email)
    
    if not user_email:
        flash('Please connect your YouTube account first', 'error')
        return redirect(url_for('youtube_auth'))
    
    # Check if user has valid YouTube credentials
    creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()
    if not creds:
        flash('Please connect your YouTube account first', 'error')
        return redirect(url_for('youtube_auth'))
    
    # Start upload in background
    uploader = YouTubeUploader()
    thread = threading.Thread(target=uploader.upload_short, args=(short.id, user_email))
    thread.daemon = True
    thread.start()
    
    # Update status
    short.upload_status = UploadStatus.PENDING
    db.session.commit()
    
    flash('Upload started! This may take a few minutes.', 'success')
    return redirect(url_for('results', job_id=short.job_id))

@app.route('/youtube/auth')
def youtube_auth():
    """Start YouTube OAuth process"""
    try:
        oauth_handler = OAuthHandler()
        auth_url = oauth_handler.get_authorization_url()
        return redirect(auth_url)
    except Exception as e:
        flash(f'YouTube authentication setup failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/youtube/callback')
def youtube_callback():
    """Handle YouTube OAuth callback"""
    code = request.args.get('code')
    state = request.args.get('state')
    error = request.args.get('error')
    
    if error:
        flash(f'YouTube authentication failed: {error}', 'error')
        return redirect(url_for('index'))
    
    if not code:
        flash('No authorization code received', 'error')
        return redirect(url_for('index'))
    
    try:
        oauth_handler = OAuthHandler()
        result = oauth_handler.exchange_code_for_tokens(code, state)
        
        # Store user email in session
        session['user_email'] = result['email']
        
        flash(f'Successfully connected YouTube account: {result["email"]}', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        flash(f'Failed to connect YouTube account: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/youtube/disconnect', methods=['POST'])
def youtube_disconnect():
    """Disconnect YouTube account"""
    user_email = session.get('user_email')
    if not user_email:
        flash('No YouTube account connected', 'error')
        return redirect(url_for('index'))
    
    try:
        oauth_handler = OAuthHandler()
        oauth_handler.revoke_token(user_email)
        session.pop('user_email', None)
        flash('YouTube account disconnected successfully', 'success')
    except Exception as e:
        flash(f'Failed to disconnect YouTube account: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/jobs')
def list_jobs():
    """List all processing jobs"""
    page = request.args.get('page', 1, type=int)
    jobs = VideoJob.query.order_by(VideoJob.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template('jobs.html', jobs=jobs)

@app.route('/delete/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    """Delete a job and its associated files"""
    job = VideoJob.query.get_or_404(job_id)
    
    # Delete associated files
    files_to_delete = [job.video_path, job.audio_path, job.transcript_path]
    for short in job.shorts:
        files_to_delete.extend([short.output_path, short.thumbnail_path])
    
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.warning(f"Could not delete file {file_path}: {e}")
    
    # Delete from database
    db.session.delete(job)
    db.session.commit()
    
    flash('Job and associated files deleted successfully', 'success')
    return redirect(url_for('index'))

@app.route('/clear-data', methods=['POST'])
def clear_all_data():
    """Clear all data including cache, files, database, and .git folder"""
    try:
        # Stop any ongoing processing
        flash('Starting data cleanup...', 'info')
        
        # 1. Clear database tables
        try:
            db.session.query(VideoShort).delete()
            db.session.query(TranscriptSegment).delete()  
            db.session.query(VideoJob).delete()
            db.session.query(YouTubeCredentials).delete()
            db.session.commit()
            app.logger.info("Database tables cleared")
        except Exception as e:
            app.logger.error(f"Error clearing database: {e}")
            db.session.rollback()
        
        # 2. Remove file directories
        directories_to_clear = ['uploads', 'outputs', 'temp', '__pycache__']
        for directory in directories_to_clear:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    app.logger.info(f"Removed directory: {directory}")
                except Exception as e:
                    app.logger.warning(f"Failed to remove {directory}: {e}")
        
        # 3. Remove .git folder if it exists
        if os.path.exists('.git'):
            try:
                shutil.rmtree('.git')
                app.logger.info("Removed .git folder")
            except Exception as e:
                app.logger.warning(f"Failed to remove .git folder: {e}")
        
        # 4. Clear Python cache files
        cache_patterns = ['*.pyc', '*.pyo', '*__pycache__*']
        for pattern in cache_patterns:
            try:
                result = subprocess.run(['find', '.', '-name', pattern, '-delete'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    app.logger.info(f"Cleared cache pattern: {pattern}")
            except Exception as e:
                app.logger.warning(f"Failed to clear cache {pattern}: {e}")
        
        # 5. Clear session data
        session.clear()
        
        # 6. Recreate necessary directories
        for directory in ['uploads', 'outputs', 'temp']:
            os.makedirs(directory, exist_ok=True)
        
        flash('All data, cache, and files have been cleared successfully!', 'success')
        app.logger.info("Complete data cleanup finished")
        
    except Exception as e:
        flash(f'Error during data cleanup: {str(e)}', 'error')
        app.logger.error(f"Data cleanup failed: {e}")
    
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('base.html', error_title="Page Not Found", 
                         error_message="The page you're looking for doesn't exist."), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('base.html', error_title="Internal Server Error", 
                         error_message="An unexpected error occurred."), 500
