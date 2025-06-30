import os
import shutil
import logging
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from app import app, db
from models import VideoShort, YouTubeCredentials, UploadStatus
from oauth_handler import OAuthHandler

logger = logging.getLogger(__name__)

class YouTubeUploader:
    def __init__(self):
        self.oauth_handler = OAuthHandler()
        
    def upload_short(self, short_id, user_email):
        """Upload a video short to YouTube"""
        with app.app_context():
            short = VideoShort.query.get(short_id)
            if not short:
                logger.error(f"Short {short_id} not found")
                return
            
            try:
                logger.info(f"Starting YouTube upload for short {short_id}")
                
                # Update status
                short.upload_status = UploadStatus.UPLOADING
                db.session.commit()
                
                # Get valid credentials
                creds = self._get_valid_credentials(user_email)
                if not creds:
                    raise Exception("No valid YouTube credentials found")
                
                # Build YouTube service
                youtube = build('youtube', 'v3', credentials=creds)
                
                # Upload video
                video_id = self._upload_video(youtube, short)
                
                # Update short with YouTube video ID
                short.youtube_video_id = video_id
                short.upload_status = UploadStatus.COMPLETED
                db.session.commit()
                
                # Cleanup files after successful upload
                self._cleanup_short_files(short)
                
                logger.info(f"Successfully uploaded short {short_id} to YouTube: {video_id}")
                
            except Exception as e:
                logger.error(f"Failed to upload short {short_id}: {e}")
                short.upload_status = UploadStatus.FAILED
                short.upload_error = str(e)
                db.session.commit()
    
    def _get_valid_credentials(self, user_email):
        """Get valid YouTube credentials, refreshing if necessary"""
        try:
            db_creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()
            if not db_creds:
                return None
            
            # Create OAuth2 credentials object
            creds = Credentials(
                token=db_creds.access_token,
                refresh_token=db_creds.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.oauth_handler.client_id,
                client_secret=self.oauth_handler.client_secret
            )
            
            # Refresh if expired
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                
                # Update database with new token
                db_creds.access_token = creds.token
                if creds.expiry:
                    db_creds.token_expires = creds.expiry
                db.session.commit()
                
                logger.info(f"Refreshed credentials for {user_email}")
            
            return creds
            
        except Exception as e:
            logger.error(f"Failed to get valid credentials: {e}")
            return None
    
    def _upload_video(self, youtube, short):
        """Upload video to YouTube"""
        try:
            if not short.output_path or not os.path.exists(short.output_path):
                raise Exception("Video file not found")
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': short.title or f"YouTube Short #{short.id}",
                    'description': short.description or "Generated YouTube Short",
                    'tags': short.tags or ['shorts', 'viral'],
                    'categoryId': '22',  # People & Blogs
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': 'public',  # Can be 'private', 'unlisted', or 'public'
                    'madeForKids': False,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Create media upload object
            media = MediaFileUpload(
                short.output_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/mp4'
            )
            
            # Insert video
            insert_request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute upload
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    logger.info(f"Upload progress {int(status.progress() * 100)}%")
            
            if 'id' not in response:
                raise Exception(f"Upload failed: {response}")
            
            video_id = response['id']
            logger.info(f"Video uploaded successfully: https://www.youtube.com/watch?v={video_id}")
            
            return video_id
            
        except Exception as e:
            raise Exception(f"Video upload failed: {e}")
    
    def _cleanup_short_files(self, short):
        """Clean up files after successful upload"""
        try:
            # Delete the short video file
            if short.output_path and os.path.exists(short.output_path):
                os.remove(short.output_path)
                logger.info(f"Deleted short video file: {short.output_path}")
            
            # Delete thumbnail if exists
            if short.thumbnail_path and os.path.exists(short.thumbnail_path):
                os.remove(short.thumbnail_path)
                logger.info(f"Deleted thumbnail file: {short.thumbnail_path}")
            
            # Check if this was the last short for the job
            job = short.job
            remaining_shorts = VideoShort.query.filter_by(job_id=job.id).filter(
                VideoShort.upload_status != UploadStatus.COMPLETED
            ).count()
            
            if remaining_shorts == 0:
                # All shorts uploaded, clean up job files
                self._cleanup_job_files(job)
                
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
    
    def _cleanup_job_files(self, job):
        """Clean up all files related to a job after all shorts are uploaded"""
        try:
            # Delete original video file
            if job.video_path and os.path.exists(job.video_path):
                os.remove(job.video_path)
                logger.info(f"Deleted original video file: {job.video_path}")
            
            # Delete audio file
            if job.audio_path and os.path.exists(job.audio_path):
                os.remove(job.audio_path)
                logger.info(f"Deleted audio file: {job.audio_path}")
            
            # Delete transcript file
            if job.transcript_path and os.path.exists(job.transcript_path):
                os.remove(job.transcript_path)
                logger.info(f"Deleted transcript file: {job.transcript_path}")
            
            # Clean up empty directories
            self._cleanup_empty_directories()
            
            logger.info(f"Completed cleanup for job {job.id}")
            
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")
    
    def _cleanup_empty_directories(self):
        """Remove empty uploads and temp directories"""
        try:
            directories_to_clean = ['uploads', 'temp', 'outputs']
            
            for dir_name in directories_to_clean:
                if os.path.exists(dir_name):
                    # Check if directory is empty
                    if not os.listdir(dir_name):
                        # Directory is empty, but don't delete it as it might be needed later
                        logger.info(f"Directory {dir_name} is empty and ready for next use")
                    else:
                        # Remove any remaining temporary files older than 1 hour
                        import time
                        current_time = time.time()
                        for filename in os.listdir(dir_name):
                            file_path = os.path.join(dir_name, filename)
                            if os.path.isfile(file_path):
                                file_age = current_time - os.path.getmtime(file_path)
                                # Remove files older than 1 hour (3600 seconds)
                                if file_age > 3600:
                                    os.remove(file_path)
                                    logger.info(f"Removed old temporary file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during directory cleanup: {e}")
