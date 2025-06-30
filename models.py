from datetime import datetime, timezone
from app import db
from sqlalchemy import Enum, Text, JSON
import enum

class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    EDITING = "editing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"

class UploadStatus(enum.Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoJob(db.Model):
    __tablename__ = 'video_jobs'

    id = db.Column(db.Integer, primary_key=True)
    youtube_url = db.Column(db.String(500), nullable=False)
    title = db.Column(db.String(200))
    duration = db.Column(db.Integer)  # in seconds
    video_quality = db.Column(db.String(20), default='1080p')
    aspect_ratio = db.Column(db.String(10), default='9:16')
    num_shorts = db.Column(db.Integer, default=3)
    short_length = db.Column(db.Integer, default=30)  # Length in seconds
    user_email = db.Column(db.String(120))
    language = db.Column(db.String(20), default='hinglish')
    auto_upload = db.Column(db.Boolean, default=False)

    # Processing status
    status = db.Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    progress = db.Column(db.Integer, default=0)
    error_message = db.Column(Text)

    # File paths
    video_path = db.Column(db.String(500))
    audio_path = db.Column(db.String(500))
    transcript_path = db.Column(db.String(500))

    # Video metadata
    video_info = db.Column(JSON)

    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    shorts = db.relationship('VideoShort', backref='job', lazy=True, cascade='all, delete-orphan')
    transcript_segments = db.relationship('TranscriptSegment', backref='job', lazy=True, cascade='all, delete-orphan')

class VideoShort(db.Model):
    __tablename__ = 'video_shorts'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('video_jobs.id'), nullable=False)

    # Segment info
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Float)

    # AI analysis scores
    engagement_score = db.Column(db.Float, default=0.0)
    emotion_score = db.Column(db.Float, default=0.0)
    viral_potential = db.Column(db.Float, default=0.0)
    quotability = db.Column(db.Float, default=0.0)
    overall_score = db.Column(db.Float, default=0.0)

    # Content analysis
    emotions_detected = db.Column(JSON)
    keywords = db.Column(JSON)
    analysis_notes = db.Column(Text)

    # Generated content
    title = db.Column(db.String(200))
    description = db.Column(Text)
    tags = db.Column(JSON)

    # File paths
    output_path = db.Column(db.String(500))
    thumbnail_path = db.Column(db.String(500))

    # Upload status
    upload_status = db.Column(Enum(UploadStatus), default=UploadStatus.PENDING)
    youtube_video_id = db.Column(db.String(50))
    upload_error = db.Column(Text)

    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class TranscriptSegment(db.Model):
    __tablename__ = 'transcript_segments'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('video_jobs.id'), nullable=False)

    # Segment timing
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    text = db.Column(Text, nullable=False)

    # AI analysis scores
    engagement_score = db.Column(db.Float, default=0.0)
    emotion_score = db.Column(db.Float, default=0.0)
    viral_potential = db.Column(db.Float, default=0.0)
    quotability = db.Column(db.Float, default=0.0)
    overall_score = db.Column(db.Float, default=0.0)

    # Content analysis
    emotions_detected = db.Column(JSON)
    keywords = db.Column(JSON)
    analysis_notes = db.Column(Text)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class YouTubeCredentials(db.Model):
    __tablename__ = 'youtube_credentials'

    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), unique=True, nullable=False)

    # OAuth tokens
    access_token = db.Column(Text, nullable=False)
    refresh_token = db.Column(Text)
    token_expires = db.Column(db.DateTime)
    scope = db.Column(Text)

    # Channel information
    channel_id = db.Column(db.String(100))
    channel_title = db.Column(db.String(200))
    channel_thumbnail = db.Column(db.String(500))

    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))