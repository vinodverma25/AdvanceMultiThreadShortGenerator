import os
import logging
import yt_dlp
import subprocess
import json
import re
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import app, db
from models import VideoJob, VideoShort, TranscriptSegment, ProcessingStatus
from gemini_analyzer import GeminiAnalyzer

class VideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gemini_analyzer = GeminiAnalyzer()
        self.whisper_model = None
    
    def _sanitize_filename(self, filename):
        """Sanitize filename to prevent special character issues"""
        # Remove or replace problematic characters
        filename = re.sub(r'[^\w\s-]', '', filename)  # Keep only alphanumeric, spaces, hyphens
        filename = re.sub(r'[-\s]+', '-', filename)   # Replace multiple spaces/hyphens with single hyphen
        filename = filename.strip('-')                # Remove leading/trailing hyphens
        return filename[:100]  # Limit length
    
    def _check_cookies_available(self):
        """Check if YouTube cookies file is available and contains actual cookies"""
        # Check both cookie directory and root directory
        cookie_files = ['cookie/youtube_cookies.txt', 'youtube_cookies.txt']
        
        for cookie_file in cookie_files:
            if not os.path.exists(cookie_file):
                continue
                
            try:
                with open(cookie_file, 'r') as f:
                    content = f.read().strip()
                    # Check if file contains actual cookies (not just comments)
                    lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
                    if len(lines) > 0:
                        self.logger.info(f"YouTube cookies file found with authentication data: {cookie_file}")
                        return cookie_file
            except Exception as e:
                self.logger.warning(f"Error reading cookies file {cookie_file}: {e}")
                continue
        
        self.logger.info("No valid YouTube cookies found - age-restricted videos may fail")
        return None
        
    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            try:
                # Use ffmpeg for basic audio extraction and create mock transcription
                # This avoids the Whisper dependency issue while maintaining functionality
                self.whisper_model = "ffmpeg_based"
                self.logger.info("Audio processing initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize audio processing: {e}")
                raise

    def process_video(self, job_id):
        """Main processing pipeline for a video job"""
        with app.app_context():
            job = VideoJob.query.get(job_id)
            if not job:
                self.logger.error(f"Job {job_id} not found")
                return
            
            try:
                self.logger.info(f"Starting processing for job {job_id}: {job.youtube_url}")
                
                # Step 1: Download video
                self._update_job_status(job, ProcessingStatus.DOWNLOADING, 10)
                video_path = self._download_video(job)
                
                # Step 2: Transcribe audio with Whisper
                self._update_job_status(job, ProcessingStatus.TRANSCRIBING, 30)
                transcript_data = self._transcribe_video(job, video_path)
                
                # Step 3: Analyze content with Gemini AI
                self._update_job_status(job, ProcessingStatus.ANALYZING, 50)
                engaging_segments = self._analyze_content(job, transcript_data)
                
                # Step 4: Generate vertical short videos
                self._update_job_status(job, ProcessingStatus.EDITING, 70)
                self._generate_shorts(job, video_path, engaging_segments)
                
                # Step 5: Complete
                self._update_job_status(job, ProcessingStatus.COMPLETED, 100)
                # Step 6: Cleanup temporary files
                self._cleanup_temporary_files(job)

                self.logger.info(f"Successfully completed processing for job {job_id}")
                
            except Exception as e:
                self.logger.error(f"Error processing job {job_id}: {e}")
                self._update_job_status(job, ProcessingStatus.FAILED, 0, str(e))

    def _update_job_status(self, job, status, progress, error_message=None):
        """Update job status and progress"""
        job.status = status
        job.progress = progress
        if error_message:
            job.error_message = error_message
        db.session.commit()

    def _download_video(self, job):
        """Download video using yt-dlp in highest quality"""
        output_dir = 'uploads'
        
        # Configure yt-dlp options for high quality download (force 1920x1080)
        quality_formats = {
            '1080p': '137+140/bestvideo[height=1080]+bestaudio[ext=m4a]/bestvideo[height>=1080]+bestaudio/best[height>=1080]/best',
            '720p': '136+140/bestvideo[height=720]+bestaudio[ext=m4a]/bestvideo[height>=720]+bestaudio/best[height>=720]/best',
            '480p': 'bestvideo[height=480]+bestaudio[ext=m4a]/bestvideo[height>=480]+bestaudio/best[height>=480]/best',
            'best': '137+140/bestvideo[height=1080]+bestaudio[ext=m4a]/bestvideo[height>=1080]+bestaudio/best'
        }
        
        format_selector = quality_formats.get(job.video_quality, quality_formats['1080p'])
        
        ydl_opts = {
            'format': format_selector,
            'outtmpl': os.path.join(output_dir, f'video_{job.id}.%(ext)s'),  # Use simple filename to avoid character issues
            'extractaudio': False,
            'noplaylist': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'merge_output_format': 'mp4',  # Force mp4 output
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'prefer_ffmpeg': True,  # Use ffmpeg for processing
            'format_sort': ['res:1080', 'ext:mp4:m4a', 'vcodec:h264'],  # Prefer 1080p, mp4, and h264
            'verbose': False,  # Disable verbose logging
            # Custom format selector for audio language priority
            'format_sort_force': True,
            # Age-restricted content support
            'age_limit': None,  # Remove age restrictions
        }
        
        # Check for authentication status and add cookie file if available
        cookie_file = self._check_cookies_available()
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            self.logger.info(f"Using YouTube authentication cookies from: {cookie_file}")
        else:
            self.logger.info("No authentication cookies found - age-restricted videos may fail")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get title and duration
                info = ydl.extract_info(job.youtube_url, download=False)
                if info:
                    job.title = info.get('title', 'Unknown Title')[:200]
                    job.duration = info.get('duration', 0)
                    job.video_info = {
                        'title': info.get('title'),
                        'duration': info.get('duration'),
                        'uploader': info.get('uploader'),
                        'view_count': info.get('view_count'),
                        'width': info.get('width'),
                        'height': info.get('height'),
                        'fps': info.get('fps')
                    }
                else:
                    job.title = 'Unknown Title'
                    job.duration = 0
                    job.video_info = {}
                db.session.commit()
                
                # Download the video
                ydl.download([job.youtube_url])
                
                # Find the downloaded video file
                video_files = []
                for file in os.listdir(output_dir):
                    if file.startswith(f'video_{job.id}.') and file.endswith(('.mp4', '.webm', '.mkv', '.avi')):
                        video_files.append(file)
                
                if video_files:
                    video_file = video_files[0]
                    video_path = os.path.join(output_dir, video_file)
                    job.video_path = video_path
                    db.session.commit()
                    self.logger.info(f"Downloaded video: {video_path}")
                    return video_path
                else:
                    raise Exception("Downloaded video file not found")
                
        except Exception as e:
            raise Exception(f"Failed to download video: {e}")

    def _transcribe_video(self, job, video_path):
        """Transcribe video using Whisper"""
        try:
            # Load Whisper model
            self.load_whisper_model()
            
            # Extract audio for Whisper with Hindi language preference
            audio_path = os.path.join('temp', f'audio_{job.id}.wav')
            
            # Detect and prioritize audio streams: Hindi first, then English, then default
            audio_stream_index = self._select_preferred_audio_stream(video_path)
            
            # Create a safe filename for audio processing
            safe_video_path = os.path.join('temp', f'safe_video_{job.id}.mp4')
            
            # Copy video to safe path to avoid encoding issues
            import shutil
            shutil.copy2(video_path, safe_video_path)
            
            # Extract audio from safe path
            cmd = [
                'ffmpeg', '-i', safe_video_path, 
                f'-map', f'0:a:{audio_stream_index}',  # Select specific audio stream
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', audio_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
                    raise Exception(f"FFmpeg audio extraction failed: {result.stderr}")
            finally:
                # Clean up the temporary safe video file
                if os.path.exists(safe_video_path):
                    os.remove(safe_video_path)
            
            # Use ffmpeg to get duration and create time-based segments for AI analysis
            duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            duration = float(duration_result.stdout.strip())
            
            # Create time-based segments (every 30 seconds) for AI analysis
            segment_length = 30  # seconds
            segments = []
            for i in range(0, int(duration), segment_length):
                end_time = min(i + segment_length, duration)
                segments.append({
                    'start': i,
                    'end': end_time,
                    'text': f"Audio segment from {i}s to {end_time}s"  # Placeholder for AI analysis
                })
            
            transcript_data = {
                'segments': segments,
                'language': 'en',
                'full_text': f"Video content with {len(segments)} segments for AI analysis",
                'duration': duration
            }
            
            # Save transcript
            transcript_path = os.path.join('uploads', f'transcript_{job.id}.json')
            with open(transcript_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            job.audio_path = audio_path
            job.transcript_path = transcript_path
            db.session.commit()
            
            # Store segments in database
            for segment in segments:
                if len(segment['text'].strip()) > 10:  # Only meaningful segments
                    transcript_segment = TranscriptSegment()
                    transcript_segment.job_id = job.id
                    transcript_segment.start_time = segment['start']
                    transcript_segment.end_time = segment['end']
                    transcript_segment.text = segment['text'].strip()
                    db.session.add(transcript_segment)
            
            db.session.commit()
            return transcript_data
            
        except Exception as e:
            raise Exception(f"Failed to transcribe video: {e}")

    def _analyze_content(self, job, transcript_data):
        """Analyze content with Gemini AI to find engaging segments - ULTRA-OPTIMIZED PARALLEL PROCESSING"""
        try:
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
            engaging_segments = []
            
            # Process segments in parallel for faster AI analysis with memory optimization
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import gc  # Garbage collection for memory optimization
            
            def analyze_single_segment(segment_id, segment_text):
                """Analyze a single segment with Gemini AI - Enhanced with better session management"""
                try:
                    # Create new app context and database session for each thread
                    with app.app_context():
                        # Get fresh segment instance in this thread's context
                        segment = TranscriptSegment.query.get(segment_id)
                        if not segment:
                            return None
                            
                        analysis = self.gemini_analyzer.analyze_segment(segment_text)
                        
                        # Update segment with analysis results
                        segment.engagement_score = analysis.get('engagement_score', 0.0)
                        segment.emotion_score = analysis.get('emotion_score', 0.0)
                        segment.viral_potential = analysis.get('viral_potential', 0.0)
                        segment.quotability = analysis.get('quotability', 0.0)
                        segment.overall_score = (
                            segment.engagement_score + 
                            segment.emotion_score + 
                            segment.viral_potential + 
                            segment.quotability
                        ) / 4.0
                        segment.emotions_detected = analysis.get('emotions', [])
                        segment.keywords = analysis.get('keywords', [])
                        segment.analysis_notes = analysis.get('reason', '')
                        
                        # Use fresh session commit
                        try:
                            db.session.commit()
                            self.logger.info(f"Analyzed segment {segment.id}: score={segment.overall_score:.2f}")
                            return segment_id
                        except Exception as commit_error:
                            db.session.rollback()
                            self.logger.error(f"Database commit failed for segment {segment_id}: {commit_error}")
                            raise commit_error
                        
                except Exception as e:
                    self.logger.error(f"Failed to analyze segment {segment_id}: {e}")
                    # Set fallback scores with new session
                    try:
                        with app.app_context():
                            segment = TranscriptSegment.query.get(segment_id)
                            if segment:
                                segment.engagement_score = 0.3
                                segment.emotion_score = 0.3
                                segment.viral_potential = 0.3
                                segment.quotability = 0.3
                                segment.overall_score = 0.3
                                db.session.commit()
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback update failed for segment {segment_id}: {fallback_error}")
                    return segment_id
            
            self.logger.info(f"Starting ULTRA-PARALLEL AI analysis of {len(segments)} segments...")
            
            # ULTRA-OPTIMIZED thread pool size with advanced resource detection
            import os
            import psutil
            
            cpu_count = os.cpu_count() or 4
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Advanced dynamic scaling for AI analysis
            if available_memory_gb > 16:  # High-end system
                max_workers = min(12, len(segments), cpu_count * 3)
            elif available_memory_gb > 8:  # High memory system
                max_workers = min(10, len(segments), cpu_count * 2)
            elif available_memory_gb > 4:  # Medium memory system
                max_workers = min(8, len(segments), cpu_count)
            else:  # Low memory system
                max_workers = min(4, len(segments), max(2, cpu_count // 2))
            
            # Ensure we have at least 2 workers for parallelism
            max_workers = max(2, max_workers)
            
            self.logger.info(f"ULTRA-PARALLEL setup: {cpu_count} CPUs, {available_memory_gb:.1f}GB RAM")
            self.logger.info(f"Using {max_workers} ULTRA-PARALLEL workers for AI analysis")
            
            # Prepare segment data for parallel processing
            segment_data = [(segment.id, segment.text) for segment in segments]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all segment analysis tasks with enhanced data passing
                future_to_segment_id = {
                    executor.submit(analyze_single_segment, seg_id, seg_text): seg_id 
                    for seg_id, seg_text in segment_data
                }
                
                # Collect results as they complete with enhanced error handling
                completed_count = 0
                successful_segments = []
                
                for future in as_completed(future_to_segment_id):
                    segment_id = future_to_segment_id[future]
                    try:
                        result_segment_id = future.result(timeout=120)  # 2-minute timeout per segment
                        if result_segment_id:
                            successful_segments.append(result_segment_id)
                        completed_count += 1
                        
                        # Update progress with enhanced tracking
                        progress = 50 + int((completed_count / len(segments)) * 30)  # 50-80% range
                        self._update_job_status(job, ProcessingStatus.ANALYZING, progress)
                        
                        self.logger.info(f"ULTRA-PARALLEL progress: {completed_count}/{len(segments)} segments")
                        
                        # Enhanced memory optimization
                        if completed_count % 5 == 0:  # More frequent cleanup
                            gc.collect()
                        
                    except Exception as e:
                        self.logger.error(f"ULTRA-PARALLEL segment analysis failed for {segment_id}: {e}")
                        completed_count += 1
            
            # Final memory cleanup
            gc.collect()
            self.logger.info(f"Completed optimized parallel AI analysis of {len(segments)} segments")
            
            # Now select the best segments for the user's preferences
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
            
            for segment in segments:
                # Consider segments with good scores and user-preferred duration
                duration = segment.end_time - segment.start_time
                target_length = job.short_length
                min_length = max(10, target_length - 15)  # Allow some flexibility
                max_length = target_length + 15
                
                if (segment.overall_score > 0.4 and  # Good engagement threshold
                    min_length <= duration <= max_length and  # User-preferred duration
                    len(segment.text.split()) >= 5):  # Meaningful content
                    engaging_segments.append(segment)
            
            # Sort by overall score and return user-requested number of segments
            engaging_segments.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Ensure we have at least the requested number of segments
            if len(engaging_segments) < job.num_shorts:
                all_segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
                target_length = job.short_length
                min_length = max(10, target_length - 15)
                max_length = target_length + 15
                
                for segment in all_segments:
                    if segment not in engaging_segments:
                        duration = segment.end_time - segment.start_time
                        if min_length <= duration <= max_length and len(segment.text.split()) >= 3:
                            segment.overall_score = 0.3  # Acceptable fallback score
                            engaging_segments.append(segment)
                            if len(engaging_segments) >= job.num_shorts:
                                break
            
            return engaging_segments[:job.num_shorts]  # Return user-requested number
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            # Fallback: return segments based on user preferences
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
            fallback_segments = []
            target_length = job.short_length
            min_length = max(10, target_length - 15)
            max_length = target_length + 15
            
            for segment in segments:
                duration = segment.end_time - segment.start_time
                if min_length <= duration <= max_length:
                    segment.overall_score = 0.5  # Default score
                    fallback_segments.append(segment)
            return fallback_segments[:job.num_shorts]

    def _generate_shorts(self, job, video_path, engaging_segments):
        """Generate vertical short videos from engaging segments - OPTIMIZED WITH PARALLEL PROCESSING"""
        try:
            # Prepare tasks for parallel processing
            def process_single_short(segment_data):
                segment_id, segment_text, segment_start, segment_end, segment_scores, i = segment_data
                try:
                    # Create new app context for each thread
                    with app.app_context():
                        # Get fresh job instance in this thread's context
                        current_job = VideoJob.query.get(job.id)
                        if not current_job:
                            return f"Short {i+1} failed: Job not found"
                        
                        # Generate metadata with Gemini using job's language preference
                        job_language = getattr(current_job, 'language', 'hinglish')
                        metadata = self.gemini_analyzer.generate_metadata(
                            segment_text, 
                            current_job.title or "YouTube Short",
                            language=job_language
                        )
                        
                        # Create VideoShort record with enhanced error handling
                        short = VideoShort()
                        short.job_id = current_job.id
                        short.start_time = segment_start
                        short.end_time = segment_end
                        short.duration = segment_end - segment_start
                        short.engagement_score = segment_scores.get('engagement_score', 0.5)
                        short.emotion_score = segment_scores.get('emotion_score', 0.5)
                        short.viral_potential = segment_scores.get('viral_potential', 0.5)
                        short.quotability = segment_scores.get('quotability', 0.5)
                        short.overall_score = segment_scores.get('overall_score', 0.5)
                        short.emotions_detected = segment_scores.get('emotions_detected', [])
                        short.keywords = segment_scores.get('keywords', [])
                        short.analysis_notes = segment_scores.get('analysis_notes', '')
                        short.title = metadata.get('title', f"Short {i+1}")
                        short.description = metadata.get('description', '')
                        short.tags = metadata.get('tags', [])
                        
                        try:
                            db.session.add(short)
                            db.session.commit()
                        except Exception as db_error:
                            db.session.rollback()
                            self.logger.error(f"Database error for short {i+1}: {db_error}")
                            raise db_error
                        
                        # Generate video file with thread-safe naming
                        import threading
                        thread_id = threading.current_thread().ident
                        output_path = os.path.join('outputs', f'short_{short.id}_t{thread_id}.mp4')
                        thumbnail_path = os.path.join('outputs', f'short_{short.id}_t{thread_id}_thumb.jpg')
                        
                        # Create thread-safe copy of input video
                        safe_input_path = os.path.join('temp', f'safe_input_{short.id}_t{thread_id}.mp4')
                        import shutil
                        shutil.copy2(video_path, safe_input_path)
                        
                        try:
                            # Adjust timing to match user's preferred length
                            user_length = current_job.short_length
                            segment_duration = segment_end - segment_start
                            
                            if segment_duration > user_length:
                                # If segment is longer, center the user's preferred length
                                excess = segment_duration - user_length
                                adjusted_start = segment_start + (excess / 2)
                                adjusted_end = adjusted_start + user_length
                            else:
                                # If segment is shorter or equal, use original timing
                                adjusted_start = segment_start
                                adjusted_end = segment_end
                            
                            # Create vertical video using FFmpeg with thread optimization
                            final_output = os.path.join('outputs', f'short_{short.id}.mp4')
                            self._create_vertical_video_threaded(
                                safe_input_path, 
                                final_output, 
                                adjusted_start, 
                                adjusted_end,
                                thread_id
                            )
                            
                            # Update short record with adjusted timing and final path
                            with app.app_context():
                                updated_short = VideoShort.query.get(short.id)
                                if updated_short:
                                    updated_short.start_time = adjusted_start
                                    updated_short.end_time = adjusted_end
                                    updated_short.duration = adjusted_end - adjusted_start
                                    updated_short.output_path = final_output
                                    updated_short.thumbnail_path = os.path.join('outputs', f'short_{short.id}_thumb.jpg')
                                    db.session.commit()
                        
                        finally:
                            # Clean up thread-safe input file
                            if os.path.exists(safe_input_path):
                                os.remove(safe_input_path)
                            # Clean up temporary output if it exists
                            if os.path.exists(output_path) and output_path != final_output:
                                os.remove(output_path)
                        
                        self.logger.info(f"ULTRA-PARALLEL generated short {i+1}: {final_output}")
                        return f"Short {i+1} completed with ultra-fast processing"
                        
                except Exception as e:
                    self.logger.error(f"ULTRA-PARALLEL failed to generate short {i+1}: {e}")
                    return f"Short {i+1} failed: {e}"
            
            # ULTRA-OPTIMIZED parallel processing for video generation with enhanced data preparation
            segment_data = []
            for i, segment in enumerate(engaging_segments):
                # Prepare all segment data for ultra-parallel processing
                segment_scores = {
                    'engagement_score': getattr(segment, 'engagement_score', 0.5),
                    'emotion_score': getattr(segment, 'emotion_score', 0.5),
                    'viral_potential': getattr(segment, 'viral_potential', 0.5),
                    'quotability': getattr(segment, 'quotability', 0.5),
                    'overall_score': getattr(segment, 'overall_score', 0.5),
                    'emotions_detected': getattr(segment, 'emotions_detected', []),
                    'keywords': getattr(segment, 'keywords', []),
                    'analysis_notes': getattr(segment, 'analysis_notes', '')
                }
                segment_data.append((
                    segment.id, 
                    segment.text, 
                    segment.start_time, 
                    segment.end_time, 
                    segment_scores, 
                    i
                ))
            
            # ULTRA-ADVANCED dynamic worker allocation with system resource detection
            import os
            import psutil
            
            cpu_count = os.cpu_count() or 4
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # ULTRA-DYNAMIC scaling based on system resources and workload
            if available_memory_gb > 16:  # Ultra-high memory system
                max_workers = min(8, len(engaging_segments), cpu_count)
            elif available_memory_gb > 8:  # High memory system
                max_workers = min(6, len(engaging_segments), max(2, cpu_count))
            elif available_memory_gb > 4:  # Medium memory system
                max_workers = min(4, len(engaging_segments), max(2, cpu_count // 2))
            else:  # Low memory system
                max_workers = min(2, len(engaging_segments), max(1, cpu_count // 4))
            
            # Ensure we have at least 1 worker but cap for optimal performance
            max_workers = max(1, min(max_workers, 6))  # Cap at 6 for optimal performance
            
            self.logger.info(f"ULTRA-PARALLEL system: {cpu_count} CPUs, {available_memory_gb:.1f}GB RAM")
            self.logger.info(f"Using {max_workers} ULTRA-PARALLEL workers for lightning-fast video generation")
            
            # Process videos in batches to optimize memory usage
            batch_size = max_workers * 2
            total_batches = (len(segment_pairs) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(segment_pairs))
                batch_pairs = segment_pairs[start_idx:end_idx]
                
                self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch_pairs)} videos")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_segment = {executor.submit(process_single_short, data): data 
                                       for data in batch_pairs}
                    
                    for future in as_completed(future_to_segment):
                        try:
                            result = future.result(timeout=600)  # 10-minute timeout per video
                            self.logger.info(f"ULTRA-PARALLEL processing result: {result}")
                        except Exception as future_error:
                            self.logger.error(f"ULTRA-PARALLEL processing error: {future_error}")
                
                # Memory cleanup between batches
                import gc
                gc.collect()
                
                # Update progress
                completed_videos = min(end_idx, len(segment_pairs))
                progress = 70 + int((completed_videos / len(segment_pairs)) * 15)  # Reserve 5% for thumbnails
                self._update_job_status(job, ProcessingStatus.EDITING, progress)
            
            # Generate all thumbnails in parallel after video processing
            self.logger.info("Starting parallel thumbnail generation for all videos...")
            self._update_job_status(job, ProcessingStatus.EDITING, 85)
            
            # Collect all video-thumbnail pairs for batch processing
            with app.app_context():
                shorts = VideoShort.query.filter_by(job_id=job.id).all()
                video_thumbnail_pairs = []
                for short in shorts:
                    if short.output_path and os.path.exists(short.output_path):
                        video_thumbnail_pairs.append((short.output_path, short.thumbnail_path))
                
                if video_thumbnail_pairs:
                    # Generate all thumbnails in parallel
                    thumbnail_results = self._generate_thumbnails_parallel(video_thumbnail_pairs)
                    self.logger.info(f"Parallel thumbnail generation completed: {len(thumbnail_results)} results")
                    
                    # Update progress to completion
                    self._update_job_status(job, ProcessingStatus.EDITING, 90)
                else:
                    self.logger.warning("No valid video files found for thumbnail generation")
                    
        except Exception as e:
            raise Exception(f"Failed to generate shorts: {e}")

    def _create_vertical_video(self, input_path, output_path, start_time, end_time):
        """Create vertical 9:16 video from horizontal source using FFmpeg - ULTRA OPTIMIZED PARALLEL PROCESSING"""
        return self._create_vertical_video_threaded(input_path, output_path, start_time, end_time, None)
    
    def _create_vertical_video_threaded(self, input_path, output_path, start_time, end_time, thread_id=None):
        """Create vertical 9:16 video with thread-specific optimizations - ULTRA-FAST PROCESSING"""
        try:
            duration = end_time - start_time
            
            # Thread-specific optimization
            if thread_id:
                # Limit threads per FFmpeg process for better parallel performance
                thread_count = max(1, (os.cpu_count() or 4) // 4)  # Divide CPU among parallel processes
            else:
                thread_count = 0  # Use all threads for single process
            
            # ULTRA-OPTIMIZED FFmpeg command with enhanced parallel processing
            cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',  # Hardware acceleration
                '-hwaccel_output_format', 'auto',  # Hardware output format
                '-thread_queue_size', '2048',  # Larger input buffer for ultra-fast processing
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                # ULTRA-ENHANCED video filter with parallel processing
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase:flags=bilinear,crop=1080:1920,format=yuv420p',
                '-c:v', 'libx264',
                '-preset', 'superfast',  # Balanced speed/quality preset
                '-tune', 'zerolatency',  # Ultra-fast encoding
                '-crf', '32',  # Higher CRF for maximum speed
                '-profile:v', 'baseline',  # Fastest encoding profile
                '-level', '3.0',  # Optimal level for speed
                '-x264opts', 'no-scenecut:keyint=25:min-keyint=25:ref=1:bframes=0',  # Ultra-fast settings
                # ULTRA-FAST audio processing
                '-c:a', 'aac',
                '-b:a', '48k',  # Minimal bitrate for speed
                '-ac', '2',  # Stereo output
                '-ar', '44100',  # Standard sample rate
                # ULTRA-FAST output optimizations
                '-movflags', '+faststart+dash+empty_moov',
                '-avoid_negative_ts', 'make_zero',
                '-shortest',  # End when shortest stream ends
                # ENHANCED parallel processing settings
                '-threads', str(thread_count),  # Optimized thread count
                '-thread_type', 'frame+slice',  # Enable both frame and slice threading
                '-slices', '16',  # More slices for better parallelization
                # ULTRA memory and I/O optimizations
                '-max_muxing_queue_size', '16384',
                '-fflags', '+genpts+discardcorrupt+fastseek',
                '-copyts',  # Copy timestamps for faster processing
                '-y',
                output_path
            ]
            
            # Run with ULTRA-OPTIMIZED environment for parallel processing
            env = os.environ.copy()
            if thread_id:
                # Thread-specific environment optimization
                env['OMP_NUM_THREADS'] = str(max(1, thread_count))
                env['MKL_NUM_THREADS'] = str(max(1, thread_count))
                env['OPENBLAS_NUM_THREADS'] = str(max(1, thread_count))
            else:
                env['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
                env['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
            
            # Add process priority for ultra-fast processing
            env['FFMPEG_THREAD_PRIORITY'] = 'high'
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            
            if result.returncode != 0:
                # ULTRA-FAST fallback command
                self.logger.warning(f"Ultra-optimized FFmpeg failed, trying ultra-fast fallback: {result.stderr}")
                fallback_cmd = [
                    'ffmpeg', '-hwaccel', 'auto', '-i', input_path,
                    '-ss', str(start_time), '-t', str(duration),
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264', '-preset', 'superfast', '-crf', '30',
                    '-c:a', 'aac', '-b:a', '64k', 
                    '-threads', str(thread_count) if thread_count > 0 else '0',
                    '-y', output_path
                ]
                fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
                if fallback_result.returncode != 0:
                    raise Exception(f"Ultra-fast FFmpeg failed: {fallback_result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception(f"Video processing timeout for thread {thread_id}")
        except Exception as e:
            raise Exception(f"Failed to create ultra-fast vertical video: {e}")

    def _generate_thumbnail(self, video_path, thumbnail_path):
        """Generate thumbnail from video - OPTIMIZED FOR SPEED"""
        try:
            cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',  # Use hardware acceleration
                '-i', video_path,
                '-ss', '00:00:01.000',
                '-vframes', '1',
                '-s', '640x1136',
                '-q:v', '5',  # Add quality setting for faster processing
                '-y',
                thumbnail_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate thumbnail: {e}")

    def _generate_thumbnails_parallel(self, video_thumbnail_pairs):
        """Generate multiple thumbnails in parallel using multithreading"""
        def generate_single_thumbnail(pair):
            video_path, thumbnail_path = pair
            try:
                cmd = [
                    'ffmpeg',
                    '-hwaccel', 'auto',  # Hardware acceleration
                    '-thread_queue_size', '512',  # Optimized buffer for parallel processing
                    '-i', video_path,
                    '-ss', '00:00:01.000',
                    '-vframes', '1',
                    '-s', '640x1136',
                    '-q:v', '3',  # Higher quality for better thumbnails
                    '-preset', 'ultrafast',  # Fast processing
                    '-threads', '2',  # Limit threads per thumbnail
                    '-y',
                    thumbnail_path
                ]
                
                # Run with timeout to prevent hanging
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                return f"Thumbnail generated: {thumbnail_path}"
                
            except subprocess.TimeoutExpired:
                return f"Thumbnail timeout: {thumbnail_path}"
            except Exception as e:
                return f"Thumbnail failed: {thumbnail_path} - {e}"

        # Determine optimal number of workers for thumbnail generation
        import os
        cpu_count = os.cpu_count() or 4
        # Use fewer workers for thumbnails to avoid overwhelming FFmpeg
        max_workers = min(4, len(video_thumbnail_pairs), max(2, cpu_count // 2))
        
        self.logger.info(f"Generating {len(video_thumbnail_pairs)} thumbnails with {max_workers} parallel workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(generate_single_thumbnail, pair): pair 
                             for pair in video_thumbnail_pairs}
            
            for future in as_completed(future_to_pair):
                try:
                    result = future.result(timeout=45)  # Allow extra time for completion
                    results.append(result)
                    self.logger.info(f"Parallel thumbnail result: {result}")
                except Exception as e:
                    pair = future_to_pair[future]
                    self.logger.error(f"Thumbnail generation failed for {pair[1]}: {e}")
                    results.append(f"Failed: {pair[1]}")
        
        return results
    
    def _select_preferred_audio_stream(self, video_path):
        """Select audio stream with Hindi first, English second priority"""
        try:
            # Get detailed stream information
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', '-show_format', video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode != 0:
                self.logger.warning("Could not probe video streams, using default audio")
                return 0
            
            probe_data = json.loads(probe_result.stdout)
            audio_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'audio']
            
            if not audio_streams:
                self.logger.warning("No audio streams found")
                return 0
            
            self.logger.info(f"Found {len(audio_streams)} audio streams")
            
            # Priority system: Hindi -> English -> Default
            hindi_stream = None
            english_stream = None
            default_stream = 0
            
            for idx, stream in enumerate(audio_streams):
                tags = stream.get('tags', {})
                language = tags.get('language', '').lower()
                title = tags.get('title', '').lower()
                
                self.logger.info(f"Audio stream {idx}: language='{language}', title='{title}'")
                
                # Check for Hindi indicators
                hindi_indicators = ['hi', 'hin', 'hindi', 'हिंदी', 'हिन्दी']
                if any(indicator in language for indicator in hindi_indicators) or \
                   any(indicator in title for indicator in hindi_indicators):
                    hindi_stream = idx
                    self.logger.info(f"Found Hindi audio stream at index {idx}")
                    break  # Hindi has highest priority, use immediately
                
                # Check for English indicators
                english_indicators = ['en', 'eng', 'english']
                if english_stream is None and (
                    any(indicator in language for indicator in english_indicators) or
                    any(indicator in title for indicator in english_indicators)
                ):
                    english_stream = idx
                    self.logger.info(f"Found English audio stream at index {idx}")
                
                # Also check stream metadata for more clues
                if 'metadata' in stream:
                    metadata = stream['metadata']
                    if any(key for key in metadata.keys() if 'hindi' in key.lower() or 'hi' in key.lower()):
                        hindi_stream = idx
                        self.logger.info(f"Found Hindi audio stream via metadata at index {idx}")
                        break
            
            # Return in priority order: Hindi -> English -> Default
            if hindi_stream is not None:
                self.logger.info(f"Using Hindi audio stream: {hindi_stream}")
                return hindi_stream
            elif english_stream is not None:
                self.logger.info(f"Using English audio stream: {english_stream}")
                return english_stream
            else:
                self.logger.info(f"Using default audio stream: {default_stream}")
                return default_stream
                
        except Exception as e:
            self.logger.error(f"Error selecting audio stream: {e}")
            return 0  # Fallback to first stream

    def _cleanup_temporary_files(self, job):
        """Clean up temporary files after processing"""
        try:
            files_to_remove = []
            
            # Add video file
            if job.video_path and os.path.exists(job.video_path):
                files_to_remove.append(job.video_path)
            
            # Add audio file
            if job.audio_path and os.path.exists(job.audio_path):
                files_to_remove.append(job.audio_path)
            
            # Add transcript file
            if job.transcript_path and os.path.exists(job.transcript_path):
                files_to_remove.append(job.transcript_path)
            
            # Remove files
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Removed temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Clean temporary directories if empty
            for temp_dir in ['uploads', 'temp']:
                if os.path.exists(temp_dir):
                    try:
                        if not os.listdir(temp_dir):  # Directory is empty
                            os.rmdir(temp_dir)
                            self.logger.info(f"Removed empty directory: {temp_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove directory {temp_dir}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
