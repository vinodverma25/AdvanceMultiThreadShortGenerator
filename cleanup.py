#!/usr/bin/env python3
import os
import shutil
import logging
from datetime import datetime, timedelta
from app import app, db
from models import VideoJob, VideoShort


def cleanup_old_files(days_old=7):
    """Clean up files older than specified days"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)

    # Directories to clean
    directories = ['uploads', 'outputs', 'temp']

    total_freed = 0

    for directory in directories:
        if os.path.exists(directory):
            print(f"Cleaning {directory}...")
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                try:
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(
                            os.path.getmtime(filepath))
                        if file_time < cutoff_date:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            total_freed += file_size
                            print(f"Deleted: {filepath} ({file_size} bytes)")
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

    return total_freed


def cleanup_orphaned_files():
    """Clean up files not referenced in database"""
    with app.app_context():
        total_freed = 0

        # Get all file paths from database
        db_files = set()
        jobs = VideoJob.query.all()
        for job in jobs:
            if job.video_path:
                db_files.add(job.video_path)
            if job.audio_path:
                db_files.add(job.audio_path)
            if job.transcript_path:
                db_files.add(job.transcript_path)

        shorts = VideoShort.query.all()
        for short in shorts:
            if short.output_path:
                db_files.add(short.output_path)
            if short.thumbnail_path:
                db_files.add(short.thumbnail_path)

        # Check uploads and outputs directories
        for directory in ['uploads', 'outputs']:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath) and filepath not in db_files:
                        try:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            total_freed += file_size
                            print(
                                f"Deleted orphaned file: {filepath} ({file_size} bytes)"
                            )
                        except Exception as e:
                            print(f"Error deleting {filepath}: {e}")

        return total_freed


def cleanup_temp_directory():
    """Clean up temporary directory completely"""
    temp_dir = 'temp'
    total_freed = 0

    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    total_freed += file_size
                    print(f"Deleted temp file: {filepath} ({file_size} bytes)")
                elif os.path.isdir(filepath):
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(filepath)
                        for filename in filenames)
                    shutil.rmtree(filepath)
                    total_freed += dir_size
                    print(
                        f"Deleted temp directory: {filepath} ({dir_size} bytes)"
                    )
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")

    return total_freed


def get_directory_size(directory):
    """Get total size of directory in bytes"""
    total_size = 0
    if os.path.exists(directory):
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
    return total_size


def format_bytes(bytes_size):
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def main():
    print("=== Disk Space Cleanup ===")

    # Show current usage
    print("\nCurrent disk usage:")
    for directory in ['uploads', 'outputs', 'temp']:
        size = get_directory_size(directory)
        print(f"{directory}: {format_bytes(size)}")

    # Clean temp directory first (safest)
    print("\n1. Cleaning temporary files...")
    temp_freed = cleanup_temp_directory()
    print(f"Freed from temp: {format_bytes(temp_freed)}")

    # Clean orphaned files
    print("\n2. Cleaning orphaned files...")
    orphaned_freed = cleanup_orphaned_files()
    print(f"Freed from orphaned files: {format_bytes(orphaned_freed)}")

    # Clean old files (7+ days)
    print("\n3. Cleaning files older than 7 days...")
    old_freed = cleanup_old_files(7)
    print(f"Freed from old files: {format_bytes(old_freed)}")

    total_freed = temp_freed + orphaned_freed + old_freed
    print(f"\nTotal space freed: {format_bytes(total_freed)}")

    # Show usage after cleanup
    print("\nDisk usage after cleanup:")
    for directory in ['uploads', 'outputs', 'temp']:
        size = get_directory_size(directory)
        print(f"{directory}: {format_bytes(size)}")


if __name__ == "__main__":
    main()
