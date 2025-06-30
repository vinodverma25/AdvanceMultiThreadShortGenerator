# YouTube Data API Setup Guide

## Quick Setup Instructions

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable billing (required for API access)

### 2. Enable YouTube Data API v3
1. Go to [APIs & Services > Library](https://console.cloud.google.com/apis/library)
2. Search for "YouTube Data API v3"
3. Click "Enable"

### 3. Create OAuth 2.0 Credentials
1. Go to [APIs & Services > Credentials](https://console.cloud.google.com/apis/credentials)
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Select "Web application"
4. Add these redirect URIs:
   ```
   https://c27a0196-fd54-4876-a24c-3fd7e625b350-00-8eg7yhoxmb56.picard.replit.dev/youtube/callback
   http://localhost:5000/youtube/callback
   ```

### 4. Configure Environment Variables
Set these secrets in your Replit environment:

- **YOUTUBE_CLIENT_ID**: Your OAuth 2.0 Client ID
- **YOUTUBE_CLIENT_SECRET**: Your OAuth 2.0 Client Secret

### 5. OAuth Consent Screen
1. Go to [APIs & Services > OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent)
2. Select "External" user type
3. Fill required fields:
   - App name: "YouTube Shorts Generator"
   - User support email: Your email
   - Developer contact: Your email
4. Add scopes:
   - `https://www.googleapis.com/auth/youtube.upload`
   - `https://www.googleapis.com/auth/youtube.readonly`

### 6. Test Configuration
1. Start the application
2. Go to the shorts generator interface
3. Click "Connect YouTube Account"
4. Complete OAuth flow
5. Upload a test short

## Current Configuration Status

✅ **Application Setup**: Flask app configured for YouTube API
✅ **OAuth Handler**: Implemented with proper redirect URIs
✅ **Upload System**: Ready for video uploads with metadata
⚠️ **API Credentials**: Need to be configured in environment

## Required Scopes

- `https://www.googleapis.com/auth/youtube.upload` - Upload videos
- `https://www.googleapis.com/auth/youtube.readonly` - Read channel info

## Quota Information

- **Free Tier**: 10,000 units per day
- **Upload Cost**: 1,600 units per video
- **Daily Uploads**: ~6 videos on free tier

## Security Notes

- Keep Client Secret secure
- Use HTTPS redirect URIs only
- Regularly rotate credentials
- Monitor API usage in Google Cloud Console

## Troubleshooting

**Common Issues:**
1. **Invalid redirect URI**: Ensure exact match in Google Console
2. **Quota exceeded**: Check daily usage limits
3. **Scope errors**: Verify OAuth consent screen scopes
4. **Token expired**: App handles automatic refresh

**Debug Steps:**
1. Check application logs for OAuth errors
2. Verify redirect URI matches exactly
3. Confirm API is enabled in Google Console
4. Test with fresh OAuth consent