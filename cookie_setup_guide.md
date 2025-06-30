# YouTube Cookies Setup Guide

This guide helps you set up YouTube authentication cookies to download age-restricted content.

## Why Cookies Are Needed

YouTube restricts access to certain content based on:
- Age restrictions (18+ content)
- Regional restrictions
- Private/unlisted videos
- Videos requiring YouTube Premium

## Step-by-Step Setup

### Method 1: Using Browser Extension (Recommended)

1. **Install Cookie Extension**
   - Chrome: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Login to YouTube**
   - Open YouTube in your browser
   - Sign in to your Google/YouTube account
   - Verify you can access age-restricted content

3. **Export Cookies**
   - Navigate to `youtube.com`
   - Click the cookie extension icon
   - Select "Export" or "Download cookies.txt"
   - Save the file as `youtube_cookies.txt`

4. **Upload to Project**
   - Replace the existing `youtube_cookies.txt` file with your exported cookies
   - The file should contain lines like:
   ```
   .youtube.com	TRUE	/	FALSE	1672531200	session_token	your_actual_token
   ```

### Method 2: Manual Cookie Extraction

1. **Open Developer Tools**
   - Go to YouTube and login
   - Press F12 to open Developer Tools
   - Go to "Application" tab > "Cookies" > "https://youtube.com"

2. **Extract Required Cookies**
   - Look for these important cookies:
     - `session_token`
     - `VISITOR_INFO1_LIVE`
     - `LOGIN_INFO`
     - `SAPISID`
     - `HSID`

3. **Format Cookies**
   - Create entries in Netscape format:
   ```
   .youtube.com	TRUE	/	FALSE	[expiry_timestamp]	[cookie_name]	[cookie_value]
   ```

## Security Notes

⚠️ **Important Security Information:**
- Never share your cookies file publicly
- Cookies contain your login session data
- Keep the file secure and private
- Regenerate cookies periodically for security

## Verification

The system will automatically:
- Check if cookies file exists
- Validate cookies contain actual data
- Log authentication status during downloads
- Show warnings if no valid cookies found

## Troubleshooting

**Cookies not working?**
- Ensure you're logged into YouTube when exporting
- Check cookie expiration dates
- Verify file format is correct
- Try re-exporting fresh cookies

**Still getting age-restriction errors?**
- Confirm your YouTube account can access the content
- Try using a different YouTube account
- Check if the video requires YouTube Premium

**Format errors?**
- Ensure no extra spaces or characters
- Check that tabs separate fields (not spaces)
- Verify timestamp format is correct

## Success Indicators

You'll know cookies are working when:
- Download logs show "Using YouTube authentication cookies"
- Age-restricted videos download successfully
- No authentication errors in processing logs

## Support

If you continue having issues:
1. Check the application logs for specific error messages
2. Verify your YouTube account has access to the restricted content
3. Try exporting cookies from a fresh browser session