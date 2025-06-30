import json
import logging
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Dict, Any

class SegmentAnalysis(BaseModel):
    engagement_score: float
    emotion_score: float
    viral_potential: float
    quotability: float
    emotions: List[str]
    keywords: List[str]
    reason: str

class VideoMetadata(BaseModel):
    title: str
    description: str
    tags: List[str]

class GeminiAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.use_fallback_only = False
        self.api_keys = []
        self.current_key_index = 0
        
        # Collect all available API keys
        self._collect_api_keys()
        
        # Initialize Gemini client with first available key
        if self.api_keys:
            self._initialize_client()
        else:
            self.logger.warning("No Gemini API keys found in environment variables")
            self.logger.info("Will use fallback analysis methods only")
            self.use_fallback_only = True

    def _collect_api_keys(self):
        """Collect all available Gemini API keys from environment"""
        # Primary key
        primary_key = os.environ.get("GEMINI_API_KEY")
        if primary_key:
            self.api_keys.append(primary_key)
        
        # Backup keys
        for i in range(1, 5):  # Support up to 4 backup keys
            backup_key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if backup_key:
                self.api_keys.append(backup_key)
        
        self.logger.info(f"Found {len(self.api_keys)} Gemini API key(s)")

    def _initialize_client(self):
        """Initialize client with current API key"""
        if self.current_key_index < len(self.api_keys):
            try:
                api_key = self.api_keys[self.current_key_index]
                self.client = genai.Client(api_key=api_key)
                self.logger.info(f"Gemini client initialized with API key #{self.current_key_index + 1}")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini client with key #{self.current_key_index + 1}: {e}")
                return False
        return False

    def _switch_to_next_key(self):
        """Switch to next available API key"""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            self.logger.info(f"Switching to backup API key #{self.current_key_index + 1}")
            if self._initialize_client():
                return True
        
        # No more keys available
        self.logger.warning("All Gemini API keys exhausted, switching to fallback mode")
        self.use_fallback_only = True
        self.client = None
        return False

    def _handle_api_error(self, error_msg: str):
        """Handle API errors and attempt key switching"""
        # Check for quota exceeded or rate limit errors
        if any(indicator in error_msg.lower() for indicator in ["429", "resource_exhausted", "quota", "rate limit"]):
            self.logger.warning(f"API quota/rate limit hit: {error_msg}")
            return self._switch_to_next_key()
        
        # For other errors, log but don't switch keys
        self.logger.error(f"API error: {error_msg}")
        return False

    def analyze_segment(self, text: str) -> Dict[str, Any]:
        """Analyze a text segment for engagement and viral potential using Gemini - OPTIMIZED"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info("Using fallback analysis (no Gemini API available)")
            return self._fallback_analysis(text)
        
        # Optimize text length for faster processing
        if len(text) > 2000:  # Limit text length for faster AI processing
            text = text[:2000] + "..."
        
        try:
            system_prompt = """You are an expert content analyst specializing in viral social media content and YouTube Shorts.
            
            Analyze the given text segment for its potential to create engaging short-form video content.
            
            Consider these factors:
            - Engagement Score (0.0-1.0): How likely this content is to engage viewers
            - Emotion Score (0.0-1.0): Emotional impact and intensity
            - Viral Potential (0.0-1.0): Likelihood to be shared and go viral
            - Quotability (0.0-1.0): How memorable and quotable the content is
            - Emotions: List of emotions detected (humor, surprise, excitement, inspiration, etc.)
            - Keywords: Important keywords that make this content engaging
            - Reason: Brief explanation of why this segment is engaging
            
            Focus on content that has:
            - Strong emotional hooks
            - Surprising or unexpected elements
            - Humor or entertainment value
            - Inspirational or motivational content
            - Controversial or debate-worthy topics
            - Clear storytelling elements
            - Quotable phrases or moments"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=f"Analyze this content segment for YouTube Shorts potential:\n\n{text}")])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=SegmentAnalysis,
                ),
            )

            if response.text:
                result = json.loads(response.text)
                return {
                    'engagement_score': max(0.0, min(1.0, result.get('engagement_score', 0.5))),
                    'emotion_score': max(0.0, min(1.0, result.get('emotion_score', 0.5))),
                    'viral_potential': max(0.0, min(1.0, result.get('viral_potential', 0.5))),
                    'quotability': max(0.0, min(1.0, result.get('quotability', 0.5))),
                    'emotions': result.get('emotions', [])[:5],  # Limit to 5 emotions
                    'keywords': result.get('keywords', [])[:10],  # Limit to 10 keywords
                    'reason': result.get('reason', 'Content has potential for engagement')[:500]
                }
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            error_msg = str(e)
            
            # Try to switch to next API key if error is quota-related
            if self._handle_api_error(error_msg) and not self.use_fallback_only:
                # Retry with new key
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=f"Analyze this content segment for YouTube Shorts potential:\n\n{text}")])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=SegmentAnalysis,
                        ),
                    )

                    if response.text:
                        result = json.loads(response.text)
                        return {
                            'engagement_score': max(0.0, min(1.0, result.get('engagement_score', 0.5))),
                            'emotion_score': max(0.0, min(1.0, result.get('emotion_score', 0.5))),
                            'viral_potential': max(0.0, min(1.0, result.get('viral_potential', 0.5))),
                            'quotability': max(0.0, min(1.0, result.get('quotability', 0.5))),
                            'emotions': result.get('emotions', [])[:5],
                            'keywords': result.get('keywords', [])[:10],
                            'reason': result.get('reason', 'Content has potential for engagement')[:500]
                        }
                except Exception as retry_e:
                    self.logger.error(f"Retry with backup key failed: {retry_e}")
            
            # Fallback analysis
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced fallback analysis when Gemini is unavailable"""
        text_lower = text.lower()
        words = text.split()
        
        # Enhanced keyword categories
        engagement_keywords = ['amazing', 'incredible', 'wow', 'shocking', 'unbelievable', 'funny', 'hilarious', 
                              'awesome', 'fantastic', 'mind-blowing', 'crazy', 'insane', 'epic', 'legendary']
        emotion_keywords = ['love', 'hate', 'excited', 'surprised', 'happy', 'angry', 'scared', 'thrilled',
                           'disappointed', 'frustrated', 'overwhelmed', 'passionate', 'emotional', 'heartwarming']
        viral_keywords = ['viral', 'trending', 'share', 'like', 'subscribe', 'follow', 'must-see', 'breaking',
                         'exclusive', 'revealed', 'secret', 'exposed', 'truth', 'shocking']
        quotable_keywords = ['said', 'quote', 'tells', 'explains', 'reveals', 'admits', 'confesses', 'announces']
        
        # Calculate scores based on keyword presence
        engagement_score = min(1.0, sum(1 for word in engagement_keywords if word in text_lower) * 0.15)
        emotion_score = min(1.0, sum(1 for word in emotion_keywords if word in text_lower) * 0.15)
        viral_score = min(1.0, sum(1 for word in viral_keywords if word in text_lower) * 0.2)
        quotability_score = min(1.0, sum(1 for word in quotable_keywords if word in text_lower) * 0.2)
        
        # Length-based scoring (optimal length for shorts)
        text_length = len(words)
        if 20 <= text_length <= 50:  # Optimal length for short clips
            length_bonus = 0.2
        elif 10 <= text_length <= 80:  # Good length
            length_bonus = 0.1
        else:
            length_bonus = 0.0
        
        # Add length bonus to all scores
        engagement_score = min(1.0, engagement_score + length_bonus)
        emotion_score = min(1.0, emotion_score + length_bonus)
        viral_score = min(1.0, viral_score + length_bonus)
        quotability_score = min(1.0, quotability_score + length_bonus)
        
        # Ensure minimum scores for content viability
        engagement_score = max(0.4, engagement_score)
        emotion_score = max(0.3, emotion_score)
        viral_score = max(0.3, viral_score)
        quotability_score = max(0.2, quotability_score)
        
        # Detect emotions based on keywords
        detected_emotions = []
        if any(word in text_lower for word in ['funny', 'hilarious', 'joke', 'laugh']):
            detected_emotions.append('humor')
        if any(word in text_lower for word in ['shocking', 'surprised', 'unexpected']):
            detected_emotions.append('surprise')
        if any(word in text_lower for word in ['love', 'heartwarming', 'beautiful']):
            detected_emotions.append('inspiration')
        if any(word in text_lower for word in ['angry', 'frustrated', 'hate']):
            detected_emotions.append('controversy')
        if not detected_emotions:
            detected_emotions = ['general']
        
        # Extract meaningful keywords (longer words, excluding common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an'}
        keywords = [word for word in words if len(word) > 3 and word.lower() not in common_words][:8]
        
        return {
            'engagement_score': engagement_score,
            'emotion_score': emotion_score,
            'viral_potential': viral_score,
            'quotability': quotability_score,
            'emotions': detected_emotions[:5],
            'keywords': keywords,
            'reason': f'Fallback analysis: {len(words)} words, detected {", ".join(detected_emotions)} content'
        }
    
    def _enhance_description_to_500_words(self, description: str, segment_text: str, original_title: str) -> str:
        """Enhance description to approximately 500 words for viral potential"""
        if len(description.split()) >= 400:
            return description
        
        # Extract key elements from the segment
        words = segment_text.split()
        key_points = []
        
        # Add context about the original video
        enhanced_desc = f"🔥 VIRAL MOMENT ALERT! 🔥\n\n"
        enhanced_desc += f"This INCREDIBLE clip from '{original_title}' is absolutely MIND-BLOWING! "
        enhanced_desc += f"What you're about to see will leave you SPEECHLESS and wondering how this even happened! "
        
        # Add the original description
        enhanced_desc += f"\n\n{description}\n\n"
        
        # Add engaging context
        enhanced_desc += f"But wait, there's MORE! This isn't just any ordinary moment - this is the kind of content that "
        enhanced_desc += f"BREAKS the internet and gets shared millions of times! The way everything unfolds is absolutely "
        enhanced_desc += f"INSANE and you need to see it to believe it!\n\n"
        
        # Add emotional hooks
        enhanced_desc += f"💥 Why is this going VIRAL?\n"
        enhanced_desc += f"✨ The timing is PERFECT\n"
        enhanced_desc += f"🤯 The reaction is PRICELESS\n"
        enhanced_desc += f"🔥 This moment is LEGENDARY\n\n"
        
        # Add engagement prompts
        enhanced_desc += f"👆 SMASH that LIKE button if this gave you CHILLS!\n"
        enhanced_desc += f"💬 COMMENT below what you think about this CRAZY moment!\n"
        enhanced_desc += f"🔔 SUBSCRIBE for more VIRAL content like this!\n"
        enhanced_desc += f"📱 SHARE this with everyone - they NEED to see this!\n\n"
        
        # Add trending hashtags
        enhanced_desc += f"#Shorts #Viral #Trending #Fyp #Amazing #Shocking #MustWatch #Unbelievable #MindBlown #Epic "
        enhanced_desc += f"#Insane #Crazy #OMG #NoWay #Incredible #Legendary #Speechless #Chills #Goosebumps #WOW\n\n"
        
        # Add questions for engagement
        enhanced_desc += f"🤔 What did you think when you saw this?\n"
        enhanced_desc += f"😱 Have you ever experienced something like this?\n"
        enhanced_desc += f"🔥 What would YOUR reaction be?\n\n"
        
        # Add urgency and FOMO
        enhanced_desc += f"⚡ This is the moment EVERYONE is talking about!\n"
        enhanced_desc += f"🚨 Don't miss out on the HOTTEST viral content!\n"
        enhanced_desc += f"⏰ Watch before it gets even MORE popular!\n\n"
        
        # Final call to action
        enhanced_desc += f"DROP a 🔥 in the comments if this was AMAZING! Let's get this to 1 MILLION views!"
        
        return enhanced_desc

    def generate_metadata(self, segment_text: str, original_title: str, language: str = "hinglish") -> Dict[str, Any]:
        """Generate title, description, and tags for a video short using Gemini"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info("Using fallback metadata generation (no Gemini API available)")
            return self._fallback_metadata(segment_text, original_title, language)
            
        try:
            if language.lower() == "hindi":
                system_prompt = """आप एक वायरल YouTube Shorts विशेषज्ञ हैं जो करोड़ों व्यूज पाने वाला कंटेंट बनाते हैं।

वायरल YouTube Short के लिए मेटाडेटा बनाएं जो अधिकतम एंगेजमेंट और व्यूज लाए।

वायरल टाइटल नियम (अधिकतम 60 अक्षर):
- शक्तिशाली शब्द इस्तेमाल करें: 😱 चौंकाने वाला, 🔥 अविश्वसनीय, 💥 खुलासा, ⚡ रहस्य
- जिज्ञासा पैदा करें: "यह देखकर आप हैरान रह जाएंगे"
- इमोजी और नंबर शामिल करें
- स्क्रॉल न करने पर मजबूर करें

वायरल विवरण नियम (500 शब्द):
- पहली लाइन में चौंकाने वाला कथन या प्रश्न
- भावनात्मक कहानी बताएं
- 15-20 रणनीतिक हैशटैग शामिल करें
- CAPS का उपयोग करें जोर देने के लिए
- कई कॉल-टू-एक्शन जोड़ें (LIKE, SUBSCRIBE, COMMENT)
- तुरंत देखने की जरूरत पैदा करें

वायरल टैग्स नियम (12-15 टैग्स):
- हमेशा शामिल करें: #Shorts, #Viral, #Trending, #Fyp
- भावनात्मक टैग्स: #Shocking, #Amazing, #हिंदी, #भारत
- लोकप्रिय और लक्षित हैशटैग मिलाएं"""
            else:  # Hinglish
                system_prompt = """Aap ek viral YouTube Shorts expert hain jo crores views paane wala content banate hain.

Generate VIRAL metadata for YouTube Short jo maximum engagement aur views laye - Hindi aur English mix karte hue.

VIRAL TITLE RULES (max 60 characters):
- Power words use karein: 😱 SHOCKING, 🔥 INSANE, 💥 EXPOSED, ⚡ SECRET
- Curiosity gap banayein: "Ye Dekh Kar Aap Pagal Ho Jaenge!"
- Emojis aur numbers include karein
- Hinglish mein viral hooks: "Bhai Ye Kya Dekh Liya!", "OMG Ye Kaise Possible Hai!"

VIRAL DESCRIPTION RULES (500 words):
- First line mein shocking statement ya question (Hinglish mein)
- Emotional story batayein mixing Hindi-English
- 15-20 strategic hashtags naturally include karein
- ALL CAPS use karein emphasis ke liye
- Multiple calls-to-action: LIKE kar do, SUBSCRIBE karna, COMMENT mein batao
- FOMO create karein: "Abhi dekho warna miss ho jaega!"
- Engaging questions puchein throughout
- Emojis use karein har paragraph mein

VIRAL TAGS RULES (12-15 tags):
- Always include: #Shorts, #Viral, #Trending, #Fyp, #Hinglish, #India
- Emotional tags: #Shocking, #Amazing, #Unbelievable, #MindBlown, #Desi
- Hindi tags: #हिंदी, #भारत, #देसी
- Mix popular aur targeted hashtags"""

            if language.lower() == "hindi":
                prompt = f"""वायरल YouTube Shorts मेटाडेटा बनाएं अधिकतम व्यूज और एंगेजमेंट के लिए:

मूल वीडियो: {original_title}
कंटेंट सेगमेंट: {segment_text}

ऐसा मेटाडेटा बनाएं जो वायरल हो जाए करोड़ों व्यूज के साथ। हुक्स, जिज्ञासा और एंगेजमेंट पर फोकस करें। इमोजी का भरपूर इस्तेमाल करें।"""
            else:  # Hinglish
                prompt = f"""Create VIRAL YouTube Shorts metadata maximum views aur engagement ke liye - Hinglish style mein:

ORIGINAL VIDEO: {original_title}
CONTENT SEGMENT: {segment_text}

Aisa metadata banao jo VIRAL ho jaye millions of views ke saath. Hooks, curiosity aur engagement pe focus karo. Emojis ka bharpur use karo. Hindi-English mix kar ke catchy banao!"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=VideoMetadata,
                ),
            )

            if response.text:
                result = json.loads(response.text)
                
                # Ensure description is around 500 words
                description = result.get('description', '')
                if len(description.split()) < 400:
                    description = self._enhance_description_to_500_words(description, segment_text, original_title)
                
                return {
                    'title': result.get('title', f"SHOCKING Moment from {original_title} GOES VIRAL!")[:60],
                    'description': description,
                    'tags': result.get('tags', ['#Shorts', '#Viral', '#Trending', '#Fyp', '#Amazing', '#Shocking', '#MustWatch', '#Unbelievable'])[:15]
                }
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            error_msg = str(e)
            
            # Try to switch to next API key if error is quota-related
            if self._handle_api_error(error_msg) and not self.use_fallback_only:
                # Retry with new key
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=prompt)])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=VideoMetadata,
                        ),
                    )

                    if response.text:
                        result = json.loads(response.text)
                        return {
                            'title': result.get('title', f"Viral Moment from {original_title}")[:100],
                            'description': result.get('description', f"Amazing clip from {original_title}\n\n#Shorts #Viral #Trending"),
                            'tags': result.get('tags', ['shorts', 'viral', 'trending', 'entertainment'])[:15]
                        }
                except Exception as retry_e:
                    self.logger.error(f"Retry with backup key failed: {retry_e}")
            
            return self._fallback_metadata(segment_text, original_title)

    def _fallback_metadata(self, segment_text: str, original_title: str, language: str = "hinglish") -> Dict[str, Any]:
        """Enhanced fallback metadata generation"""
        words = segment_text.split()
        text_lower = segment_text.lower()
        
        # Extract meaningful keywords (exclude common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an', 'this', 'that'}
        key_words = [word for word in words if len(word) > 3 and word.lower() not in common_words][:5]
        
        # Generate viral title based on content type and language
        if language.lower() == "hindi":
            if any(word in text_lower for word in ['funny', 'hilarious', 'joke']):
                title_prefix = "😂 मजेदार"
            elif any(word in text_lower for word in ['shocking', 'unbelievable', 'incredible']):
                title_prefix = "😱 चौंकाने वाला"
            elif any(word in text_lower for word in ['amazing', 'awesome', 'fantastic']):
                title_prefix = "🔥 अविश्वसनीय"
            elif any(word in text_lower for word in ['secret', 'revealed', 'truth']):
                title_prefix = "💥 खुलासा"
            else:
                title_prefix = "⚡ वायरल"
                
            if key_words:
                title = f"{title_prefix}: {' '.join(key_words[:2])} वायरल!"[:60]
            else:
                title = f"{title_prefix} पल इंटरनेट तोड़ देता है!"[:60]
        else:  # Hinglish
            if any(word in text_lower for word in ['funny', 'hilarious', 'joke']):
                title_prefix = "😂 HILARIOUS Bhai"
            elif any(word in text_lower for word in ['shocking', 'unbelievable', 'incredible']):
                title_prefix = "😱 SHOCKING Yaar"
            elif any(word in text_lower for word in ['amazing', 'awesome', 'fantastic']):
                title_prefix = "🔥 INSANE Dost"
            elif any(word in text_lower for word in ['secret', 'revealed', 'truth']):
                title_prefix = "💥 EXPOSED Sach"
            else:
                title_prefix = "⚡ VIRAL Dekho"
                
            if key_words:
                title = f"{title_prefix}: {' '.join(key_words[:2])} VIRAL!"[:60]
            else:
                title = f"{title_prefix} Internet Tod Diya!"[:60]
        
        # Generate viral 500-word description
        description = self._enhance_description_to_500_words("", segment_text, original_title)
        
        # Generate viral tags with hashtags based on language
        if language.lower() == "hindi":
            viral_tags = ['#Shorts', '#Viral', '#Trending', '#Fyp', '#हिंदी', '#भारत', '#देसी', '#चौंकाने_वाला']
            
            if any(word in text_lower for word in ['funny', 'comedy', 'hilarious']):
                viral_tags.extend(['#मजेदार', '#कॉमेडी', '#हंसी'])
            if any(word in text_lower for word in ['music', 'song', 'dance']):
                viral_tags.extend(['#संगीत', '#गाना', '#नृत्य'])
            if any(word in text_lower for word in ['food', 'cooking']):
                viral_tags.extend(['#खाना', '#पकाना', '#रेसिपी'])
                
            viral_tags.extend(['#अविश्वसनीय', '#महान', '#पागल', '#ओएमजी'])
        else:  # Hinglish
            viral_tags = ['#Shorts', '#Viral', '#Trending', '#Fyp', '#Hinglish', '#India', '#Desi', '#MustWatch']
            
            if any(word in text_lower for word in ['funny', 'comedy', 'hilarious']):
                viral_tags.extend(['#Funny', '#Comedy', '#LOL', '#Mazedaar'])
            if any(word in text_lower for word in ['music', 'song', 'dance']):
                viral_tags.extend(['#Music', '#Song', '#Dance', '#Bollywood'])
            if any(word in text_lower for word in ['food', 'cooking']):
                viral_tags.extend(['#Food', '#Indian', '#Recipe', '#Khana'])
                
            viral_tags.extend(['#MindBlown', '#Epic', '#Insane', '#OMG', '#Bhai', '#यार'])
        
        return {
            'title': title,
            'description': description,
            'tags': viral_tags[:15]  # Limit to 15 tags
        }

    def analyze_video_file(self, video_path: str) -> Dict[str, Any]:
        """Analyze video file directly with Gemini vision capabilities"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info("Video file analysis not available (no Gemini API)")
            return {'analysis': 'Video analysis not available - using audio transcript analysis instead'}
        
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(
                        data=video_bytes,
                        mime_type="video/mp4",
                    ),
                    "Analyze this video for engaging moments, emotional highlights, and viral potential. "
                    "Identify the most interesting segments that would work well as YouTube Shorts."
                ],
            )

            return {'analysis': response.text if response.text else 'No analysis available'}

        except Exception as e:
            self.logger.error(f"Video file analysis failed: {e}")
            return {'analysis': 'Video analysis not available'}
