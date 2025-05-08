#!/usr/bin/env python3

# ------------------------ Imports ------------------------ #
import os
import subprocess
import sys
import time
from typing import Optional, Tuple, List

try:
    import speech_recognition as sr
except ModuleNotFoundError:
    print("The 'speech_recognition' module is not installed. Please install it using 'pip install SpeechRecognition'.")
    exit(1)

try:
    from gtts import gTTS
except ModuleNotFoundError:
    print("The 'gtts' module is not installed. Please install it using 'pip install gtts'.")
    exit(1)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    print("The 'python-dotenv' module is not installed. Please install it using 'pip install python-dotenv'.")
    exit(1)

try:
    import pygame
except ModuleNotFoundError:
    print("The 'pygame' module is not installed. Please install it using 'pip install pygame'.")
    exit(1)

try:
    from groq import Groq
except ModuleNotFoundError:
    print("The 'groq' module is not installed. Please install it using 'pip install groq'.")
    exit(1)

try:
    import pyaudio
except ModuleNotFoundError:
    print("The 'pyaudio' module is not installed. Please install it using 'pip install pyaudio'.")
    exit(1)

# Handle missing modules gracefully
try:
    from textblob import TextBlob  # Added for sentiment analysis fallback
except ModuleNotFoundError:
    print("The 'textblob' module is not installed. Please install it using 'pip install textblob'.")
    exit(1)

# -------------------- Configuration -------------------- #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# -------------------- Voice Assistant -------------------- #
class VoiceAssistant:
    def __init__(self):
        pygame.mixer.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self._configure_audio_input()

    def _configure_audio_input(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def speak(self, text: str, lang: str = 'en') -> None:
        """Convert text to speech using gTTS with enhanced error handling"""
        try:
            print(f"\nAssistant: {text}")
            tts = gTTS(text=text, lang=lang)
            tts.save("output.mp3")
            
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            os.remove("output.mp3")
            
        except Exception as e:
            print(f"Audio Error: {str(e)}")
            raise RuntimeError("Text-to-speech conversion failed") from e

    def listen(self) -> Optional[str]:
        """Capture and recognize speech input with noise compensation"""
        try:
            with self.microphone as source:
                print("\nListening...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
                
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"API Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
        return None

# ------------------------ AI Integration ------------------------ #
class StoryProcessor:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.voice = VoiceAssistant()

    def _groq_streaming_query(
        self,
        prompt: str,
        model: str = "llama3-70b-8192",  # Updated default model
        temperature: float = 0.85,
        max_tokens: int = 300,
        top_p: float = 1.0,
        stream: bool = True,
        stop: Optional[List[str]] = None
    ) -> str:
        """Execute streaming query with real-time output handling"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop
            )
            
            collected_chunks = []
            print("\nProcessing: ", end="")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_chunks.append(content)
                
            return "".join(collected_chunks).strip()
        except Exception as e:
            print(f"Groq Error: {str(e)}")
            return "GROQ_API_ERROR"  # Special error indicator

    def generate_creative_summary(self, transcript: str) -> str:
        """Generate narrative-driven summary with dramatic structure using DeepSeek"""
        prompt = f"""Craft a 3-sentence engaging preview that begins with "The story starts like..." 
        and ends with a cliffhanger. Include dramatic pauses (marked by ...) and emotional hooks. 
        Conclude with: "Want to discover what happens next? Click play to start the audiobook!"
        
        Story Content: {transcript[:2000]}"""
        
        return self._groq_streaming_query(
            prompt,
            model="deepseek-r1-distill-llama-70b",
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stop=None
        )

    def analyze_emotional_profile(self, text: str) -> Tuple[str, str]:
        """Enhanced emotion analysis with fallback to TextBlob sentiment"""
        # First try Groq analysis
        analysis_prompt = f"""Analyze this text and respond in EXACT format:
        Mood:[happy/sad/angry/excited/neutral]
        Intensity:[1-5]
        Keywords:[3 comma-separated phrases]
        
        Text: {text}"""
        
        result = self._groq_streaming_query(analysis_prompt, model="llama3-70b-8192")
        
        if "GROQ_API_ERROR" in result:
            return self._textblob_sentiment_analysis(text)
            
        return self._parse_emotional_analysis(result)

    def _parse_emotional_analysis(self, response: str) -> Tuple[str, str]:
        """Extract structured data from analysis response"""
        mood = "neutral"
        try:
            for line in response.split('\n'):
                if line.startswith("Mood:"):
                    mood = line.split(':')[1].strip().lower()
                    break
            return (mood, response)
        except Exception:
            return self._textblob_sentiment_analysis(response)

    def _textblob_sentiment_analysis(self, text: str) -> Tuple[str, str]:
        """Fallback sentiment analysis using TextBlob"""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.2:
                mood = "happy"
            elif polarity < -0.2:
                mood = "sad"
            else:
                mood = "neutral"
                
            return (mood, f"TextBlob Analysis: Mood={mood}, Polarity={polarity:.2f}")
        except Exception as e:
            print(f"Sentiment Analysis Error: {str(e)}")
            return ("neutral", "Fallback analysis failed")

# ------------------------ Interaction Flow ------------------------ #
class StoryRecommender:
    def __init__(self):
        self.processor = StoryProcessor()
        self.user_mood = None

    def process_audio(self, audio_path: str) -> None:
        """Full processing pipeline for audio content"""
        try:
            transcript = self._transcribe_audio(audio_path)
            if not transcript:
                raise ValueError("Transcription failed")
            
            # Generate creative preview
            self.processor.voice.speak("Crafting immersive preview...")
            summary = self.processor.generate_creative_summary(transcript)
            self.processor.voice.speak(summary)
            
            # Story analysis
            story_mood, analysis = self.processor.analyze_emotional_profile(transcript)
            print(f"\nStory Mood: {story_mood.upper()}")
            
            # User mood detection
            self._get_user_mood()
            self._provide_recommendation(story_mood)
            
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            self.processor.voice.speak("Could not process the story")

    def _transcribe_audio(self, path: str) -> Optional[str]:
        """Handle audio transcription with Whisper"""
        try:
            print(f"\nTranscribing {os.path.basename(path)}...")
            result = subprocess.run(
                ["whisper", path, "--model", "base", "--output_format", "txt"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Transcription Error: {e.stderr.decode()}")
            return None
        except FileNotFoundError:
            print("Error: Whisper not found. Install with 'pip install openai-whisper'")
            return None

    def _get_user_mood(self) -> None:
        """Multi-modal mood input handling"""
        self.processor.voice.speak("Should I detect your mood by voice or text?")
        choice = input("\nChoose input method (voice/text): ").lower()
        
        if choice == 'voice':
            self._voice_mood_detection()
        else:
            self._text_mood_detection()

    def _voice_mood_detection(self) -> None:
        """Voice-based mood recognition flow"""
        self.processor.voice.speak("Please describe your current mood in a few words")
        attempt = 0
        while attempt < 3:
            user_input = self.processor.voice.listen()
            if user_input:
                self.user_mood, _ = self.processor.analyze_emotional_profile(user_input)
                return
            attempt += 1
            self.processor.voice.speak("Let me try that again...")
        self.user_mood = "neutral"

    def _text_mood_detection(self) -> None:
        """Text-based mood input with validation"""
        while True:
            user_input = input("\nHow are you feeling right now? ").strip()
            if user_input:
                self.user_mood, _ = self.processor.analyze_emotional_profile(user_input)
                return
            print("Please share your mood to get better recommendations")

    def _provide_recommendation(self, story_mood: str) -> None:
        """Personalized recommendation logic"""
        if self.user_mood == story_mood:
            self.processor.voice.speak(f"Perfect match! This {story_mood} story is ideal for your current mood.")
        else:
            self.processor.voice.speak(f"Based on your mood, this {story_mood} story offers a fresh perspective.")

# ------------------------ Main Execution ------------------------ #
if __name__ == "__main__":
    print("Kuku FM Creative Story Processor")
    recommender = StoryRecommender()
    
    while True:
        try:
            raw_path = input("\nEnter audio path (or 'exit'): ").strip()
            input_path = raw_path.strip("'\"")  # Remove surrounding quotes
            
            if input_path.lower() == 'exit':
                break
                
            # Validate file path
            if not os.path.exists(input_path):
                print(f"Error: File not found - {input_path}")
                print("Note: Avoid spaces in filename or use quotes around path")
                print("Trying system path...")
                input_path = os.path.abspath(os.path.expanduser(input_path))
                
            if not os.path.exists(input_path):
                print(f"File not found: {input_path}")
                print("Please provide full path to audio file")
                continue
            
            if not os.access(input_path, os.R_OK):
                print(f"Permission denied: {input_path}")
                print("Ensure the file is readable and try again.")
                continue
            
            print(f"Processing: {os.path.basename(input_path)}")
            recommender.process_audio(input_path)
            
            if input("\nProcess another file? (y/n): ").lower() != 'y':
                print("Happy listening!")
                break
                
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            break
            
        except Exception as e:
            print(f"Critical Error: {str(e)}")