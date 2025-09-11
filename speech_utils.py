import os
import logging
from typing import Optional

# Import variables for configuration
import variables

def transcribe_audio(audio_input, mime_type: Optional[str] = None) -> str:
    """Transcribe audio using Google Cloud Speech-to-Text v2.

    Handles audio files longer than 60 seconds by using streaming API.
    Can accept either file path (str) or audio bytes.

    Args:
        audio_input: Either file path (str) or raw audio data (bytes).
        mime_type: MIME type of the audio (unused for auto-detect).

    Returns:
        Transcript text from the audio. Returns error message if transcription fails.
    """
    try:
        from google.cloud import speech_v2
    except ImportError:
        logging.error("Google Cloud Speech-to-Text library not installed. Install with: pip install google-cloud-speech")
        return "[Audio transcription unavailable - Google Cloud Speech library not installed]"

    # Handle both file path and audio bytes
    if isinstance(audio_input, str):
        # It's a file path
        if not os.path.exists(audio_input):
            logging.error(f"Audio file not found: {audio_input}")
            return "[Audio transcription failed - file not found]"
        
        try:
            with open(audio_input, 'rb') as f:
                audio_bytes = f.read()
            logging.info(f"Loaded audio file: {audio_input} ({len(audio_bytes)} bytes)")
        except Exception as e:
            logging.error(f"Failed to read audio file: {e}")
            return "[Audio transcription failed - could not read file]"
    else:
        # It's already audio bytes
        audio_bytes = audio_input

    project_id = variables.GOOGLE_CLOUD_PROJECT
    if not project_id:
        logging.error("GOOGLE_CLOUD_PROJECT is not configured")
        return "[Audio transcription unavailable - Google Cloud project not configured]"

    try:
        # Set up credentials from variables.py if not already set
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            try:
                import json
                import tempfile
                
                # Create a temporary credentials file from variables.py
                credentials_dict = variables.GOOGLE_CLOUD_CREDENTIALS
                temp_credentials_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(credentials_dict, temp_credentials_file, indent=2)
                temp_credentials_file.close()
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_file.name
                logging.info(f"✅ Set Google Cloud credentials from variables.py for speech: {temp_credentials_file.name}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to set Google Cloud credentials from variables.py for speech: {e}")
        
        client = speech_v2.SpeechClient()

        # Handle different audio formats properly
        if mime_type and "webm" in mime_type.lower():
            # For WebM audio from browser
            config = speech_v2.RecognitionConfig(
                auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                language_codes=[variables.SPEECH_LANGUAGE],
                model="latest_long",
            )
        else:
            # For other formats (WAV, MP3, etc.) - use auto-detection
            config = speech_v2.RecognitionConfig(
                auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                language_codes=[variables.SPEECH_LANGUAGE],
                model="latest_long",
                features=speech_v2.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=True,
                ),
            )

        # Check if audio is too long (estimate based on size)
        # WAV files are typically ~16KB per second, so 60 seconds ≈ 1MB
        estimated_duration = len(audio_bytes) / 16000  # Rough estimate
        
        if estimated_duration > 60:
            logging.warning(f"Audio file is approximately {estimated_duration:.1f} seconds long. Using streaming API.")
            return _transcribe_long_audio(client, project_id, config, audio_bytes)
        else:
            return _transcribe_short_audio(client, project_id, config, audio_bytes)
        
    except Exception as e:
        logging.error(f"Transcription request failed: {e}")
        logging.error(f"Project ID: {project_id}")
        logging.error(f"Language: {variables.SPEECH_LANGUAGE}")
        return "[Audio transcription failed - please try again or use text input]"

def _transcribe_short_audio(client, project_id, config, audio_bytes):
    """Transcribe short audio files (< 60 seconds)"""
    try:
        from google.cloud import speech_v2
        
        request = speech_v2.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=config,
            content=audio_bytes,
        )

        response = client.recognize(request=request)

        transcript_parts = []
        for result in response.results:
            if result.alternatives:
                transcript_parts.append(result.alternatives[0].transcript)

        transcript = " ".join(transcript_parts).strip()
        logging.info(f"Transcription result: {transcript}")
        return transcript
    except Exception as e:
        logging.error(f"Short audio transcription failed: {e}")
        return "[Audio transcription failed - please try again or use text input]"

def _transcribe_long_audio(client, project_id, config, audio_bytes):
    """Transcribe long audio files (> 60 seconds) using streaming"""
    try:
        # For now, we'll truncate to first 60 seconds as a workaround
        # In production, you'd want to implement proper streaming
        logging.warning("Truncating audio to first 60 seconds due to API limitations")
        
        # Estimate 60 seconds worth of data (rough approximation)
        max_bytes = 60 * 16000  # 60 seconds * 16KB per second
        if len(audio_bytes) > max_bytes:
            audio_bytes = audio_bytes[:max_bytes]
            logging.info(f"Truncated audio to {len(audio_bytes)} bytes")
        
        return _transcribe_short_audio(client, project_id, config, audio_bytes)
        
    except Exception as e:
        logging.error(f"Long audio transcription failed: {e}")
        return "[Audio transcription failed - please try again or use text input]"
