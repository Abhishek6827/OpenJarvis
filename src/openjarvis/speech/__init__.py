"""Speech subsystem â€” speech-to-text and text-to-speech backends."""

import importlib

# Optional STT backends â€” each registers itself via @SpeechRegistry.register()
for _mod in ("faster_whisper", "openai_whisper", "deepgram"):
    try:
        importlib.import_module(f".{_mod}", __name__)
    except ImportError:
        pass

# Optional TTS backends â€” each registers itself via @TTSRegistry.register()
for _mod in ("cartesia_tts", "kokoro_tts", "openai_tts", "elevenlabs_tts"):
    try:
        importlib.import_module(f".{_mod}", __name__)
    except ImportError:
        pass

