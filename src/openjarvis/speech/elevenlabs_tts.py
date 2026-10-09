"""ElevenLabs text-to-speech backend.

Uses the ElevenLabs REST API for voice synthesis.
Requires ELEVENLABS_API_KEY environment variable or config.
"""

from __future__ import annotations

import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
_DEFAULT_VOICE_ID = "Gubgw9l4dtIoQA9YZHgx"  # Brian / user-selected voice


def _elevenlabs_synthesize(
    api_key: str,
    text: str,
    voice_id: str,
    model: str = "eleven_multilingual_v2",
) -> bytes:
    """Call ElevenLabs TTS API and return raw audio bytes."""
    url = f"{_ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.content


@TTSRegistry.register("elevenlabs")
@TTSRegistry.register("elevenlabs_tts")
class ElevenLabsTTSBackend(TTSBackend):
    """ElevenLabs TTS backend — high-fidelity AI voice synthesis."""

    backend_id = "elevenlabs"

    def __init__(
        self, *, api_key: str = "", model: str = "eleven_multilingual_v2"
    ) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._model = model

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY environment variable is not set. "
                "Set it via: $env:ELEVENLABS_API_KEY='your_key'"
            )

        target_voice = voice_id or _DEFAULT_VOICE_ID

        audio = _elevenlabs_synthesize(
            self._api_key,
            text,
            voice_id=target_voice,
            model=self._model,
        )

        return TTSResult(
            audio=audio,
            format=output_format,
            voice_id=target_voice,
            metadata={"backend": "elevenlabs", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        if not self._api_key:
            return []
        try:
            resp = httpx.get(
                f"{_ELEVENLABS_API_BASE}/voices",
                headers={"xi-api-key": self._api_key},
                timeout=30.0,
            )
            resp.raise_for_status()
            return [v.get("voice_id", "") for v in resp.json().get("voices", []) if "voice_id" in v]
        except Exception:
            return [_DEFAULT_VOICE_ID]

    def health(self) -> bool:
        return bool(self._api_key)
