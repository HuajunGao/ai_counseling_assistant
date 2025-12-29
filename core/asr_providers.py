import abc
import io
import os
from typing import Optional

import numpy as np
import soundfile as sf


class ASRProvider(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return transcription text for the given mono float32 audio."""


class FasterWhisperProvider(ASRProvider):
    def __init__(self, model_size: str, device: str, compute_type: str, language: str):
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=None if self.language == "auto" else self.language,
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text


class OpenAIProvider(ASRProvider):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(f"openai package not available: {exc}") from exc

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI ASR.")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        buffer = io.BytesIO()
        setattr(buffer, "name", "audio.wav")
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
        )
        return (response.text or "").strip()


class FunASRProvider(ASRProvider):
    def __init__(
        self,
        model: str,
        device: str,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        disable_update: bool = False,
        trust_remote_code: bool = False,
    ):
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(f"funasr package not available: {exc}") from exc

        model_kwargs = {
            "model": model,
            "device": device,
            "disable_update": disable_update,
        }
        if vad_model:
            model_kwargs["vad_model"] = vad_model
        if punc_model:
            model_kwargs["punc_model"] = punc_model
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        self.model = AutoModel(**model_kwargs)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        result = self.model.generate(audio)
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        if isinstance(result, list):
            text = " ".join((item.get("text") or "").strip() for item in result if isinstance(item, dict))
            return text.strip()
        return ""


class AzureProvider(ASRProvider):
    def __init__(self, key: str, region: str | None, endpoint: str | None, language: str):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception as exc:
            raise RuntimeError(f"azure.cognitiveservices.speech not available: {exc}") from exc

        if not key:
            raise RuntimeError("SPEECH_KEY is required for Azure ASR.")
        if not region and not endpoint:
            raise RuntimeError("SPEECH_REGION or SPEECH_ENDPOINT is required for Azure ASR.")

        if endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
        else:
            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)

        if language == "zh":
            speech_config.speech_recognition_language = "zh-CN"
        elif language == "auto":
            pass
        else:
            speech_config.speech_recognition_language = language

        self._speechsdk = speechsdk
        self._speech_config = speech_config

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        speechsdk = self._speechsdk
        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=sample_rate,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        push_stream.write(pcm.tobytes())
        push_stream.close()

        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return (result.text or "").strip()
        return ""


def create_asr_provider(config) -> ASRProvider:
    backend = config.ASR_BACKEND.lower()
    if backend == "azure":
        return AzureProvider(
            key=config.SPEECH_KEY,
            region=config.SPEECH_REGION or None,
            endpoint=config.SPEECH_ENDPOINT or None,
            language=config.ASR_LANGUAGE,
        )
    if backend == "openai":
        return OpenAIProvider(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_TRANSCRIBE_MODEL,
            base_url=config.OPENAI_BASE_URL or None,
        )
    if backend == "funasr":
        return FunASRProvider(
            model=config.FUNASR_MODEL,
            device=config.FUNASR_DEVICE,
            vad_model=config.FUNASR_VAD_MODEL or None,
            punc_model=config.FUNASR_PUNC_MODEL or None,
            disable_update=config.FUNASR_DISABLE_UPDATE,
            trust_remote_code=config.FUNASR_TRUST_REMOTE_CODE,
        )
    return FasterWhisperProvider(
        model_size=config.WHISPER_MODEL_SIZE,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
        language=config.ASR_LANGUAGE,
    )
