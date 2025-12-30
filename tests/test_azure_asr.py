"""
Simple test script for Azure ASR.
Usage: uv run tests/test_azure_asr.py
"""
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def test_azure_asr():
    # Check if azure SDK is available
    try:
        import azure.cognitiveservices.speech as speechsdk
        print("✓ azure-cognitiveservices-speech is installed")
    except ImportError as e:
        print(f"✗ azure-cognitiveservices-speech is NOT installed: {e}")
        print("\nTo install, run:")
        print("  uv add azure-cognitiveservices-speech")
        return

    # Get credentials from environment
    key = os.getenv("SPEECH_KEY", "")
    region = os.getenv("SPEECH_REGION", "")
    endpoint = os.getenv("SPEECH_ENDPOINT", "")

    if not key:
        print("✗ SPEECH_KEY is not set in .env")
        return
    print(f"✓ SPEECH_KEY is set")

    if not region and not endpoint:
        print("✗ SPEECH_REGION or SPEECH_ENDPOINT is required in .env")
        return
    print(f"✓ SPEECH_REGION={region or '(using endpoint)'}")

    # Create speech config
    try:
        if endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
        else:
            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_recognition_language = "zh-CN"
        print("✓ SpeechConfig created successfully")
    except Exception as e:
        print(f"✗ Failed to create SpeechConfig: {e}")
        return

    # Create a simple test audio (1 second of silence)
    sample_rate = 16000
    duration = 1.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Try transcription
    try:
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
            speech_config=speech_config,
            audio_config=audio_config,
        )
        result = recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"✓ Recognition succeeded: {result.text}")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("✓ No speech detected (expected for silence)")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"✗ Recognition canceled: {cancellation.reason}")
            if cancellation.error_details:
                print(f"   Error details: {cancellation.error_details}")
        else:
            print(f"? Unknown result: {result.reason}")
            
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return

    print("\n✓ Azure ASR test completed successfully!")


if __name__ == "__main__":
    test_azure_asr()
