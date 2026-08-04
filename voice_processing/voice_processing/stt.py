import io

from openai import OpenAI


class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def speech2text(self, wav_data):
        """Transcribe an in-memory WAV recorded by MicController."""
        audio_file = io.BytesIO(wav_data)
        audio_file.name = "voice_command.wav"
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        print("STT 결과: ", transcript.text)
        return transcript.text
