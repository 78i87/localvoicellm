import numpy as np
import sounddevice as sd
import whisper
import webrtcvad
from io import BytesIO
import pygame
import pyautogui
import os
import base64
from google.cloud import texttospeech
from datetime import datetime
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
import aiohttp
import asyncio
import json
from collections import deque
import re


app = FastAPI()

# Environment variables and model loading
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mixtral-410509-b4e97c8839e3.json"
whisper_model = whisper.load_model("small")
vad = webrtcvad.Vad(1)  # Initialize Voice Activity Detection

# Text-to-Speech Client and Warning Configuration
client = texttospeech.TextToSpeechClient()
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")

# Deque for conversation history
conversation = deque(maxlen=50)

screenshot_path = None
screenshot_base64 = None

def speak(text, language_code="en-US", voice_name="en-US-Neural2-F", speaking_rate=1.1):
    # Filter out text within square brackets
    text_for_speech = re.sub(r'\[.*?\]', '', text)
    # Ensure that text is a non-empty string
    if not isinstance(text_for_speech, str) or not text_for_speech.strip():
        raise ValueError("Text to be spoken must be a non-empty string")

    synthesis_input = texttospeech.SynthesisInput(text=text_for_speech)


    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return BytesIO(response.audio_content)

class Memory(BaseModel):
    content: str
    role: str
    timestamp: datetime = datetime.now()
    tags: list = []

def retrieve_memory(conversation, keyword):
    for memory in reversed(conversation):
        if keyword in memory.content:
            return memory
    return None

def save_memory(conversation_history, memory_file):
    try:
        with open(memory_file, 'w', encoding='utf-8') as file:
            memory_data = [memory.__dict__ for memory in conversation_history]
            json.dump(memory_data, file, default=str, indent=4)
        print(f"Conversation saved to {memory_file}.")
    except IOError as e:
        print(f"Error saving memory to {memory_file}: {e}")

def load_memory(memory_file):
    try:
        with open(memory_file, 'r', encoding='utf-8') as file:
            memory_data = json.load(file)
            loaded_memory = [Memory(**memory) for memory in memory_data]
            return loaded_memory[-50:]  # Keep only the last 50 memories
    except FileNotFoundError:
        print(f"No existing memory found in {memory_file}. Starting with an empty memory.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error reading memory from {memory_file}: {e}")
        return []


def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot_path = os.path.join('/Volumes/T7/pycharm/aistream', 'screenshot.png')
    screenshot.save(screenshot_path)

    with open(screenshot_path, "rb") as image_file:
        screenshot_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    return screenshot_path, screenshot_base64


# Initialize Whisper model
model = whisper.load_model("small.en")

# Initialize VAD
vad = webrtcvad.Vad(1)  # '1' is the aggressiveness level

def record_audio(fs=16000, post_speech_silence_limit=30):
    print("Listening...")
    frame_duration = 30  # Frame duration in ms
    frame_size = int(fs * frame_duration / 1000)  # Frame size in samples

    with sd.RawInputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        audio = np.array([], dtype=np.int16)
        post_speech_silence_frames = 0
        speech_detected = False

        while True:
            data, _ = stream.read(frame_size)
            data_array = np.frombuffer(data, dtype=np.int16)
            is_speech = vad.is_speech(data_array.tobytes(), fs)

            if is_speech:
                speech_detected = True
                post_speech_silence_frames = 0
                audio = np.append(audio, data_array)
            else:
                if speech_detected:
                    post_speech_silence_frames += 1
                    if post_speech_silence_frames > post_speech_silence_limit:
                        break

    return audio

def transcribe_audio(audio, model):
    global screenshot_base64
    audio = audio.astype(np.float32)
    audio /= np.iinfo(np.int16).max
    result = model.transcribe(audio)
    transcribed_text = result['text']

    # Detect special commands like "on the screen" for screenshots
    if "on the screen" in transcribed_text.lower():
        screenshot_base64 = take_screenshot()

    return transcribed_text

async def call_llm_api(data):
    # Replace with your FastAPI server URL
    llm_api_url = 'http://localhost:8000/chat'

    async with aiohttp.ClientSession() as session:
        async with session.post(llm_api_url, json=data) as response:
            if response.status != 200:
                print(f"HTTP error: {response.status}")
                return None
            return await response.json()  # Assuming the response is JSON


def process_llm_response(response_json):
    # Modify according to how the FastAPI response is structured
    return response_json.get('response', '')


async def send_to_llm(text, screenshot_base64=None):
    print("Sending to llm:", text)

    # Prepare your request payload here. Adjust to match the FastAPI endpoint's expected format.
    data = {
        "text": text,
        # Include any other data required by your FastAPI endpoint
    }

    response_json = await call_llm_api(data)
    if response_json is None:
        print("HTTP error or no response")
        return None, screenshot_base64

    try:
        llm_response = process_llm_response(response_json)

        # Check for [memory] tag in LLM response
        if "[memory]" in llm_response:
            save_memory(list(conversation), 'memory.json')

        return llm_response, screenshot_base64
    except Exception as e:
        print(f"Error processing response: {e}")
        return None, screenshot_base64


def read_initial_prompt(filename):
    """ Reads the initial prompt from a file. """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""  # Return an empty string or a default message if file not found

pygame.init()
pygame.mixer.init()

async def main():
    conversation.extend(load_memory('memory.json'))  # Load existing memories

    initial_prompt = read_initial_prompt('prompt.txt')
    if initial_prompt:
        initial_memory = Memory(content="Assistant: " + initial_prompt, role="assistant")
        conversation.append(initial_memory)

    screenshot_base64 = None  # Initialize screenshot_base64 here

    while True:
        # Wait until the spoken response has finished playing
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

        audio_data = record_audio()
        transcribed_text = transcribe_audio(audio_data, whisper_model)

        if transcribed_text.strip():
            user_memory = Memory(content=transcribed_text, role="user")
            conversation.append(user_memory)  # Add to short-term memory

            llm_response, screenshot_base64 = await send_to_llm(transcribed_text, screenshot_base64)

            if llm_response:
                response_text = llm_response
                assistant_memory = Memory(content=response_text, role="assistant")
                conversation.append(assistant_memory)

                print("LLM Response:", response_text)
                sound = speak(response_text)
                pygame.mixer.music.load(sound, 'mp3')
                pygame.mixer.music.play()
            else:
                print("No response received from LLM.")

if __name__ == "__main__":
    asyncio.run(main())
