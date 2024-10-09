import numpy as np
import sounddevice as sd
import whisper
import webrtcvad
import json
from io import BytesIO
import pygame
import pyautogui
import os
import base64
from google.cloud import texttospeech
from datetime import datetime
import warnings
import aiohttp
import asyncio
from collections import deque
import socket
import threading






LLM = 'Openhermes2.5-Mistral'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mixtral-410509-b4e97c8839e3.json"
whisper_model = whisper.load_model("small.en")
conversation = deque(maxlen=50)

client = texttospeech.TextToSpeechClient()
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")


def speak(text, language_code="en-US", voice_name="en-US-Neural2-F", speaking_rate=1.1):
    # Ensure that text is a non-empty string
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to be spoken must be a non-empty string")

    synthesis_input = texttospeech.SynthesisInput(text=text)


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

screenshot_path = None
screenshot_base64 = None


class Memory:
    def __init__(self, content, role, timestamp=None, tags=None):
        self.content = content
        self.role = role
        self.timestamp = timestamp if timestamp else datetime.now()
        self.tags = tags if tags else []
    def __repr__(self):
        return f"Memory(content={self.content}, role={self.role}, timestamp={self.timestamp}, tags={self.tags})"

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

# Your Ollama API URL
OLLAMA_API_URL = 'http://localhost:11434/api/chat'

# Initialize VAD
vad = webrtcvad.Vad(1)  # '1' is the aggressiveness level



def record_audio(fs=16000, post_speech_silence_limit=30):
    print("Listening...")  # Print once when listening starts
    frame_duration = 30  # Frame duration in ms
    frame_size = int(fs * frame_duration / 1000)  # Frame size in samples
    has_printed_listening = False  # Flag to track whether "Listening..." has been printed

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
    save_to_memory = "your memory" in transcribed_text.lower()

    if "on the screen" in transcribed_text.lower():
        screenshot_base64 = take_screenshot()

    return transcribed_text, save_to_memory



def prepare_ollama_messages(conversation_history, text, screenshot_base64):
    messages = [{"role": "user" if memory.role == "user" else "assistant", "content": memory.content}
                for memory in conversation_history]
    messages.append({"role": "user", "content": text})
    if screenshot_base64:
        messages.append({"role": "user", "images": [screenshot_base64]})
    return messages

async def call_ollama_api(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_API_URL, json=data) as response:
            if response.status != 200:
                print(f"HTTP error: {response.status}")
                return None
            return await response.text()
def process_ollama_response(response_text):
    llm_response = ''
    processed_ids = set()
    for line in response_text.splitlines():  # Use splitlines() to process each line
        if line:
            decoded_line = json.loads(line)
            message_id = decoded_line.get('created_at', '')
            if message_id not in processed_ids:
                if decoded_line['message']['role'] == 'assistant':
                    llm_response += decoded_line['message']['content']
                processed_ids.add(message_id)
            if decoded_line.get('done', False):
                break
    return llm_response.strip()


async def send_to_ollama(socket_client, text, save_to_memory, screenshot_base64=None):
    print("Sending to Ollama:", text)
    await socket_client.send_message("thinking")  # Send "thinking" message to Java app

    # Prepare messages from conversation
    messages = prepare_ollama_messages(list(conversation), text, screenshot_base64)
    data = {
        "model": LLM,
        "messages": messages,
        "options": {"num_predict": 128, "temperature": 0.9, "repeat_penalty": 1.2}
    }

    response_text = await call_ollama_api(data)
    if response_text is None:
        print(f"HTTP error")
        return None, screenshot_base64

    try:
        llm_response = process_ollama_response(response_text)

        # Save to memory.json if the flag is set
        if save_to_memory:
            save_memory([Memory(content=text, role="user")], 'memory.json')

        return llm_response, screenshot_base64
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON response")
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

class SocketClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.conn = None
        self.connected = False

    def start(self):
        threading.Thread(target=self.run_server, daemon=True).start()

    def run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            print(f"Python server listening on {self.host}:{self.port}")
            self.conn, addr = s.accept()
            self.connected = True
            print(f'Connected by {addr}')

            while self.connected:
                try:
                    data = self.conn.recv(1024)
                    if not data:
                        break
                    print("Received from Java:", data.decode())
                except Exception as e:
                    print(f"Connection error: {e}")
                    break

    async def send_message(self, message):
        if self.conn and self.connected:
            try:
                self.conn.sendall(message.encode())
            except Exception as e:
                print(f"Error sending message: {e}")

    def close(self):
        self.connected = False
        if self.conn:
            self.conn.close()

# Usage example

async def main():
    socket_client = SocketClient()
    socket_client.start()  # Start the socket server in a separate thread

    conversation.extend(load_memory('memory.json'))  # Load existing memories

    initial_prompt = read_initial_prompt('prompt.txt')
    if initial_prompt:
        initial_memory = Memory(content="Assistant: " + initial_prompt, role="assistant")
        conversation.append(initial_memory)

    while True:
        # Wait until the spoken response has finished playing
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)  # non-blocking sleep

        await socket_client.send_message("listening")
        audio_data = record_audio()
        transcribed_text, save_to_memory = transcribe_audio(audio_data, whisper_model)

        if transcribed_text.strip():
            await socket_client.send_message("talking")
            user_memory = Memory(content=transcribed_text, role="user")
            conversation.append(user_memory)  # Add to short-term memory

            llm_response, _ = await send_to_ollama(socket_client, transcribed_text, save_to_memory, screenshot_base64)

            if llm_response:
                response_text = llm_response
                assistant_memory = Memory(content=response_text, role="assistant")
                conversation.append(assistant_memory)

                print("LLM Response:", response_text)
                sound = speak(response_text)
                pygame.mixer.music.load(sound, 'mp3')
                pygame.mixer.music.play()
            else:
                print("No response received from Ollama.")

        socket_client.close()

if __name__ == "__main__":
    asyncio.run(main())
