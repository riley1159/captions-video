import os
import subprocess
import whisper
import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile


def extract_audio_from_video(input_video, audio_output):
    """
    Extract audio from the input video file and save it as an audio file.
    """
    command = [
        "ffmpeg",
        "-i", input_video,
        "-q:a", "0",
        "-map", "a",
        "-y",
        audio_output
    ]
    subprocess.run(command, check=True)


def transcribe_audio_with_whisper(audio_path):
    """
    Use Whisper to transcribe the audio and return segments with start and end timestamps.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["segments"]


def get_word_timestamps(segments):
    """
    Break down transcribed segments into individual words with precise timestamps.
    """
    words_with_timestamps = []
    for segment in segments:
        segment_text = segment["text"]
        segment_start = segment["start"]
        segment_end = segment["end"]

        words = segment_text.split()
        if not words:
            continue

        duration_per_word = (segment_end - segment_start) / len(words)

        current_start = segment_start
        for word in words:
            current_end = current_start + duration_per_word
            words_with_timestamps.append({
                "word": word,
                "start": current_start,
                "end": current_end
            })
            current_start = current_end

    return words_with_timestamps


def create_srt_file(word_timestamps, output_srt_path):
    """
    Generate an SRT file from word timestamps.
    """
    def convert_seconds_to_srt_format(seconds):
        """
        Convert seconds to the SRT timestamp format (HH:MM:SS,ms).
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    with open(output_srt_path, 'w', encoding='utf-8') as srt_file:
        for index, word_data in enumerate(word_timestamps):
            start_time = word_data["start"]
            end_time = word_data["end"]

            start_srt = convert_seconds_to_srt_format(start_time)
            end_srt = convert_seconds_to_srt_format(end_time)

            srt_file.write(f"{index + 1}\n")
            srt_file.write(f"{start_srt} --> {end_srt}\n")
            srt_file.write(f"{word_data['word']}\n\n")


def create_captioned_video(input_video, word_timestamps, output_video):
    """
    Create a captioned video with subtitles embedded at the center of each frame.
    """
    video = cv2.VideoCapture(input_video)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = 1080  # Force resolution to 1080x1920
    frame_height = 1920

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Ensure a font file exists
    font_path = "Helvetica.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf"
        if not os.path.exists(font_path):
            raise FileNotFoundError("No suitable font file found (Helvetica.ttf or arial.ttf).")
    font = ImageFont.truetype(font_path, 80)

    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_time = frame_count / fps

        # Resize frame to forced resolution
        frame = cv2.resize(frame, (frame_width, frame_height))

        current_word = None
        for word_data in word_timestamps:
            if word_data["start"] <= frame_time < word_data["end"]:
                current_word = word_data["word"]
                break

        if current_word:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)

            text_width, text_height = font.getbbox(current_word)[2:]
            text_x = (frame_width - text_width) // 2
            text_y = (frame_height - text_height) // 2

            draw.text((text_x, text_y), current_word, font=font, fill="yellow")
            frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_count += 1

    video.release()
    out.release()


def process_video(video_file):
    """
    Process the input video: extract audio, transcribe, add captions, and generate an SRT file.
    """
    audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    captioned_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    final_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    srt_path = tempfile.NamedTemporaryFile(suffix=".srt", delete=False).name

    # Extract audio
    extract_audio_from_video(video_file, audio_path)

    # Transcribe audio
    segments = transcribe_audio_with_whisper(audio_path)
    word_timestamps = get_word_timestamps(segments)

    # Create captioned video
    create_captioned_video(video_file, word_timestamps, captioned_path)

    # Generate SRT file
    create_srt_file(word_timestamps, srt_path)

    # Add audio back to captioned video
    command = [
        "ffmpeg",
        "-i", captioned_path,
        "-i", video_file,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-y",
        final_output
    ]
    subprocess.run(command, check=True)

    return final_output, srt_path


def process_and_return(video):
    """
    Wrapper for processing video and returning the output paths for preview and download.
    """
    output_video_path, srt_path = process_video(video)

    # Gradio expects three outputs: the video preview, video download, and SRT download.
    return output_video_path, output_video_path, srt_path


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Automatic Video Captioning Tool")

    with gr.Row():
        video_input = gr.Video(label="Upload Video", interactive=True)
        video_output = gr.Video(label="Captioned Video (Preview)", interactive=False)
        download_video_button = gr.File(label="Download Captioned Video")
        download_srt_button = gr.File(label="Download SRT File")

    submit = gr.Button("Generate Captions")
    submit.click(
        process_and_return,
        inputs=video_input,
        outputs=[video_output, download_video_button, download_srt_button]
    )

demo.launch()