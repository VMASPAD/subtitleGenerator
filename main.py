import subprocess
import whisper
import torch
import whisper.utils

def extract_audio(input_video, output_audio):
    subprocess.run(['ffmpeg', '-i', input_video, '-q:a', '0', '-map', 'a', output_audio], check=True)

def split_segments_by_word(segments, max_words_per_segment):
    new_segments = []
    for segment in segments:
        words = segment['text'].strip().split()
        start = segment['start']
        while words:
            text = ' '.join(words[:max_words_per_segment])
            words = words[max_words_per_segment:]
            end = start + (segment["end"] - segment["start"]) * (len(words) / len(segment["text"].split()))
            new_segments.append({'text': text, 'start': start, 'end': end})
            start = end
    return new_segments

def transcribe_audio(audio_path, vram_limit_gb, srt_output_path, max_words_per_segment): 
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(audio_path)
    segments = split_segments_by_word(result['segments'], max_words_per_segment)
    with open(srt_output_path, 'w', encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments):
            start = whisper.utils.format_timestamp(segment['start'], always_include_hours=True,decimal_marker=',')
            end = whisper.utils.format_timestamp(segment['end'], always_include_hours=True,decimal_marker=',')
            text = segment['text'].strip()
            srt_file.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")

def main():
    input_video = "test.mp4" 
    audio_path = "audio.wav"
    vram_limit_gb = 3.5
    srt_output_path = "output.srt"
    max_words_per_segment = 4

    extract_audio(input_video, audio_path)
    transcribe_audio(audio_path, vram_limit_gb, srt_output_path, max_words_per_segment)

if __name__ == "__main__":
    main()