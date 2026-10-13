import os
import sys
import subprocess
import json

def get_duration(file_path):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def mux_scene_audio(scene_num):
    audio_dir = os.path.abspath("demo_automation/voiceover_audio")
    video_dir = os.path.abspath("demo_automation/raw_videos")
    output_dir = os.path.abspath("demo_automation/final_videos")
    os.makedirs(output_dir, exist_ok=True)

    # Find matching audio
    audio_file = None
    for ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]:
        candidate = os.path.join(audio_dir, f"scene{scene_num}{ext}")
        if os.path.exists(candidate):
            audio_file = candidate
            break

    if not audio_file:
        print(f"[ERROR] No audio file found for scene {scene_num} in {audio_dir}.")
        print(f"Please place your audio recording as: scene{scene_num}.mp3 (or .wav / .m4a)")
        return False

    # Find video file
    video_file = os.path.join(output_dir, f"scene{scene_num}_noaudio.mp4")
    if not os.path.exists(video_file):
        # Fallback to existing scene{scene_num}.mp4 if noaudio doesn't exist
        video_file = os.path.join(output_dir, f"scene{scene_num}.mp4")

    if not os.path.exists(video_file):
        print(f"[ERROR] Video for scene {scene_num} not found at {video_file}")
        return False

    audio_dur = get_duration(audio_file)
    video_dur = get_duration(video_file)
    print(f"Scene {scene_num}: Audio length = {audio_dur:.2f}s, Video length = {video_dur:.2f}s")

    final_output = os.path.join(output_dir, f"scene{scene_num}.mp4")
    temp_output = os.path.join(output_dir, f"scene{scene_num}_muxed_temp.mp4")

    if audio_dur > video_dur:
        pad_dur = audio_dur - video_dur + 1.0  # pad 1 extra second at end
        print(f"Extending final video frame by {pad_dur:.2f}s to match audio duration...")
        filter_str = f"tpad=stop_mode=clone:stop_duration={pad_dur}"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-i", audio_file,
            "-filter_complex", f"[0:v]{filter_str}[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            temp_output
        ]
    else:
        print("Muxing audio with video...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            temp_output
        ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        if os.path.exists(final_output):
            os.remove(final_output)
        os.rename(temp_output, final_output)
        print(f"[SUCCESS] Scene {scene_num} exported with voiceover to:\n  {final_output}")
        return True
    else:
        print("[ERROR] FFmpeg error:", res.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scene = int(sys.argv[1])
        mux_scene_audio(scene)
    else:
        print("Usage: python sync_audio.py <scene_number>")
