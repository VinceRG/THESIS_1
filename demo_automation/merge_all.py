import os
import subprocess

def merge_all_scenes():
    video_dir = os.path.abspath("demo_automation/final_videos")
    concat_list_path = os.path.join(video_dir, "concat_list.txt")
    output_path = os.path.join(video_dir, "complete_system_demo.mp4")

    lines = []
    for i in range(1, 9):
        scene_path = os.path.join(video_dir, f"scene{i}.mp4").replace('\\', '/')
        if os.path.exists(scene_path):
            lines.append(f"file '{scene_path}'\n")

    with open(concat_list_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Master demo compiled successfully to:\n  {output_path}")
    else:
        # If stream params differ slightly, re-encode concat
        cmd_reencode = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            output_path
        ]
        res2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
        if res2.returncode == 0:
            print(f"[SUCCESS] Master demo compiled successfully to:\n  {output_path}")
        else:
            print("[ERROR]:", res2.stderr)

if __name__ == "__main__":
    merge_all_scenes()
