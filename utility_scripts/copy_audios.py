import os
import shutil

def copy_speaker_audio_files():
    source_dir = "youtube_scraping/ham"
    target_dir = "data/audios/raw_audios/ham"
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file in ["processed_audio.mp3"]:
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                folder_name = os.path.basename(os.path.dirname(source_path))

                new_filename = folder_name[:20] + "_" + file
                target_path = os.path.join(target_dir, new_filename)
                shutil.copy2(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

if __name__ == "__main__":
    copy_speaker_audio_files()
