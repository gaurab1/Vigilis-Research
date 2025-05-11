import os
import numpy as np
import yt_dlp
from pydub import AudioSegment
from pydub.silence import split_on_silence
import argparse
import librosa
import librosa.display
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# import whisper
import json
from openai import OpenAI
import datetime

# Load the Whisper model if you want to do it locally
# model = whisper.load_model("small.en")

def create_output_directory(video_name=None):
    timestamp = datetime.datetime.now().strftime("%H%M")
    if video_name:
        # Sanitize the video name to remove invalid characters for filenames
        sanitized_name = ''.join(c for c in video_name if c.isalnum() or c in ' -_').strip()
        sanitized_name = sanitized_name.replace(' ', '_')
        sanitized_name = sanitized_name[:60].strip()
        dir_name = f"output_1{sanitized_name}"
    else:
        dir_name = f"output_{timestamp}"
    
    output_dir = os.path.join(os.getcwd(), dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def download_youtube_audio(url, output_path=None, output_filename="audio", skip_start=0):
    print(f"Downloading audio from: {url}")

    if output_path is None:
        output_path = os.getcwd()
    
    mp3_file = os.path.join(output_path, f"{output_filename}.mp3")
    
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, f"{output_filename}.%(ext)s"),
            'verbose': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                print(f"Successfully downloaded: {info.get('title', 'Unknown title')}")
            else:
                print("No information was returned from YouTube")
                return None
        
        if skip_start > 0:
            audio = AudioSegment.from_file(mp3_file, format="mp3")
            trimmed_audio = audio[skip_start * 1000:]
            trimmed_audio.export(mp3_file, format="mp3")
            print(f"Removed first {skip_start} seconds from the audio")
        print(f"Audio file saved to: {mp3_file}")
        return mp3_file
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return None

def extract_conversation(audio_file, visualization_file="speaker_clusters.png", output_dir=None):
    print(f"Processing audio file: {audio_file}")
    
    y, sr = librosa.load(audio_file, sr=None)
    
    # Calculate the duration of the audio in seconds
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")

    segment_length_sec = 1
    segment_samples = int(segment_length_sec * sr)
    
    # Split the audio into segments
    segments = []
    segment_times = []
    
    for start in range(0, len(y), segment_samples):
        end = min(start + segment_samples, len(y))
        segment = y[start:end]
        
        if len(segment) >= segment_samples * 0.5:
            segments.append(segment)
            segment_times.append((start / sr, end / sr))
    
    print(f"Split audio into {len(segments)} segments")
    
    # Extract features from each segment
    features = []
    for segment in segments:
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        
        # Combine features and take mean across time
        segment_features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.array([np.mean(spectral_centroid)]),
            np.array([np.mean(spectral_bandwidth)]),
            np.array([np.mean(spectral_rolloff)])
        ])
        
        features.append(segment_features)
    features = np.array(features)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    kmeans = KMeans(n_clusters=2, random_state=42, max_iter=1000)
    clusters = kmeans.fit_predict(scaled_features)
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    print(f"Number of clusters: {len(unique_clusters)}")
    print("Cluster counts:", dict(zip(unique_clusters, cluster_counts)))
    
    silhouette = silhouette_score(scaled_features, clusters)
    print(f"Silhouette score: {silhouette:.3f}")
    
    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    
    # Create a subplot for cluster visualization
    plt.subplot(2, 1, 1)
    colors = plt.cm.tab10(clusters)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=colors, s=50, alpha=0.7)
    plt.title('Speaker Clusters (PCA)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    # Create a subplot for speaker times
    plt.subplot(2, 1, 2)
    
    y_positions = [0, 1]
    colors = ['blue', 'red']
    
    for i, pos in enumerate(y_positions):
        speaker_segments = [(start, end-start) for j, (start, end) in enumerate(segment_times) if clusters[j] == i]
        
        for start, duration in speaker_segments:
            plt.barh(pos, duration, left=start, height=0.8, color=colors[i], alpha=0.6)
    
    plt.title('Speaker Timeline')
    plt.xlabel('Time (seconds)')
    plt.yticks(y_positions, ['Speaker 1', 'Speaker 2'])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    if segment_times:
        max_time = max(end for _, end in segment_times)
        plt.xlim(0, max_time)
    
    # Save the visualization
    if output_dir is None:
        output_dir = os.getcwd()
    plt.tight_layout()
    plt.savefig(visualization_file)
    plt.close()
    
    print(f"Saved visualization to {visualization_file}")
    
    audio = AudioSegment.from_file(audio_file)
    
    speaker1_audio = AudioSegment.silent(duration=len(audio))
    speaker2_audio = AudioSegment.silent(duration=len(audio))
    
    speaker1_times = []
    speaker2_times = []
    for i, (start_time, end_time) in enumerate(segment_times):
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment_audio = audio[start_ms:end_ms]
        
        if clusters[i] == 0:
            speaker1_audio = speaker1_audio.overlay(segment_audio, position=start_ms)
            if speaker1_times and start_time == speaker1_times[-1][1]:
                speaker1_times[-1] = (speaker1_times[-1][0], end_time)
            else:
                speaker1_times.append((start_time, end_time))
        else:
            speaker2_audio = speaker2_audio.overlay(segment_audio, position=start_ms)
            if speaker2_times and start_time == speaker2_times[-1][1]:
                speaker2_times[-1] = (speaker2_times[-1][0], end_time)
            else:
                speaker2_times.append((start_time, end_time))
            
    # Save the separated audio files
    if output_dir is None:
        output_dir = os.getcwd()
    speaker1_path = os.path.join(output_dir, "speaker1.mp3")
    speaker2_path = os.path.join(output_dir, "speaker2.mp3")
    
    speaker1_audio.export(speaker1_path, format="mp3")
    speaker2_audio.export(speaker2_path, format="mp3")
    
    print(f"Saved speaker 1 audio to {speaker1_path}")
    print(f"Saved speaker 2 audio to {speaker2_path}")
    
    return speaker1_times, speaker2_times

def transcribe_audio(audio_file, model_name="base") -> dict:
    print(f"Transcribing audio using Whisper...")
    
    # Transcribe with word-level timestamps
    # result = model.transcribe(audio_file, word_timestamps=True, language="en", fp16=False)
    client = OpenAI()
    audio = open(audio_file, "rb")
    result = client.audio.transcriptions.create(
        file=audio,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )
    
    transcript_with_timestamps = {}
    
    if hasattr(result, 'model_dump'):  # OpenAI API response
        result_dict = result.model_dump()

        for segment in result_dict['segments']:
            start_time = round(segment.get('start', 0), 2)
            end_time = round(segment.get('end', 0), 2)
            text = segment.get('text', '').strip()
            
            timestamp_key = f"{start_time:.2f}-{end_time:.2f}"
            transcript_with_timestamps[timestamp_key] = text
    else:  # Local Whisper model response
        for segment in result.get("segments", []):
            start_time = round(segment.get("start", 0), 2)
            end_time = round(segment.get("end", 0), 2)
            text = segment.get("text", "").strip()
            
            timestamp_key = f"{start_time:.2f}-{end_time:.2f}"
            transcript_with_timestamps[timestamp_key] = text
    
    return {
        "transcript": result_dict.get("text", "") if hasattr(result, 'model_dump') else result.get("text", ""),
        "segments": transcript_with_timestamps,
    }

def combine_transcripts(speaker1_transcript, speaker2_transcript, output_dir=None):
    segments1 = speaker1_transcript.get("segments", {})
    segments2 = speaker2_transcript.get("segments", {})
    
    proc_segments1 = [(float(key.split("-")[0]), float(key.split("-")[1]), value) for (key, value) in segments1.items()] 
    proc_segments2 = [(float(key.split("-")[0]), float(key.split("-")[1]), value) for (key, value) in segments2.items()] 

    proc_segments1.sort(key=lambda x: x[0])
    proc_segments2.sort(key=lambda x: x[0])

    overall_transcript = ""
    last_speaker = None
    i = 0
    j = 0
    while i < len(proc_segments1) or j < len(proc_segments2):
        if (i < len(proc_segments1) and j >= len(proc_segments2)) or \
           (i < len(proc_segments1) and j < len(proc_segments2) and proc_segments1[i][0] < proc_segments2[j][0]):
            if last_speaker == "Speaker 1":
                overall_transcript += f" {proc_segments1[i][2]}"
            else:
                overall_transcript += f"\nSpeaker 1: {proc_segments1[i][2]}"
            i += 1
            last_speaker = "Speaker 1"
        elif j < len(proc_segments2):
            if last_speaker == "Speaker 2":
                overall_transcript += f" {proc_segments2[j][2]}"
            else:
                overall_transcript += f"\nSpeaker 2: {proc_segments2[j][2]}"
            j += 1
            last_speaker = "Speaker 2"

    if output_dir is None:
        output_dir = os.getcwd()
    with open(os.path.join(output_dir, "combined_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(overall_transcript)
    
    return overall_transcript

def transcribe_parts(full_transcript, speaker1_times, speaker2_times, output_dir=None):
    print(speaker1_times)
    segments = full_transcript.get("segments", {})
    proc_segments = [(float(key.split("-")[0]), float(key.split("-")[1]), value) for (key, value) in segments.items()]

    proc_segments.sort(key=lambda x: x[0])

    i = 0
    j = 0
    overall_transcript = ""
    last_speaker = None

    for start, end, text in proc_segments:
        votes1 = 0
        votes2 = 0
        
        while i < len(speaker1_times) and speaker1_times[i][0] <= end:
            if max(start, speaker1_times[i][0]) <= min(end, speaker1_times[i][1]):
                votes1 += 1
            i += 1
            
        while j < len(speaker2_times) and speaker2_times[j][0] <= end:
            # Check if there's an overlap between the segment and speaker time
            if max(start, speaker2_times[j][0]) <= min(end, speaker2_times[j][1]):
                votes2 += 1
            j += 1

        i = 0
        j = 0

        if votes1 == 0 and votes2 == 0:
            if last_speaker == "Speaker 1":
                current_speaker = "Speaker 1"
            elif last_speaker == "Speaker 2":
                current_speaker = "Speaker 2"
            else:
                current_speaker = "Speaker 1"
        elif votes1 >= votes2:
            current_speaker = "Speaker 1"
        else:
            current_speaker = "Speaker 2"
        print(start, end, votes1, votes2, current_speaker)
        # Format the output based on the speaker
        if last_speaker == current_speaker:
            overall_transcript += f" {text}"
        else:
            # if last_speaker is not None and text[0].islower():
            #     splits = text.split('.')
            #     overall_transcript += f" {splits[0]}"
            #     if len(splits) > 1:
            #         overall_transcript += f"\n{current_speaker}: {splits[1].strip()}"
            #     else:
            #         current_speaker = last_speaker
            # else:
            overall_transcript += f"\n{current_speaker}: {text}"
        last_speaker = current_speaker

    if output_dir is None:
        output_dir = os.getcwd()
    with open(os.path.join(output_dir, "combined_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(overall_transcript)
    
    return overall_transcript

def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio, extract conversations, and transcribe")
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=WKLRKCVar6Y", 
                        help="YouTube video URL")
    parser.add_argument("--start_time", default=0, type=float,
                        help="Start time of conversation in seconds")
    parser.add_argument("--output", default="processed_audio.mp3", 
                        help="Output file path for processed YouTube audio")
    parser.add_argument("--speaker1_output", default="speaker1.mp3",
                        help="Output file for speaker 1")
    parser.add_argument("--speaker2_output", default="speaker2.mp3",
                        help="Output file for speaker 2")
    parser.add_argument("--visualization_file", default="speaker_clusters.png",
                        help="Path to save the speaker clustering visualization")
    parser.add_argument("--whisper_model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--transcript_output", default="transcript.json",
                        help="Output file for full transcript")
    parser.add_argument("--speaker1_transcript", default="speaker1_transcript.json",
                        help="Output file for speaker 1 transcript")
    parser.add_argument("--speaker2_transcript", default="speaker2_transcript.json",
                        help="Output file for speaker 2 transcript")
    
    args = parser.parse_args()
    
    video_title = None
    # Configure yt-dlp options for info extraction
    ydl_opts_info = {
        'quiet': True,
        'no_warnings': True
    }
        
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(args.url, download=False)
        video_title = info.get('title', 'Unknown')
        print(f"Video title: {video_title}")
    
    output_dir = 'output_' + video_title
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(args.output))[0]
    audio_file = download_youtube_audio(args.url, output_path=output_dir, output_filename=output_filename, skip_start=args.start_time)

    if audio_file:
        processing_input = audio_file
            
        print("\nExtracting conversation and separating speakers...")
        speaker1_times, speaker2_times = extract_conversation(
            processing_input, 
            visualization_file=os.path.join(output_dir, args.visualization_file),
            output_dir=output_dir
        )
            
        # Transcribe full audio
        # print("\nTranscribing full audio...")
        # full_transcript = transcribe_audio(processing_input, model_name=args.whisper_model)
            
        # # Save full transcript
        # with open(os.path.join(output_dir, args.transcript_output), 'w', encoding='utf-8') as f:
        #     json.dump(full_transcript, f, indent=2, ensure_ascii=False)
        # print(f"Full transcript saved to: {os.path.join(output_dir, args.transcript_output)}")

        # combined_transcript = transcribe_parts(full_transcript, speaker1_times, speaker2_times, output_dir)
        # Transcribe speaker 1
        print("\nTranscribing Speaker 1 audio...")
        speaker1_audio_path = os.path.join(output_dir, args.speaker1_output)
        speaker1_transcript = transcribe_audio(speaker1_audio_path, model_name=args.whisper_model)
            
        # Save speaker 1 transcript
        speaker1_transcript_path = os.path.join(output_dir, args.speaker1_transcript)
        with open(speaker1_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(speaker1_transcript, f, indent=2, ensure_ascii=False)
        print(f"Speaker 1 transcript saved to: {speaker1_transcript_path}")
            
        # Transcribe speaker 2
        print("\nTranscribing Speaker 2 audio...")
        speaker2_audio_path = os.path.join(output_dir, args.speaker2_output)
        speaker2_transcript = transcribe_audio(speaker2_audio_path, model_name=args.whisper_model)
            
        # Save speaker 2 transcript
        speaker2_transcript_path = os.path.join(output_dir, args.speaker2_transcript)
        with open(speaker2_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(speaker2_transcript, f, indent=2, ensure_ascii=False)
        print(f"Speaker 2 transcript saved to: {speaker2_transcript_path}")

        # Combine transcripts and save to output directory
        combined_transcript = combine_transcripts(speaker1_transcript, speaker2_transcript, output_dir)
        print(f"Combined transcript saved to: {os.path.join(output_dir, 'combined_transcript.txt')}")
    else:
        print("Failed to download YouTube audio.")

if __name__ == "__main__":
    main()