# youtube_downloader_ui.py

# --- Dependency Checker and Installer ---
import sys
import subprocess
import importlib.util
import os # Also needed early for DOWNLOAD_DIR check

REQUIRED_PACKAGES = ['gradio', 'yt-dlp']

def check_and_install_packages(packages):
    """Checks if required packages are installed and installs them if not."""
    print("--- Checking required Python packages ---")
    all_installed = True
    for package in packages:
        module_name = package.split('==')[0].replace('-', '_') # Handle potential version pins and name differences
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"Package '{package}' not found. Attempting installation using pip...")
            try:
                # Ensure pip is run for the current Python interpreter
                # Use check_call to raise an error if installation fails
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed '{package}'.")
                # Optional: Verify again after installation
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    print(f"!! ERROR: Installation of '{package}' reported success, but the module '{module_name}' still cannot be found.")
                    all_installed = False
            except subprocess.CalledProcessError as e:
                print(f"!! ERROR: Failed to install '{package}' using pip.")
                print(f"   Command failed: {' '.join(e.cmd)}")
                # You might want to add more detailed error handling or advice here
                all_installed = False
            except Exception as e:
                print(f"!! ERROR: An unexpected error occurred during installation of '{package}': {e}")
                all_installed = False
        else:
            print(f"Package '{package}' is already installed.")
    print("--- Package check complete ---")
    return all_installed

# Run the check before importing the packages
if not check_and_install_packages(REQUIRED_PACKAGES):
    print("\nOne or more required packages could not be installed automatically.")
    print("Please try installing them manually (e.g., using 'pip install gradio yt-dlp')")
    print("Also ensure you have the necessary permissions to install packages.")
    sys.exit(1) # Exit the script if dependencies are missing and couldn't be installed

# --- Now proceed with the rest of the imports and the script ---
print("All required packages are available. Starting the application...")

import gradio as gr
import yt_dlp
import subprocess # Already imported, but fine to list again for clarity
import json
import re
import tempfile
import shutil
from datetime import datetime
import time
import traceback # Import traceback for better error logging

# --- Configuration ---
DOWNLOAD_DIR = "YT_Downloads"
# Ensure download directory exists at script startup
try:
    # Use os.makedirs which handles nested directories and doesn't raise error if exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # Check write permissions more explicitly
    if not os.access(DOWNLOAD_DIR, os.W_OK):
        raise OSError(f"Write permission denied for directory '{DOWNLOAD_DIR}'")
    print(f"Download directory '{DOWNLOAD_DIR}' is ready.")
except OSError as e:
    print(f"!! ERROR creating/accessing download directory '{DOWNLOAD_DIR}': {e}")
    print("   Please check permissions or create the directory manually in a location you can write to.")
    print("   You might need to modify the DOWNLOAD_DIR variable in the script.")
    sys.exit(1) # Exit if the download directory is problematic

# --- Helper Functions ---

def sanitize_filename(filename):
    """Removes or replaces characters invalid for filenames/paths."""
    if not isinstance(filename, str):
        filename = str(filename) # Ensure it's a string
    # Remove invalid characters (Windows & Linux/Mac common subset)
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename) # Replace with underscore
    # Replace colons often found in titles
    sanitized = sanitized.replace(":", "-")
    # Remove leading/trailing whitespace/periods
    sanitized = sanitized.strip(". ")
    # Prevent overly long filenames (adjust max_len as needed)
    max_len = 150
    if len(sanitized) > max_len:
        # Try to cut at the last space before max_len
        cutoff_point = sanitized[:max_len].rfind(' ')
        if cutoff_point != -1 and cutoff_point > max_len / 2: # Avoid cutting too early
             sanitized = sanitized[:cutoff_point]
        else: # No space found or space is too early, just cut hard
             sanitized = sanitized[:max_len]

    # Prevent empty filenames
    if not sanitized:
        sanitized = "downloaded_file"
    return sanitized

def get_video_or_playlist_info(url, first_video_only=False):
    """Fetches info for a URL, detecting if it's a playlist.
       If first_video_only is True and it's a playlist, only fetches full info for the first video.
    """
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'forcejson': True,
        'no_warnings': True,
        'playlistend': 1 if first_video_only else None,
        'ignoreerrors': True, # Important for getting partial playlist info
        # 'extract_flat': 'in_playlist' if first_video_only else False, # Potentially faster but less info
        # 'geo_bypass': True, # May help sometimes, requires careful use
        # 'cookiefile': 'cookies.txt', # If login needed
    }
    print(f"Fetching info for: {url} (first_video_only={first_video_only})")
    try:
        # Consider adding a timeout mechanism if hangs are common
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            if not info_dict: # Check if extract_info returned None (can happen with ignoreerrors)
                return {"error": "Failed to extract info. URL might be invalid, private, or geo-restricted."}
            # Sanitize (less critical here but good practice if passing around)
            # info_dict = ydl.sanitize_info(info_dict)
            return info_dict
    except yt_dlp.utils.DownloadError as e:
        error_message = str(e)
        print(f"DownloadError fetching info ({url}): {error_message}")
        if "is not available in your country" in error_message:
            return {"error": "Video/Playlist is geo-restricted."}
        if "Private video" in error_message or "confirm your age" in error_message:
             return {"error": "Video is private or requires login/age confirmation."}
        if "This playlist does not exist" in error_message:
             return {"error": "Playlist does not exist or is private."}
        if "unable to extract" in error_message.lower():
             return {"error": "Could not extract video data. URL might be wrong or video unavailable."}
        # Catch generic network or format errors
        return {"error": f"DownloadError: Check URL and network connection. ({e})"}
    except Exception as e:
        print(f"An unexpected error occurred during info fetch ({url}):")
        traceback.print_exc() # Log full traceback for unexpected errors
        return {"error": f"Unexpected error during info fetch: {e}"}


def get_format_choices(info_dict):
    """Parses info_dict of a SINGLE video to create dropdown choices."""
    video_formats = []
    audio_formats = []
    subtitles = [] # List of tuples: (Label, lang_code)

    # Add 'best' options first
    video_formats.append(("Best Available Video", "bestvideo"))
    audio_formats.append(("Best Available Audio", "bestaudio"))

    if not info_dict or info_dict.get("error") or 'formats' not in info_dict:
        print("No valid formats found in info_dict, possibly due to error or empty video.")
        return video_formats, audio_formats, subtitles # Return 'best' options anyway

    # --- Video Formats ---
    preferred_vcodec_order = ['avc1', 'av01', 'vp9', 'h264'] # Common codecs
    # Filter for video-only streams with height info
    video_only_formats = [
        f for f in info_dict.get('formats', [])
        if f.get('vcodec') != 'none' and f.get('acodec') == 'none' and f.get('height')
    ]
    sorted_formats = sorted(
        video_only_formats,
        key=lambda f: (
            f.get('height', 0),
            # Prefer codecs in order, put others last
            preferred_vcodec_order.index(f.get('vcodec', '').split('.')[0]) if f.get('vcodec', '').split('.')[0] in preferred_vcodec_order else len(preferred_vcodec_order),
            f.get('fps', 0),
            f.get('tbr', f.get('vbr', 0)) # Use video bitrate if total bitrate missing
        ),
        reverse=True # Highest quality first
    )

    added_video_res = set()
    for f in sorted_formats:
        format_id = f.get('format_id')
        ext = f.get('ext')
        height = f.get('height')
        fps = f.get('fps')
        vcodec_full = f.get('vcodec', 'unknown')
        vcodec = vcodec_full.split('.')[0] # Base codec name (e.g., avc1)
        note = f.get('format_note', '')
        dynamic_range = f.get('dynamic_range', '')
        filesize = f.get('filesize') or f.get('filesize_approx')

        # Create a unique key to avoid duplicate entries for same resolution/codec/fps
        res_key = f"{height}p{fps or ''} ({vcodec})"
        if dynamic_range and dynamic_range != 'SDR': res_key += f" {dynamic_range}"

        # Basic filter for obviously broken entries
        if not format_id or not ext or not height: continue

        if res_key not in added_video_res:
            label = f"{height}p"
            if fps: label += f"{int(fps)}" # Show integer fps
            label += f" ({ext}, {vcodec})"
            if dynamic_range and dynamic_range != 'SDR': label += f" {dynamic_range}"
            # Include note only if useful and short
            if note and note not in label and note.lower() != 'unknown' and len(note) < 20: label += f" [{note}]"
            if filesize: label += f" (~{filesize / (1024 * 1024):.1f} MB)"

            video_formats.append((label, format_id))
            added_video_res.add(res_key)

    # --- Audio Formats ---
    preferred_acodec_order = ['opus', 'aac', 'mp4a', 'vorbis', 'mp3'] # mp4a often refers to AAC
    # Filter for audio-only streams with average bitrate info
    audio_only_formats = [
        f for f in info_dict.get('formats', [])
        if f.get('acodec') != 'none' and f.get('vcodec') == 'none' and f.get('abr')
    ]
    sorted_audio_formats = sorted(
        audio_only_formats,
        key=lambda f: (
            f.get('abr', 0), # Average bitrate
            preferred_acodec_order.index(f.get('acodec', '').split('.')[0]) if f.get('acodec', '').split('.')[0] in preferred_acodec_order else len(preferred_acodec_order),
            f.get('asr', 0) # Audio sampling rate
        ),
        reverse=True # Highest quality first
    )

    added_audio_abr = set()
    for f in sorted_audio_formats:
        format_id = f.get('format_id')
        ext = f.get('ext')
        abr = f.get('abr')
        acodec_full = f.get('acodec', 'unknown')
        acodec = acodec_full.split('.')[0]
        filesize = f.get('filesize') or f.get('filesize_approx')

        # Basic filter
        if not format_id or not ext or not abr: continue

        # Key based on bitrate and codec
        audio_key = f"{int(round(abr))}k ({acodec})"

        if audio_key not in added_audio_abr:
            label = f"{int(round(abr))} kbps ({ext}, {acodec})" # Use rounded int for kbps
            if filesize: label += f" (~{filesize / (1024 * 1024):.1f} MB)"

            audio_formats.append((label, format_id))
            added_audio_abr.add(audio_key)

    # --- Subtitles (Native Only) ---
    subs_info = info_dict.get('subtitles') # Prioritize manually uploaded subs

    if subs_info:
        processed_langs = set()
        for lang, subs_list in subs_info.items():
            if lang in processed_langs: continue # Already processed this language

            best_sub_entry = None
            preferred_sub_fmts = ['vtt', 'srt', 'ass'] # Desired formats

            # Find the first available entry matching preferred formats
            for fmt in preferred_sub_fmts:
                for entry in subs_list:
                     # Ensure entry is a dict and format matches
                     if isinstance(entry, dict) and entry.get('ext') == fmt:
                        best_sub_entry = entry
                        break # Found preferred format for this lang
                if best_sub_entry: break

            # If no preferred format, take the first valid entry (if any)
            if not best_sub_entry:
                for entry in subs_list:
                     if isinstance(entry, dict) and entry.get('ext'): # Check it has an extension
                        best_sub_entry = entry
                        break

            if best_sub_entry:
                # Create label (use language name if available, fallback to code)
                # yt-dlp >= 2023.06.22 uses 'name' field for language description
                lang_name = best_sub_entry.get('name', lang) # 'name' might be 'English', lang is 'en'
                sub_label = f"{lang_name} ({best_sub_entry.get('ext', '?')})"
                subtitles.append((sub_label, lang)) # Store (Label, lang_code)
                processed_langs.add(lang)

    return video_formats, audio_formats, subtitles


def update_choices(url, download_type):
    """Gradio callable to update dropdowns based on URL and type."""
    # Basic URL validation
    if not url or not re.search(r"https?://(www\.)?(youtube\.com/|youtu\.be/)", url):
        # Clear everything and show error message
        return (
            gr.Dropdown(choices=[], value=None, label="Video Quality", interactive=False),
            gr.Dropdown(choices=[], value=None, label="Audio Quality", interactive=False),
            gr.CheckboxGroup(choices=[], value=None, label="Subtitles", interactive=False),
            gr.Textbox(value="Please enter a valid YouTube video or playlist URL."),
            gr.HTML(value=""), # Clear preview
            gr.Accordion(open=False) # Keep accordion closed
        )

    # Determine if we should fetch info for just the first video (faster for playlists)
    is_playlist_likely = 'list=' in url
    # Fetch only first video info if Playlist type is selected *or* if URL seems like a playlist
    fetch_first_only = (download_type == "Playlist") or is_playlist_likely

    info = get_video_or_playlist_info(url, first_video_only=fetch_first_only)

    # Handle errors during info fetching
    if not info or info.get("error"):
         error_msg = info.get("error", "Could not fetch video/playlist information. Check URL, network, or video permissions.")
         return (
            gr.Dropdown(choices=[], value=None, label="Video Quality (Error)", interactive=False),
            gr.Dropdown(choices=[], value=None, label="Audio Quality (Error)", interactive=False),
            gr.CheckboxGroup(choices=[], value=None, label="Subtitles (Error)", interactive=False),
            gr.Textbox(value=f"Error: {error_msg}"),
            gr.HTML(value=""),
            gr.Accordion(open=False)
        )

    is_playlist = info.get('_type') == 'playlist'
    video_info_to_parse = None
    title_html = ""

    # Logic to determine which info dict to parse for formats
    if is_playlist:
        playlist_title = info.get('title', 'Untitled Playlist')
        entries = info.get('entries', [])
        entry_count = info.get('playlist_count', len(entries) if entries else 0)

        # Try to get info from the first valid entry in the list
        first_valid_entry = next((e for e in entries if e and isinstance(e, dict) and not e.get("error")), None)

        if first_valid_entry:
            video_info_to_parse = first_valid_entry
            title_html = f"<p style='font-size: 1.1em; margin-bottom: 5px;'><b>Playlist:</b> {playlist_title} ({entry_count} videos)</p>"
            first_vid_title = video_info_to_parse.get('title', 'Unknown Title')
            title_html += f"<p style='font-size: 0.9em; margin-bottom: 5px; color: #555;'><i>(Showing options based on first video: '{first_vid_title}')</i></p>"
        else:
            # Playlist detected but couldn't get info for the first entry
            title_html = f"<p style='font-size: 1.1em; margin-bottom: 5px;'><b>Playlist:</b> {playlist_title} ({entry_count} videos)</p>"
            title_html += "<p style='color: orange; font-size: 0.9em;'>Warning: Could not load details for the first video. Quality options may be limited (using defaults). Playlist might be empty or contain only private/unavailable videos.</p>"
            video_info_to_parse = {} # Allows get_format_choices to return defaults ('best' options)
    else:
        # Assume it's a single video if _type is not 'playlist'
        video_info_to_parse = info
        video_title = video_info_to_parse.get('title', 'Unknown Title')
        upload_date_str = video_info_to_parse.get('upload_date', '') # YYYYMMDD
        upload_date_formatted = ""
        if upload_date_str:
            try:
                upload_dt = datetime.strptime(upload_date_str, '%Y%m%d')
                upload_date_formatted = f" (Uploaded: {upload_dt.strftime('%Y-%m-%d')})"
            except ValueError: pass # Ignore if date format is unexpected
        title_html = f"<p style='font-size: 1.1em; margin-bottom: 5px;'><b>Video:</b> {video_title}{upload_date_formatted}</p>"


    # Get format choices based on the determined video info
    video_choices, audio_choices, subtitle_choices = get_format_choices(video_info_to_parse)

    # Select 'best' by default if available, otherwise the first real option (index 1)
    default_video = "bestvideo" # Always default to best video
    default_audio = "bestaudio" # Always default to best audio

    # Check if specific formats were actually found beyond 'best'
    has_specific_video = len(video_choices) > 1
    has_specific_audio = len(audio_choices) > 1

    # Prepare labels and interactivity for UI update
    vid_label = "Video Quality" + (" (from first video)" if is_playlist and first_valid_entry else "")
    aud_label = "Audio Quality" + (" (from first video)" if is_playlist and first_valid_entry else "")
    sub_label = f"Subtitles ({len(subtitle_choices)} found, Native Only)" + (" (Applied per video)" if is_playlist else "")

    # Enable dropdowns/checkboxes only if choices were found
    vid_interactive = has_specific_video
    aud_interactive = has_specific_audio
    sub_interactive = len(subtitle_choices) > 0

    # If no specific formats found, add a note to the label?
    if not has_specific_video: vid_label += " (Only 'Best' available)"
    if not has_specific_audio: aud_label += " (Only 'Best' available)"

    return (
        gr.Dropdown(choices=video_choices, value=default_video, label=vid_label, interactive=vid_interactive),
        gr.Dropdown(choices=audio_choices, value=default_audio, label=aud_label, interactive=aud_interactive),
        gr.CheckboxGroup(choices=subtitle_choices, value=None, label=sub_label, interactive=sub_interactive),
        gr.Textbox(value=""), # Clear status textbox on successful fetch
        gr.HTML(value=title_html),
        gr.Accordion(open=True) # Open the accordion on successful fetch
    )


def download_and_convert(url, download_type, video_format_id, audio_format_id, subtitle_langs, progress=gr.Progress()):
    """Downloads, merges, adds metadata, handles subtitles for single video or playlist."""

    # --- Initial Checks ---
    if not url or not re.search(r"https?://(www\.)?(youtube\.com/|youtu\.be/)", url):
        return None, None, "Error: Invalid YouTube URL provided."
    if not video_format_id or not audio_format_id:
        # Handle the case where only 'best' was available and user didn't change it
        if video_format_id != 'bestvideo' or audio_format_id != 'bestaudio':
             return None, None, "Error: Video and Audio Quality selections are missing or invalid. Please fetch options first."
        # Allow proceeding if 'best' is selected (or was the only option)
    if not subtitle_langs: subtitle_langs = [] # Ensure it's a list

    # --- State Variables ---
    status_messages = ["Initiating..."]
    final_output_location = None # Can be file path (single) or directory path (playlist)
    subtitle_files_list = [] # List of final subtitle file paths (single video only)
    temp_dir = None # Path to temporary directory (single video only)
    download_success = False # Flag for overall download success
    moved_video_file = False # Flag for successful move of single video file
    error_log = [] # List to store warning/error messages
    possible_extensions = ['.mkv', '.mp4', '.webm'] # Target video extensions for finding final file

    # --- Progress Hook Definition ---
    # This hook is called by yt-dlp during download/processing
    def custom_progress_hook(d):
        try:
            if d['status'] == 'finished':
                # Log completion of different stages
                filetype = "File"
                filename = d.get('filename', d.get('info_dict', {}).get('filepath', 'N/A'))
                if d.get('postprocessor') == 'Merger':
                     filetype = "Merging"
                elif d.get('info_dict', {}).get('acodec') != 'none' and d.get('info_dict', {}).get('vcodec') == 'none':
                    filetype = "Audio stream"
                elif d.get('info_dict', {}).get('vcodec') != 'none' and d.get('info_dict', {}).get('acodec') == 'none':
                    filetype = "Video stream"
                print(f"Progress Hook: '{filetype}' finished for {os.path.basename(filename)}")

            elif d['status'] == 'downloading':
                percent_str = d.get('_percent_str', '0%').strip()
                speed_str = d.get('_speed_str', '').strip()
                eta_str = d.get('_eta_str', '').strip()
                filename = os.path.basename(d.get('filename', 'download'))

                # Extract percentage
                try: percent = float(percent_str.strip('%')) / 100.0
                except ValueError: percent = 0

                # Playlist progress info
                playlist_idx = d.get('playlist_index')
                playlist_count = d.get('playlist_autonumber') # Preferred over playlist_count for yt-dlp

                # Construct description string
                desc = f"{percent_str} of '{filename}'"
                if speed_str: desc += f" at {speed_str}"
                if eta_str: desc += f" (ETA: {eta_str})"

                # Update Gradio progress bar
                if playlist_idx is not None and playlist_count is not None and playlist_count > 0:
                     # Make sure playlist_idx is 1-based for calculation
                     overall_progress = max(0.0, min(1.0, ((playlist_idx - 1) / playlist_count) + (percent / playlist_count)))
                     desc = f"Playlist {playlist_idx}/{playlist_count}: {desc}"
                     progress(overall_progress, desc=desc)
                else:
                     progress(percent, desc=desc) # Single video progress

            elif d['status'] == 'error':
                print(f"Progress Hook: Received error status for {d.get('filename')}")
                # Error details will be caught by main try/except or return code check
        except Exception as hook_err:
             print(f"Error within progress hook: {hook_err}") # Avoid crashing the download if hook fails


    # --- Main Download Logic ---
    try:
        # --- 1. Prepare yt-dlp Options ---
        status_messages.append("Preparing download options...")
        # Robust format selection string: specific IDs first, then bestvideo/audio, then best overall
        format_selection = f"{video_format_id}+{audio_format_id}/({video_format_id}/bestvideo)+({audio_format_id}/bestaudio)/best"

        ydl_opts = {
            'format': format_selection,
            'merge_output_format': 'mkv',
            'writemetadata': True,
            'writechapters': True,
            'postprocessors': [
                {'key': 'FFmpegMetadata', 'add_metadata': True},
                # {'key': 'FFmpegEmbedSubtitle', 'already_have_subtitle': False}, # Uncomment to embed subs in MKV
            ],
            'subtitleslangs': subtitle_langs,
            'writesubtitles': bool(subtitle_langs),
            'writeautomaticsub': False, # Usually don't want auto-generated subs unless requested
            'subtitlesformat': 'srt/vtt/best',
            'progress_hooks': [custom_progress_hook],
            'ffmpeg_location': shutil.which('ffmpeg'),
            'quiet': False, # Show console output for debugging
            'verbose': False,
            'ignoreerrors': True if download_type == "Playlist" else False,
            'noprogress': True, # Disable yt-dlp's own console progress bar
            'outtmpl': {}, # To be defined based on type
            # 'cookiefile': 'cookies.txt', # For logged-in downloads
            # 'geo_bypass': False, # Use with caution
            'retries': 5, # Retry downloads a few times on transient errors
            'fragment_retries': 5, # Retry fragments (for DASH/HLS)
        }

        # Check for FFmpeg and warn if missing (critical for merging)
        if not ydl_opts['ffmpeg_location']:
            warning_msg = "⚠️ Warning: FFmpeg not found in PATH. Merging video/audio streams and advanced post-processing will fail. You might get separate files or incomplete downloads."
            status_messages.append(warning_msg)
            error_log.append(warning_msg) # Also add to error log

        base_filename_pattern = "downloaded_video" # Default base for file naming

        # --- Configure Output Paths Based on Type ---
        if download_type == "Playlist":
            ydl_opts['noplaylist'] = False
            status_messages.append("Playlist download selected.")
            # Fetch minimal info again for directory naming consistency
            playlist_info = get_video_or_playlist_info(url, first_video_only=True)
            playlist_title = "youtube_playlist"
            playlist_id = "playlist_id"
            if playlist_info and not playlist_info.get("error"):
                playlist_title = sanitize_filename(playlist_info.get('title', playlist_title))
                playlist_id = playlist_info.get('id', playlist_id)

            playlist_dir_name = f"{playlist_title}_{playlist_id}"[:120] # Limit dir name length
            playlist_output_dir = os.path.join(DOWNLOAD_DIR, playlist_dir_name)
            os.makedirs(playlist_output_dir, exist_ok=True)

            # Define output template using playlist variables
            ydl_opts['outtmpl'] = {
                'default': os.path.join(playlist_output_dir, '%(playlist_index)03d - %(title).100s [%(id)s].%(ext)s'), # Pad index, limit title
                'subtitle': os.path.join(playlist_output_dir, '%(playlist_index)03d - %(title).100s [%(id)s].%(lang)s.%(ext)s')
            }
            final_output_location = playlist_output_dir
            status_messages.append(f"Playlist items will be saved to: '{playlist_output_dir}'")

        else: # Single Video
            ydl_opts['noplaylist'] = True
            status_messages.append("Single video download selected.")
            temp_dir = tempfile.mkdtemp(prefix="yt_dlp_")
            print(f"Using temporary directory for single video: {temp_dir}")

            # Fetch detailed info for better base filename
            video_info = get_video_or_playlist_info(url)
            if video_info and not video_info.get("error"):
                 title = sanitize_filename(video_info.get('title', 'youtube_video'))
                 vid_id = video_info.get('id', 'unknown')
                 base_filename_pattern = f"{title}_{vid_id}"
            else:
                 base_filename_pattern = sanitize_filename(f"video_{int(time.time())}")
                 error_log.append("Warning: Could not fetch full video details for optimal naming.")

            # Set temporary output template (yt-dlp uses this for intermediate and final merged file in temp)
            ydl_opts['outtmpl']['default'] = os.path.join(temp_dir, f"{base_filename_pattern}.%(ext)s")
            if subtitle_langs:
                ydl_opts['outtmpl']['subtitle'] = os.path.join(temp_dir, f"{base_filename_pattern}.%(lang)s.%(ext)s")

            status_messages.append(f"Video will be processed in temporary folder: '{temp_dir}'")


        # Update UI before blocking download call
        yield None, None, "\n".join(status_messages)
        progress(0.01, desc="Connecting...")

        # --- 2. Execute Download ---
        status_messages.append("🚀 Starting download & processing...")
        yield None, None, "\n".join(status_messages)

        download_start_time = time.time()
        download_retcode = -1 # Initialize return code
        try:
            # The core download call - blocks until finished or error
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                download_retcode = ydl.download([url]) # Capture return code

            # Check return code after download finishes
            if download_retcode == 0:
                download_success = True # Success if return code is 0
                print("yt-dlp download completed successfully (return code 0).")
            else:
                # Non-zero return code indicates issues, especially relevant for playlists with ignoreerrors=True
                error_msg = f"yt-dlp finished with non-zero return code: {download_retcode}. Some items or steps might have failed."
                error_log.append(error_msg)
                print(error_msg)
                # Decide if this constitutes failure. If any file was potentially created, maybe proceed?
                # Let's assume non-zero means potential failure for single video post-processing.
                if download_type == "Single Video": download_success = False
                # For playlists, partial success might be okay, keep success True if some files downloaded.
                # Check if output dir exists and has files? Too complex here. Assume partial success possible.
                if download_type == "Playlist":
                    # If files were created in the playlist dir, consider it partially successful
                    if final_output_location and os.path.exists(final_output_location) and len(os.listdir(final_output_location)) > 0:
                         download_success = True # Allow partial playlist success if files exist
                         error_log.append("Note: Some playlist items may have failed to download.")
                    else:
                         download_success = False # No files created, consider it a failure


        except yt_dlp.utils.DownloadError as e:
            error_msg = f"DownloadError occurred: {e}"
            error_log.append(error_msg)
            print(error_msg)
            download_success = False
        except Exception as e:
            error_msg = f"An unexpected critical error occurred during download: {e}"
            error_log.append(error_msg)
            print(error_msg)
            traceback.print_exc() # Print full traceback
            download_success = False

        download_duration = time.time() - download_start_time
        status_messages.append(f"Download process finished in {download_duration:.1f} seconds.")

        # Final progress update bar
        progress(1.0, desc="Processing Complete." if download_success and not error_log else "Finished with Errors/Warnings.")

        # --- 3. Post-processing (Single Video Only: Move from Temp to Final Dir) ---
        if download_type == "Single Video":
            if download_success:
                status_messages.append("Checking for final file(s) in temp directory...")
                yield None, None, "\n".join(status_messages)
                time.sleep(0.2) # Tiny pause

                final_video_path = None
                found_video_candidate_name = None
                downloaded_files_in_temp = []
                try:
                     downloaded_files_in_temp = os.listdir(temp_dir)
                     print(f"Files found in temp dir ('{temp_dir}'): {downloaded_files_in_temp}")
                except Exception as list_err:
                     error_log.append(f"Critical Error: Could not list temp directory '{temp_dir}': {list_err}")
                     downloaded_files_in_temp = []
                     download_success = False # Cannot proceed without listing temp

                # --- FIND THE MAIN VIDEO FILE ---
                if download_success:
                    # Look for the expected merged file (.mkv first), then fallbacks
                    for fname in downloaded_files_in_temp:
                        fpath = os.path.join(temp_dir, fname)
                        if os.path.isfile(fpath) and fname.lower().endswith('.mkv'):
                            found_video_candidate_name = fname
                            print(f"Found potential MKV candidate: {fname}")
                            break # Prioritize MKV

                    # If no MKV, look for other common video types
                    if not found_video_candidate_name:
                        for fname in downloaded_files_in_temp:
                            fpath = os.path.join(temp_dir, fname)
                            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in possible_extensions if ext != '.mkv'):
                                found_video_candidate_name = fname
                                print(f"Found alternative video candidate ({os.path.splitext(fname)[1]}): {fname}")
                                break

                    # --- MOVE THE VIDEO FILE ---
                    if found_video_candidate_name:
                        src_path = os.path.join(temp_dir, found_video_candidate_name)
                        dest_fname = found_video_candidate_name # Use the found name for destination
                        potential_final_path = os.path.join(DOWNLOAD_DIR, dest_fname)
                        final_path_to_use = potential_final_path
                        count = 1
                        # Avoid overwriting existing files in final directory
                        while os.path.exists(final_path_to_use):
                            name_part, ext_part = os.path.splitext(dest_fname)
                            final_path_to_use = os.path.join(DOWNLOAD_DIR, f"{name_part}_{count}{ext_part}")
                            count += 1

                        try:
                            status_messages.append(f"Moving '{found_video_candidate_name}' to '{DOWNLOAD_DIR}'...")
                            yield None, None, "\n".join(status_messages)
                            print(f"Attempting to move video '{src_path}' to '{final_path_to_use}'")
                            shutil.move(src_path, final_path_to_use)
                            final_video_path = final_path_to_use
                            final_output_location = final_video_path # Store final path
                            status_messages.append(f"✅ Video saved: {os.path.basename(final_video_path)}")
                            moved_video_file = True
                            print("Video move successful.")
                        except Exception as move_err:
                            err_msg = f"Failed to move final video file '{found_video_candidate_name}': {move_err}"
                            error_log.append(err_msg)
                            print(f"Error moving video: {err_msg}")
                            moved_video_file = False
                    else:
                        # This is a critical failure for single video mode if download reported success
                        err_msg = f"Critical Error: Download reported success, but no final video file (.mkv, .mp4, .webm) found in '{temp_dir}'"
                        error_log.append(err_msg)
                        print(err_msg)
                        print(f"Files present were: {downloaded_files_in_temp}")
                        download_success = False # Overrule success if file is missing

                # --- MOVE SUBTITLES (Only if video moved successfully) ---
                if moved_video_file and final_video_path and subtitle_langs:
                    status_messages.append("Moving subtitle file(s)...")
                    yield None, None, "\n".join(status_messages)

                    final_video_name_no_ext = os.path.splitext(os.path.basename(final_video_path))[0]
                    current_files_in_temp = os.listdir(temp_dir) # Refresh list

                    for fname in current_files_in_temp:
                        src_sub_path = os.path.join(temp_dir, fname)
                        if not os.path.isfile(src_sub_path): continue

                        # Identify subtitle files based on naming convention (lang code + common ext)
                        is_subtitle = False; file_lang = None; file_sub_ext = None
                        for lang_code in subtitle_langs:
                             # Allow flexibility in separator (. or -) before lang code
                             match = re.search(rf'[._-]{re.escape(lang_code)}\.(srt|vtt|ass)$', fname.lower())
                             if match:
                                 is_subtitle = True; file_lang = lang_code; file_sub_ext = match.group(1)
                                 break

                        if is_subtitle:
                            try:
                                # Construct final sub name based on the moved video's name
                                final_sub_name = f"{final_video_name_no_ext}.{file_lang}.{file_sub_ext}"
                                final_sub_path = os.path.join(DOWNLOAD_DIR, final_sub_name)
                                count = 1
                                while os.path.exists(final_sub_path): # Avoid overwrite
                                    final_sub_path = os.path.join(DOWNLOAD_DIR, f"{final_video_name_no_ext}_{count}.{file_lang}.{file_sub_ext}")
                                    count += 1

                                print(f"Attempting to move subtitle '{src_sub_path}' to '{final_sub_path}'")
                                shutil.move(src_sub_path, final_sub_path)
                                subtitle_files_list.append(final_sub_path) # Store final path
                                status_messages.append(f"↪ Subtitle saved: {os.path.basename(final_sub_path)}")
                                print("Subtitle move successful.")
                            except Exception as sub_move_err:
                                err_msg = f"Failed to move subtitle file '{fname}': {sub_move_err}"
                                error_log.append(err_msg)
                                print(f"Error moving subtitle: {err_msg}")
                elif not moved_video_file and subtitle_langs:
                     if download_success: # Only log if download didn't already fail
                         error_log.append("Skipping subtitle move because the main video file was not successfully moved.")

            elif not download_success:
                status_messages.append("Skipping file moving: Download/processing failed earlier.")

        # --- 4. Final Status Assembly ---
        final_status_message = ""
        if error_log:
            final_status_message = "Process finished with errors or warnings:\n- " + "\n- ".join(error_log)
            if download_success and download_type == "Single Video" and not moved_video_file:
                 # This indicates a problem finding/moving the file after download
                 final_status_message += "\n\nCRITICAL: Main video file move failed after download."
            elif download_type == "Playlist" and download_success: # If playlist finished with errors but files exist
                 final_status_message += f"\n\nPlaylist download to '{final_output_location}' may be incomplete. Check the folder."
            elif download_type == "Playlist" and not download_success: # If playlist failed outright
                 final_status_message += f"\n\nPlaylist download to '{final_output_location}' failed. No files may have been saved."
            final_status_message = "⚠️ " + final_status_message # Add warning icon

        # Check overall success for final message
        # Overall success = (Single Video success AND video moved) OR (Playlist success)
        overall_success = (download_type == "Single Video" and download_success and moved_video_file) or \
                          (download_type == "Playlist" and download_success)


        if overall_success:
             if download_type == "Playlist":
                  completion_note = "(possibly with skipped items if errors occurred)" if error_log else "successfully"
                  final_status_message = f"✅ Playlist download completed {completion_note}.\nFiles are in: '{final_output_location}'"
             elif download_type == "Single Video": # Implicitly moved_video_file is True here
                  final_status_message = f"✅ Single video download completed successfully.\nVideo: {os.path.basename(final_output_location)}"
                  if subtitle_files_list:
                       final_status_message += "\nSubtitles:\n- " + "\n- ".join([os.path.basename(s) for s in subtitle_files_list])

        elif not error_log: # If download failed but no specific error was logged
             final_status_message = "❌ Download process failed for an unknown reason. Check console output."
        # If !overall_success and error_log exists, the final_status_message already contains the errors/warnings


        # --- 5. Prepare Return Values ---
        # Add extra print for debugging the return values specifically
        print("-" * 20)
        print(f"Returning to Gradio:")
        print(f"  Video File Output: {final_output_location if (download_type == 'Single Video' and moved_video_file) else None}")
        print(f"  Subtitle Text Output: {' | '.join(subtitle_files_list) if subtitle_files_list else 'None'}")
        print(f"  Status Message: {final_status_message[:100]}...") # Show start of status
        print("-" * 20)

        if download_type == "Single Video" and moved_video_file:
             # Return file path for video, newline-separated string for subs, and status message
             return final_output_location, "\n".join(subtitle_files_list), final_status_message
        else:
             # Playlist success/failure or Single video failure: return None for file paths
             return None, None, final_status_message

    except Exception as e:
        # --- Catchall Exception Handler ---
        print("--- CRITICAL Unhandled Exception in download_and_convert ---")
        traceback.print_exc()
        final_status_message = f"❌ A critical unexpected error occurred: {e}\nPlease check the console output for details."
        # Attempt to return error state to Gradio
        return None, None, final_status_message
    finally:
        # --- Cleanup ---
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temp directory '{temp_dir}': {e}")
                # Optionally add warning to status? Might be too late.


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.Markdown(
        """
        # 🎬 YouTube Video/Playlist Downloader
        Enter a YouTube video or playlist URL, select desired quality and subtitles, then click Download.
        Requires `yt-dlp` and `ffmpeg`.
        """
    )
    # Use os.path.abspath to ensure the allowed_path matches exactly later
    abs_download_dir = os.path.abspath(DOWNLOAD_DIR)
    gr.Markdown(f"**💾 Files saved to:** `{abs_download_dir}`")

    with gr.Row():
        with gr.Column(scale=3):
            url_input = gr.Textbox(
                label="YouTube URL (Video or Playlist)",
                placeholder="e.g., https://www.youtube.com/watch?v=...",
                elem_id="youtube-url-input"
            )
        with gr.Column(scale=1, min_width=180): # Ensure radio buttons have enough space
             download_type_radio = gr.Radio(
                ["Single Video", "Playlist"], value="Single Video", label="Download Type", info="Select 'Playlist' if URL contains 'list='"
            )

    fetch_button = gr.Button("① Fetch Video/Playlist Options", variant="secondary")

    # Use Accordion for options, start closed
    with gr.Accordion("② Quality & Subtitle Options (Click to Expand)", open=False) as options_accordion:
        title_html_output = gr.HTML(value="<p style='color:#777'><i>Video/Playlist info will appear here after fetching...</i></p>") # Placeholder
        with gr.Row():
             video_quality_dropdown = gr.Dropdown(label="Video Quality", choices=[], interactive=False, info="Requires merging w/ audio.")
             audio_quality_dropdown = gr.Dropdown(label="Audio Quality", choices=[], interactive=False, info="Requires merging w/ video.")
        subtitle_checkboxgroup = gr.CheckboxGroup(label="Subtitles (Native Only, if available)", choices=[], interactive=False)

    download_button = gr.Button("③ Start Download", variant="primary", interactive=True) # Start enabled

    gr.Markdown("---") # Separator

    # Status and Output Area
    with gr.Accordion("④ Status & Output", open=True):
        status_textbox = gr.Textbox(
            label="Status / Log", lines=7, interactive=False,
            placeholder="Progress and status messages will appear here..."
            )
        with gr.Row():
            output_video_file = gr.File(label="Downloaded Video File (Single Video Only)", interactive=False)
            output_subtitle_files = gr.Textbox(label="Downloaded Subtitle Files (Single Video Only)", lines=2, interactive=False)

    # --- Event Listeners ---

    # 1. Fetch Button Click Logic
    fetch_button.click(
        fn=update_choices,
        inputs=[url_input, download_type_radio],
        outputs=[
            video_quality_dropdown,
            audio_quality_dropdown,
            subtitle_checkboxgroup,
            status_textbox, # Display fetch errors here
            title_html_output,
            options_accordion # Control accordion state
        ],
        api_name="fetch_options"
    ).then(
        # Clear previous output files, update status, re-enable download button
        lambda: (None, "", "Ready. Fetch options or start download.", gr.Button.update(interactive=True)), # CORRECT way to update button state
        outputs=[output_video_file, output_subtitle_files, status_textbox, download_button] # Update download_button state
    )

    # 2. Download Button Click Logic
    download_button.click(
        fn=lambda: (gr.Button.update(interactive=False), gr.Textbox.update(value="Starting download...")), # Disable btn, update status immediately
        outputs=[download_button, status_textbox]
    ).then(
        fn=download_and_convert,
        inputs=[
            url_input,
            download_type_radio,
            video_quality_dropdown,
            audio_quality_dropdown,
            subtitle_checkboxgroup,
        ],
        outputs=[output_video_file, output_subtitle_files, status_textbox],
        api_name="download_video"
    ).then(
         fn=lambda: gr.Button.update(interactive=True), # Re-enable button after download attempt
         outputs=[download_button]
    )


# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("-" * 50)
    # Check and print FFmpeg status
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
    else:
        print("❌ Warning: FFmpeg executable not found in system PATH.")
        print("   Merging video/audio and some post-processing features WILL FAIL.")
        print("   Please install FFmpeg and ensure it's added to your PATH environment variable.")

    # Verify download directory write permissions (already checked after creation, but good to have here too)
    abs_download_dir = os.path.abspath(DOWNLOAD_DIR) # Ensure we use the absolute path
    if not os.path.isdir(abs_download_dir) or not os.access(abs_download_dir, os.W_OK):
         print(f"❌ Error: Download directory '{abs_download_dir}' is not accessible or writable.")
         print("   The script might have failed to create it, or permissions are incorrect.")
         sys.exit(1)

    print(f"💾 Downloads will be saved to: {abs_download_dir}")
    print(f"💡 Gradio UI File Access: Allowing access to '{abs_download_dir}' for file display.")
    print("-" * 50)
    print("🚀 Launching Gradio UI...")
    print("   You can access it at the Local URL or the Public URL (if generated).")
    print("   Press CTRL+C in this terminal to stop the server.")
    print("-" * 50)

    # Launch with share=True and allow access to the download directory
    demo.launch(
        share=True,
        debug=False, # Set to True for more Gradio console logging if needed
        allowed_paths=[abs_download_dir] # Crucial for gr.File to work with YT_Downloads
        )

    print("Gradio App closed.")
