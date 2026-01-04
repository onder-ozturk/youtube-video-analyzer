# YouTube Video Comprehensive Analyzer

A Jupyter notebook that builds a playlist dataset, fetches YouTube transcripts,
summarizes and translates them, and optionally clusters video themes.

## What this notebook does
- Pulls all videos from a YouTube playlist via the YouTube Data API
- Saves a CSV with `title` and `url`
- Extracts transcripts using `youtube-transcript-api`
- Summarizes transcripts (OpenAI optional; fallback summarizer if no key)
- Translates summaries/transcripts to English
- Exports results to CSV
- Optional: embeds and clusters summaries with SentenceTransformers + UMAP +
  KMeans/HDBSCAN/SOM

## Inputs
- Playlist ID (example: `PLnAF4npbrTwzO2W6v07ktEjBU6885k5hB`)
- CSV input with video URLs, e.g. `storybox_videos_utf8_bom.csv`
  - Required column: `url`
  - Optional column: `title`
- Optional clustering input: `Helsinki_Opus_Transcript.xlsx`
  - Expected columns: `Video URL`, `Transcript`, `Summary`

## Outputs
- `storybox_videos_utf8_bom.csv` (and other encoding variants)
- `analyzed_storybox_videos.csv` / `analyzed_storybox_videos_v2.csv`
  - Columns include: `Video_URL`, `Video_Title`, `Transcript`, `Summary`,
    `Summary_English`, `Transcript_English`, `Status`
- Optional clustering results, e.g. `kmeans_labeled_llm_improved.xlsx`

## Setup
Create a Python environment and install dependencies. You can install only the
packages you need based on the sections you plan to run:

```bash
# Playlist extraction + transcript analysis
pip install google-api-python-client youtube-transcript-api pandas requests nltk
pip install googletrans==4.0.0rc1 deep-translator openai

# Optional clustering / labeling
pip install sentence-transformers transformers torch umap-learn hdbscan minisom
pip install scikit-learn matplotlib seaborn openpyxl
```

## API keys and configuration
This notebook requires a YouTube Data API key, and optionally an OpenAI API key
for higher quality summaries.

Do NOT commit real API keys to GitHub. Replace them with placeholders or load
them from environment variables.

```bash
export YOUTUBE_API_KEY="your_youtube_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # optional
```

Then use these variables inside the notebook instead of hard-coded strings.

## Usage (Jupyter / Colab)
1. Open `YouTube Video Comprehensive Analyzer.ipynb`.
2. Run the playlist section to generate a CSV of video URLs.
3. Run the analyzer section to fetch transcripts and create summaries.
4. (Optional) Run the clustering section to group videos by topic.

## Notes
- Not all videos have transcripts available. The analyzer handles missing
  transcripts and records a status per video.
- YouTube and translation services have rate limits; the notebook includes
  basic delays but you may need to slow down further for large playlists.
- Respect YouTube Terms of Service and copyright when sharing transcripts or
  derived data.

## Suggested repo contents
If you want a clean GitHub repo, include only:
- `YouTube Video Comprehensive Analyzer.ipynb`
- `README.md`
- (Optional) a small sample CSV with a few URLs

Avoid committing large result files or any API keys.
