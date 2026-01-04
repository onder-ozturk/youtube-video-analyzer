# YouTube Video Comprehensive Analyzer - Datasets Documentation
# YouTube Video Kapsamlı Analiz - Veri Setleri Dokümantasyonu

## Overview / Genel Bakış

This document describes all datasets, data sources, and data processing pipelines used in the **YouTube Video Comprehensive Analyzer** notebook.

Bu doküman, **YouTube Video Kapsamlı Analiz** notebook'unda kullanılan tüm veri setlerini, veri kaynaklarını ve veri işleme süreçlerini detaylı olarak açıklar.

---

## Table of Contents / İçindekiler

1. [Input Datasets / Giriş Veri Setleri](#input-datasets--giriş-veri-setleri)
2. [API Data Sources / API Veri Kaynakları](#api-data-sources--api-veri-kaynakları)
3. [Output Datasets / Çıktı Veri Setleri](#output-datasets--çıktı-veri-setleri)
4. [Data Processing Pipeline / Veri İşleme Süreci](#data-processing-pipeline--veri-i̇şleme-süreci)
5. [Machine Learning Models / Makine Öğrenmesi Modelleri](#machine-learning-models--makine-öğrenmesi-modelleri)
6. [Dependencies & Requirements / Bağımlılıklar ve Gereksinimler](#dependencies--requirements--bağımlılıklar-ve-gereksinimler)
7. [Data Flow Diagram / Veri Akış Diyagramı](#data-flow-diagram--veri-akış-diyagramı)
8. [Configuration Parameters / Konfigürasyon Parametreleri](#configuration-parameters--konfigürasyon-parametreleri)
9. [Troubleshooting / Sorun Giderme](#troubleshooting--sorun-giderme)

---

## Input Datasets / Giriş Veri Setleri

### 1. YouTube Playlist Data (StoryBox)

**Source / Kaynak**: YouTube Data API v3
**Format**: CSV files with multiple encoding options / Çoklu encoding seçenekleriyle CSV dosyaları

#### Input Files / Giriş Dosyaları:
- `storybox_videos_utf8_bom.csv` - **Primary input file** (UTF-8 with BOM for Excel compatibility)
  - **Excel için en uygun** (UTF-8 with BOM)
  - Türkçe karakterler için otomatik tanıma

- `storybox_videos_utf8.csv` - Pure UTF-8 encoding
  - Saf UTF-8 formatı
  - Genel programlama kullanımı için

- `storybox_videos_win1254.csv` - Windows Turkish encoding (Windows-1254)
  - Windows Türkçe sistemi için
  - Eski Windows uyumluluğu

- `storybox_videos_manual.csv` - Manual encoding control
  - Manuel encoding kontrolü

#### Content Structure / Veri Yapısı:
```csv
title,url
"Video Title","https://www.youtube.com/watch?v=VIDEO_ID"
"Video Başlığı","https://www.youtube.com/watch?v=VIDEO_ID"
```

**Fields / Alanlar**:
- `title`: Video başlığı (Turkish characters supported / Türkçe karakter destekli)
- `url`: YouTube video URL
- Internally extracted: `video_id` (otomatik çıkarılır)

**Purpose / Amaç**:
- EN: Contains the initial list of YouTube videos from a specific playlist to be analyzed for transcripts, themes, and sentiment.
- TR: Belirli bir YouTube oynatma listesinden alınan videoların başlangıç listesini içerir. Bu videolar transkript, tema ve duygu analizi için kullanılır.

**Source Details / Kaynak Detayları**:
- API Endpoint: `youtube.playlistItems().list()`
- Max results per request: 50 items
- Pagination: Uses `nextPageToken`
- Authentication: Requires `YOUTUBE_API_KEY` environment variable

---

### 2. Helsinki Opus Transcript Dataset

**File / Dosya**: `Helsinki_Opus_Transcript.xlsx`
**Format**: Microsoft Excel (.xlsx)
**Size**: Variable depending on playlist content / Playlist içeriğine göre değişken

#### Content Structure / Veri Yapısı:
The Excel file contains video transcripts with the following columns:

**Expected Columns / Beklenen Sütunlar**:
- `title` or `video_title`: Video başlığı
- `transcript` or `text`: Video transkript metni (full transcript text)
- `url` or `video_url`: YouTube video linki
- `summary` (optional): Video özeti
- Additional metadata fields (optional)

**Purpose / Amaç**:
- EN: Main dataset for theme clustering and content analysis. Contains full transcripts extracted from YouTube videos.
- TR: Tema kümeleme ve içerik analizi için ana veri seti. YouTube videolarından çıkarılan tam transkriptleri içerir.

**Usage in Notebook / Notebook'ta Kullanım**:
- Cell 10-14: Clustering analysis (kümeleme analizi)
- Default file path: `"Helsinki_Opus_Transcript.xlsx"`
- Loaded using: `pd.read_excel(file_path)`
- Encoding: UTF-8 by default

**Data Quality Requirements / Veri Kalitesi Gereksinimleri**:
- Minimum transcript length: 50 characters (çok kısa transkriptler filtrelenir)
- No duplicate video URLs (duplicate kontrol yapılır)
- Valid UTF-8 encoding (encoding doğrulaması)

---

## API Data Sources / API Veri Kaynakları

### 1. YouTube Data API v3

**Service**: Google YouTube Data API
**Authentication**: API Key required (API anahtarı gerekli)
**Environment Variable**: `YOUTUBE_API_KEY`

#### API Endpoints Used / Kullanılan API Endpoint'leri:
```python
youtube.playlistItems().list(
    part='snippet,contentDetails',
    playlistId=playlist_id,
    maxResults=50,
    pageToken=next_page_token
)
```

**Returned Data / Dönen Veri**:
- `videoId`: Video kimliği
- `title`: Video başlığı
- `snippet`: Video meta bilgileri
- `contentDetails`: İçerik detayları
- `publishedAt`: Yayınlanma tarihi

**Features / Özellikler**:
- Fetch playlist information (playlist bilgilerini çekme)
- Retrieve video metadata (video metadata'sını alma)
- Pagination support with `nextPageToken`
- Rate limiting awareness (quota yönetimi)

**API Limits / API Limitleri**:
- Daily quota: 10,000 units (günlük kota: 10,000 birim)
- playlistItems.list call: 1 unit per request

**Purpose / Amaç**: Initial data collection from YouTube playlists to create the video list CSV files.

**Documentation**: [https://developers.google.com/youtube/v3](https://developers.google.com/youtube/v3)

---

### 2. YouTube Transcript API

**Package**: `youtube-transcript-api`
**Installation**: `pip install youtube-transcript-api`
**GitHub**: [https://github.com/jdepoix/youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)

#### API Methods / API Metodları:
```python
from youtube_transcript_api import YouTubeTranscriptApi

# Get transcript with language preference
transcript = YouTubeTranscriptApi.get_transcript(
    video_id,
    languages=['tr', 'en']  # Turkish first, then English
)
```

**Output Format / Çıktı Formatı**:
```python
[
    {
        'text': 'Transcript text',
        'start': 0.0,
        'duration': 3.5
    },
    ...
]
```

**Features / Özellikler**:
- Extract video transcripts/captions (altyazı çıkarma)
- Automatic language detection (otomatik dil tespiti)
- Multi-language support: Turkish, English, etc. (çoklu dil desteği)
- Fallback to auto-generated captions (otomatik altyazılara geri dönüş)
- Text formatting support (metin formatlama)
- Timestamp information (zaman damgası bilgisi)

**Limitations / Limitasyonlar**:
- Not all videos have transcripts (tüm videolarda transkript bulunmayabilir)
- Some videos block transcript access (bazı videolar erişimi engelleyebilir)
- Auto-generated captions may have errors (otomatik altyazılar hatalı olabilir)

**Purpose / Amaç**: Extract text transcripts from YouTube videos for content analysis.

---

### 3. OpenAI API

**Package**: `openai>=1.0.0`
**Models Used**: GPT-3.5-turbo / GPT-4 (configurable)
**Authentication**: OpenAI API key required

#### Use Cases / Kullanım Alanları:

**1. Video Summarization / Video Özetleme**:
```python
from openai import OpenAI

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a video summarizer."},
        {"role": "user", "content": f"Summarize: {transcript}"}
    ],
    max_tokens=200,
    temperature=0.5
)
summary = response.choices[0].message.content
```

**2. Sentiment Analysis / Duygu Analizi**:
- Analyze emotional tone of content (içeriğin duygusal tonunu analiz etme)
- Classify as positive/negative/neutral (pozitif/negatif/nötr sınıflandırma)
- Extract key emotions (anahtar duyguları çıkarma)

**3. Theme Extraction / Tema Çıkarma**:
- Identify main topics and themes (ana konuları ve temaları belirleme)
- Categorize content (içerik kategorilendirme)
- Extract key points (anahtar noktaları çıkarma)

**4. Cluster Labeling (LLM-based) / Küme Etiketleme**:
```python
# Sample texts from each cluster
sample_texts = cluster_df.sample(5)['transcript'].tolist()

# Generate cluster label with LLM
prompt = f"""
Based on the following video summaries, define the main theme
of this group in 2-3 words:

{chr(10).join(sample_texts)}

Label (only 2-3 words):
"""
label = get_llm_response(prompt)
```

**API Parameters / API Parametreleri**:
- `model`: "gpt-3.5-turbo" or "gpt-4"
- `max_tokens`: 200-2000 (depending on task)
- `temperature`: 0.3-0.7 (lower = more consistent)
- `top_p`: 1.0 (nucleus sampling)

**Purpose / Amaç**: AI-powered content analysis and enhancement of clustering results.

---

### 4. Translation API

**Package**: `deep-translator` (preferred) / `googletrans` (fallback)
**Installation**:
```bash
pip install deep-translator
pip install googletrans==4.0.0rc1
```

#### Deep Translator (Preferred / Tercih Edilen):
```python
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='tr', target='en')
translated_text = translator.translate(text)
```

#### Googletrans (Fallback / Yedek):
```python
from googletrans import Translator

translator = Translator()
result = translator.translate(text, src='tr', dest='en')
translated_text = result.text
```

**Supported Languages / Desteklenen Diller**:
- Turkish ↔ English (Türkçe ↔ İngilizce)
- Automatic language detection (otomatik dil tespiti)
- 100+ languages supported

**Usage Scenarios / Kullanım Senaryoları**:
- Translate Turkish transcripts to English (Türkçe transkriptleri İngilizce'ye çevirme)
- Multi-language normalization for clustering (kümeleme için çok dilli normalizasyon)
- Standardize summary texts (özet metinlerini standartlaştırma)

**Purpose / Amaç**: Enable multi-language analysis and ensure compatibility with English-based NLP models.

---

## Output Datasets / Çıktı Veri Setleri

### 1. Analyzed Video CSV Files

#### Files / Dosyalar:
- `analyzed_storybox_videos.csv` - First version of analyzed results
  - İlk versiyon analiz sonuçları

- `analyzed_storybox_videos_v2.csv` - Improved version with enhanced analysis
  - İyileştirilmiş analiz ile gelişmiş versiyon

- `temp_analyzed_*.csv` - Temporary checkpoint files during processing
  - İşlem sırasında oluşturulan geçici checkpoint dosyaları

#### Output Structure / Çıktı Yapısı:
```csv
title,url,video_id,transcript,summary,sentiment,theme,analysis_date,status
```

**Fields / Alanlar**:
- `title`: Original video title (orijinal video başlığı)
- `url`: YouTube video URL
- `video_id`: YouTube video identifier
- `transcript`: Full transcript text (tam transkript metni)
- `transcript_en`: English transcript (optional)
- `summary`: AI-generated summary (AI tarafından oluşturulan özet)
- `summary_en`: English summary (optional)
- `sentiment`: Sentiment score or category (duygu skoru veya kategorisi)
  - Values: "Positive", "Negative", "Neutral"
- `theme`: Identified theme/category (belirlenen tema/kategori)
- `analysis_date`: Processing timestamp (işlem zaman damgası)
- `status`: Processing status (işlem durumu)
  - Values: "Success", "Error", "No Transcript"

**Encoding**: UTF-8 with BOM (`utf-8-sig`) for Excel compatibility

**Purpose / Amaç**: Store analyzed video data with transcripts, summaries, and AI-generated insights.

---

### 2. Clustering Analysis Results (Excel)

#### K-Means Clustering Files:
- `kmeans_labeled_llm.xlsx` - Default K-Means clustering (4 clusters)
- `kmeans_3_labeled_llm.xlsx` - K-Means with 3 clusters
- `kmeans_4_labeled_llm.xlsx` - K-Means with 4 clusters
- `kmeans_5_labeled_llm.xlsx` - K-Means with 5 clusters
- `kmeans_labeled_llm_improved.xlsx` - Enhanced version with improved labeling

#### HDBSCAN Clustering Files:
- `hdbscan_min3_labeled_llm.xlsx` - HDBSCAN with min_cluster_size=3
- `hdbscan_min5_labeled_llm.xlsx` - HDBSCAN with min_cluster_size=5
- `hdbscan_min7_labeled_llm.xlsx` - HDBSCAN with min_cluster_size=7

#### SOM (Self-Organizing Maps) Files:
- `som_4_labeled_llm.xlsx` - SOM with ~4 clusters (2×2 grid)
- `som_6_labeled_llm.xlsx` - SOM with ~6 clusters (2×3 grid)
- `som_9_labeled_llm.xlsx` - SOM with ~9 clusters (3×3 grid)

#### Output Structure / Çıktı Yapısı:

**Common Columns / Ortak Sütunlar**:
- Original transcript data (orijinal transkript verisi)
- `cluster`: Cluster assignment (küme ataması) - numeric ID
- `cluster_label`: LLM-generated cluster name (LLM tarafından oluşturulan küme adı)
- `cluster_theme`: Detailed theme description (detaylı tema açıklaması)
- `keywords`: Representative keywords for the cluster (küme için temsili anahtar kelimeler)
- `umap_x`, `umap_y`: 2D coordinates for visualization (görselleştirme için 2D koordinatlar)

**Method-Specific Columns / Metoda Özel Sütunlar**:

**K-Means**:
- `silhouette_score`: Clustering quality metric (file-level)
- `inertia`: Within-cluster sum of squares

**HDBSCAN**:
- `outlier_score`: Noise/outlier score (gürültü skoru)
- `cluster == -1`: Noise points (gürültü noktaları)
- `probabilities`: Cluster membership probabilities

**SOM**:
- `som_x`, `som_y`: Grid coordinates (grid koordinatları)
- `winner_neuron`: Winning neuron ID (kazanan nöron kimliği)

**Purpose / Amaç**: Store clustering results with LLM-generated labels and quality metrics for analysis and visualization.

---

## Data Processing Pipeline / Veri İşleme Süreci

### Stage 1: Data Collection / Veri Toplama

```
YouTube Playlist
    ↓ (YouTube Data API v3)
CSV Export (multiple encodings)
    ├── storybox_videos_utf8_bom.csv (Primary)
    ├── storybox_videos_utf8.csv
    ├── storybox_videos_win1254.csv
    └── storybox_videos_manual.csv
```

**Process / Süreç**:
1. Connect to YouTube Data API (YouTube Data API'ye bağlan)
2. Fetch playlist items with pagination (sayfalandırma ile playlist öğelerini çek)
3. Extract video metadata: title, ID, URL (video metadata'sını çıkar)
4. Export to CSV with multiple encoding formats (çoklu encoding formatıyla CSV'ye aktar)
5. Turkish character handling and validation (Türkçe karakter işleme ve doğrulama)

**Code Example / Kod Örneği**:
```python
from googleapiclient.discovery import build
import csv

API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

videos = []
next_page_token = None

while True:
    request = youtube.playlistItems().list(
        part='snippet',
        playlistId=playlist_id,
        maxResults=50,
        pageToken=next_page_token
    )
    response = request.execute()

    for item in response['items']:
        video_id = item['snippet']['resourceId']['videoId']
        title = item['snippet']['title']
        videos.append({
            'title': title,
            'url': f'https://www.youtube.com/watch?v={video_id}'
        })

    next_page_token = response.get('nextPageToken')
    if not next_page_token:
        break

# Save with UTF-8 BOM for Excel
with open('storybox_videos_utf8_bom.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['title', 'url'])
    writer.writeheader()
    writer.writerows(videos)
```

---

### Stage 2: Transcript Extraction & Analysis / Transkript Çıkarma ve Analiz

```
Video URLs
    ↓ (YouTube Transcript API)
Raw Transcripts
    ↓ (Text Cleaning & Preprocessing)
Cleaned Transcripts
    ↓ (OpenAI API - Optional)
Analyzed Content (Summary, Sentiment, Theme)
    ↓
analyzed_storybox_videos.csv
```

**Process / Süreç**:
1. Read video list from CSV (CSV'den video listesini oku)
2. For each video:
   - Extract transcript using YouTube Transcript API (transkripti çıkar)
   - Try languages: Turkish → English → Auto-generated (dilleri dene)
   - Clean and normalize text (metni temizle ve normalize et)
   - (Optional) Generate AI summary (AI özet oluştur)
   - (Optional) Perform sentiment analysis (duygu analizi yap)
   - (Optional) Extract themes (temaları çıkar)
3. Save results with checkpointing (resumable) (checkpoint ile kaydet)
4. Export to analyzed CSV (analiz edilmiş CSV'ye aktar)

**Error Handling / Hata Yönetimi**:
- No transcript available: `status = "No Transcript"`
- API error: `status = "Error"`
- Successful: `status = "Success"`

**Code Example / Kod Örneği**:
```python
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd

df = pd.read_csv('storybox_videos_utf8_bom.csv', encoding='utf-8-sig')
results = []

for index, row in df.iterrows():
    video_url = row['url']
    video_id = video_url.split('v=')[1]

    try:
        # Extract transcript
        transcript_data = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['tr', 'en']
        )
        transcript = ' '.join([item['text'] for item in transcript_data])

        # Optional: Summarize with OpenAI
        summary = summarize_with_openai(transcript)

        results.append({
            'title': row['title'],
            'url': video_url,
            'transcript': transcript,
            'summary': summary,
            'status': 'Success'
        })
    except Exception as e:
        results.append({
            'title': row['title'],
            'url': video_url,
            'status': f'Error: {str(e)}'
        })

    # Save checkpoint every 10 videos
    if (index + 1) % 10 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv('temp_analyzed.csv', encoding='utf-8-sig', index=False)

# Final save
final_df = pd.DataFrame(results)
final_df.to_csv('analyzed_storybox_videos.csv', encoding='utf-8-sig', index=False)
```

---

### Stage 3: Feature Engineering / Özellik Mühendisliği

```
Transcripts
    ↓ (Sentence-BERT)
Text Embeddings (384-dimensional)
    ↓ (UMAP)
Reduced Embeddings (2-dimensional)
    ↓ (Clustering Algorithm)
Cluster Assignments
```

#### 1. Text Embeddings / Metin Gömme
**Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
**Embedding Dimension**: 384
**Purpose**: Convert text to numerical vectors (metni sayısal vektörlere dönüştür)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)
# Output: numpy array (n_samples, 384)
```

#### 2. Dimensionality Reduction / Boyut İndirgeme
**Algorithm**: UMAP (Uniform Manifold Approximation and Projection)
**Output Dimensions**: 2 (for visualization / görselleştirme için)
**Parameters**: `n_components=2`, `random_state=42`

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
reduced_embeddings = reducer.fit_transform(embeddings)
# Output: numpy array (n_samples, 2)
```

#### 3. Text Preprocessing / Metin Ön İşleme

**Stopword Removal / Stopword Çıkarma**:
```python
TURKISH_STOPWORDS = {
    've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'gibi', 'olarak',
    'sonra', 'önce', 'yılında', 'artık', 'çok', 'tüm', 'her', 'ise',
    'daha', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'şu', 'şey',
    'var', 'yok', 'mi', 'mu', 'mü', 'den', 'dan', 'ten', 'tan'
}
```

**Text Normalization / Metin Normalizasyonu**:
- Lowercase conversion (küçük harfe çevirme)
- Whitespace trimming (boşluk temizleme)
- Special character handling (özel karakter işleme)

**TF-IDF Vectorization / TF-IDF Vektörleştirme**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words=list(TURKISH_STOPWORDS),
    max_features=100,
    ngram_range=(1, 2),  # unigram and bigram
    min_df=2  # Must appear in at least 2 documents
)
```

---

### Stage 4: Clustering Analysis / Kümeleme Analizi

```
Reduced Embeddings
    ↓ (Clustering)
    ├── K-Means (3, 4, 5 clusters)
    ├── HDBSCAN (min_size: 3, 5, 7)
    └── SOM (4, 6, 9 clusters)
    ↓ (Cluster Quality Evaluation)
Silhouette Scores
    ↓ (LLM-based Labeling)
Cluster Labels & Themes
    ↓ (Visualization)
UMAP Scatter Plots
    ↓ (Export)
Excel Files (.xlsx)
```

#### 1. K-Means Clustering

**Algorithm**: Lloyd's algorithm
**Parameters**:
- `n_clusters`: 3, 4, 5 (test different values)
- `random_state`: 42 (reproducibility)
- `n_init`: 'auto' (automatic initialization runs)
- `max_iter`: 300

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for n_clusters in [3, 4, 5]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init='auto',
        max_iter=300
    )
    clusters = kmeans.fit_predict(reduced_embeddings)

    # Quality metric
    silhouette_avg = silhouette_score(reduced_embeddings, clusters)
    print(f"K={n_clusters}, Silhouette: {silhouette_avg:.3f}")
```

**Pros / Avantajlar**:
- Fast and scalable (hızlı ve ölçeklenebilir)
- Easy to interpret (yorumlaması kolay)
- Deterministic (random_state ile)

**Cons / Dezavantajlar**:
- Assumes spherical clusters (küre şeklinde kümeler varsayar)
- Requires specifying k (küme sayısını belirtmek gerekir)

#### 2. HDBSCAN (Hierarchical Density-Based Clustering)

**Algorithm**: Density-based hierarchical clustering
**Parameters**:
- `min_cluster_size`: 3, 5, 7 (minimum cluster size)
- `min_samples`: 1 (minimum core points)
- `metric`: 'euclidean'
- `cluster_selection_method`: 'eom' (Excess of Mass)

```python
import hdbscan

for min_size in [3, 5, 7]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        gen_min_span_tree=True
    )
    clusters = clusterer.fit_predict(reduced_embeddings)

    # Count clusters and noise
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Min size={min_size}, Clusters={n_clusters}, Noise={n_noise}")
```

**Pros / Avantajlar**:
- Automatically determines cluster count (küme sayısını otomatik belirler)
- Detects noise/outliers (gürültü/outlier tespit eder)
- Handles arbitrary-shaped clusters (farklı şekillerdeki kümeleri bulabilir)

**Cons / Dezavantajlar**:
- Sensitive to parameters (parametrelere duyarlı)
- Computationally expensive for large datasets (büyük veri setlerinde yavaş)

#### 3. SOM (Self-Organizing Maps)

**Algorithm**: Kohonen network
**Grid Sizes**: Calculated to approximate desired cluster count
- 4 clusters → 2×2 grid
- 6 clusters → 2×3 grid
- 9 clusters → 3×3 grid

```python
from minisom import MiniSom
import numpy as np

# For 4 clusters (2x2 grid)
som = MiniSom(
    x=2,
    y=2,
    input_len=2,  # UMAP reduced to 2D
    sigma=0.5,
    learning_rate=0.5,
    neighborhood_function='gaussian',
    random_seed=42
)

# Train
som.train_random(reduced_embeddings, 100)

# Get cluster assignments
clusters = []
for emb in reduced_embeddings:
    winner = som.winner(emb)
    cluster_id = winner[0] * 2 + winner[1]
    clusters.append(cluster_id)
```

**Pros / Avantajlar**:
- Preserves topological relationships (topolojik yapıyı korur)
- Visual representation (görsel temsil)
- Dimensionality reduction + clustering combined

**Cons / Dezavantajlar**:
- Grid size selection critical (grid boyutu seçimi kritik)
- Training can be slow (eğitim yavaş olabilir)

**Quality Metrics / Kalite Metrikleri**:
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- Visual inspection: UMAP scatter plots with cluster colors

---

### Stage 5: Theme Labeling & Interpretation / Tema Etiketleme ve Yorumlama

```
Cluster Members (Transcripts)
    ↓ (Keyword Extraction)
Representative Keywords (TF-IDF)
    ↓ (LLM Prompting)
    ├── Flan-T5 Base
    └── Summarization Pipeline
    ↓
Cluster Labels & Descriptions
    ↓ (Add to DataFrame)
Final Labeled Results
    ↓ (Export)
Excel Files
```

**LLM Models Used / Kullanılan LLM Modelleri**:
- **Flan-T5 Base**: `google/flan-t5-base`
- **Task**: Text-to-text generation for cluster naming
- **Alternative**: Summarization pipeline for concise titles

**Labeling Process / Etiketleme Süreci**:
1. Extract representative documents from each cluster (her kümeden temsili dökümanları çıkar)
2. Generate keywords using TF-IDF or CountVectorizer (TF-IDF ile anahtar kelimeler oluştur)
3. Prompt LLM with cluster keywords and sample texts (LLM'e anahtar kelimeler ve örnek metinlerle prompt gönder)
4. Generate concise, descriptive cluster labels (kısa ve açıklayıcı küme etiketleri oluştur)
5. Add labels to output DataFrame (etiketleri DataFrame'e ekle)

**Code Example / Kod Örneği**:
```python
from transformers import pipeline

def label_clusters_with_llm(df, cluster_column='cluster', text_column='transcript', n_samples=5):
    """Generate LLM labels for each cluster"""

    # Initialize text generation pipeline
    text2text = pipeline("text2text-generation", model="google/flan-t5-base")

    cluster_labels = {}

    for cluster_id in df[cluster_column].unique():
        if cluster_id == -1:  # Skip noise in HDBSCAN
            cluster_labels[cluster_id] = "Noise/Outliers"
            continue

        # Sample texts from cluster
        cluster_texts = df[df[cluster_column] == cluster_id][text_column]
        samples = cluster_texts.sample(min(n_samples, len(cluster_texts)))

        # Create prompt
        prompt = f"""
        Summarize the main theme of these video transcripts in 2-3 words:

        {chr(10).join(samples.tolist()[:3])}

        Theme:
        """

        # Generate label
        result = text2text(prompt, max_length=20, num_return_sequences=1)
        label = result[0]['generated_text'].strip()
        cluster_labels[cluster_id] = label

    # Add labels to dataframe
    df['cluster_label'] = df[cluster_column].map(cluster_labels)

    return df
```

**Example Labels / Örnek Etiketler**:
- Cluster 0: "Education & Teaching" / "Eğitim ve Öğretim"
- Cluster 1: "Technology News" / "Teknoloji Haberleri"
- Cluster 2: "Health & Lifestyle" / "Sağlık ve Yaşam"
- Cluster 3: "Music & Arts" / "Müzik ve Sanat"

---

## Machine Learning Models / Makine Öğrenmesi Modelleri

### Pre-trained Models / Önceden Eğitilmiş Modeller

#### 1. Sentence-BERT: `all-MiniLM-L6-v2`

**Type / Tip**: Sentence embedding model
**Architecture / Mimari**: MiniLM (distilled from BERT)
**Embedding Size / Gömme Boyutu**: 384 dimensions
**Training**: Sentence similarity on large corpus
**Performance**: ~3000 sentences/second on CPU

**Use Case / Kullanım Alanı**: Convert transcripts to semantic vectors

**Pros / Avantajlar**:
- Fast inference (hızlı çıkarım)
- Pre-trained, no fine-tuning needed (önceden eğitilmiş)
- High-quality embeddings (yüksek kaliteli gömme)
- Multilingual support (çok dilli destek)

**Limitations / Limitasyonlar**:
- Not optimized for Turkish (Türkçe için optimize edilmemiş)
- Maximum token length: 256
- English-focused training

**Model Card**: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

#### 2. Flan-T5 Base: `google/flan-t5-base`

**Type / Tip**: Text-to-text transformer
**Architecture / Mimari**: T5 (Text-to-Text Transfer Transformer)
**Parameters / Parametre Sayısı**: ~250M
**Training**: Instruction fine-tuning on FLAN tasks

**Use Case / Kullanım Alanı**: Cluster labeling, text generation, summarization

**Features / Özellikler**:
- Zero-shot task understanding (sıfır-atış görev anlama)
- Instruction-following capabilities (talimat takip etme)
- Versatile for various NLP tasks

**Model Card**: [https://huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

---

### Unsupervised Learning Algorithms / Denetimsiz Öğrenme Algoritmaları

#### K-Means
- **Type**: Centroid-based clustering
- **Complexity**: O(n × k × i × d)
- **Pros**: Fast, scalable, interpretable
- **Cons**: Assumes spherical clusters, requires k specification
- **Implementation**: `sklearn.cluster.KMeans`

#### HDBSCAN
- **Type**: Density-based hierarchical clustering
- **Complexity**: O(n²) worst case
- **Pros**: No cluster count needed, handles noise, arbitrary shapes
- **Cons**: Sensitive to parameters
- **Implementation**: `hdbscan.HDBSCAN`

#### SOM
- **Type**: Neural network-based clustering
- **Topology**: 2D grid
- **Pros**: Preserves topological relationships, visual
- **Cons**: Grid size affects cluster count
- **Implementation**: `minisom.MiniSom`

#### UMAP
- **Type**: Manifold learning for dimensionality reduction
- **Purpose**: Reduce 384D embeddings to 2D for visualization
- **Pros**: Preserves local and global structure
- **Implementation**: `umap.UMAP`

---

## Dependencies & Requirements / Bağımlılıklar ve Gereksinimler

### Python Packages / Python Paketleri

#### Core Data Processing / Temel Veri İşleme:
```bash
pip install pandas numpy openpyxl
```

#### YouTube APIs:
```bash
pip install google-api-python-client youtube-transcript-api
```

#### Machine Learning & NLP:
```bash
pip install sentence-transformers transformers torch
pip install scikit-learn umap-learn hdbscan minisom
```

#### OpenAI & Translation:
```bash
pip install openai>=1.0.0
pip install deep-translator
```

#### Visualization / Görselleştirme:
```bash
pip install matplotlib seaborn
```

#### Full Installation Command / Tam Kurulum Komutu:
```bash
pip install pandas numpy openpyxl google-api-python-client \
    youtube-transcript-api sentence-transformers transformers torch \
    scikit-learn umap-learn hdbscan minisom openai deep-translator \
    matplotlib seaborn
```

### Package Versions / Paket Versiyonları

| Package | Version / Versiyon | Purpose / Amaç |
|---------|-------------------|---------------|
| `pandas` | ≥2.0.0 | DataFrame operations |
| `numpy` | ≥1.24.0 | Numerical computing |
| `sentence-transformers` | ≥2.2.0 | Text embeddings |
| `transformers` | ≥4.30.0 | Transformer models |
| `torch` | ≥2.0.0 | PyTorch backend |
| `scikit-learn` | ≥1.3.0 | K-Means, metrics |
| `hdbscan` | ≥0.8.29 | HDBSCAN clustering |
| `minisom` | ≥2.3.0 | Self-Organizing Maps |
| `umap-learn` | ≥0.5.3 | UMAP algorithm |
| `openai` | ≥1.0.0 | OpenAI API |
| `deep-translator` | ≥1.11.0 | Translation |
| `google-api-python-client` | ≥2.0.0 | YouTube Data API |
| `youtube-transcript-api` | ≥0.6.0 | Transcript extraction |
| `openpyxl` | ≥3.1.0 | Excel read/write |
| `matplotlib` | ≥3.7.0 | Plotting |
| `seaborn` | ≥0.12.0 | Statistical viz |

### API Keys Required / Gerekli API Anahtarları

1. **YouTube Data API v3**
   - Get from / Alın: [Google Cloud Console](https://console.cloud.google.com/)
   - Set as / Ayarlayın: `os.environ['YOUTUBE_API_KEY']`

2. **OpenAI API** (Optional / Opsiyonel)
   - Get from / Alın: [OpenAI Platform](https://platform.openai.com/)
   - Set as / Ayarlayın: `openai.api_key` or environment variable

### System Requirements / Sistem Gereksinimleri

- **Python**: 3.8 or higher / 3.8 veya üstü
- **RAM**: 4GB minimum (8GB recommended / önerilir for large datasets)
- **Storage / Depolama**: 500MB+ for models and data
- **Internet**: Required for API calls and model downloads (API çağrıları ve model indirmeleri için gerekli)
- **GPU** (Optional): CUDA-compatible GPU for faster embeddings

---

## Data Flow Diagram / Veri Akış Diyagramı

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION PHASE                         │
│                    VERİ TOPLAMA AŞAMASI                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  YouTube Playlist │
                    └───────────────────┘
                              │
                              │ YouTube Data API v3
                              ▼
        ┌────────────────────────────────────────┐
        │  storybox_videos_utf8_bom.csv          │
        │  (Video List: Title + URL)             │
        └────────────────────────────────────────┘
                              │
                              │ YouTube Transcript API
                              ▼
        ┌────────────────────────────────────────┐
        │  Raw Video Transcripts                 │
        │  (Ham Video Transkriptleri)            │
        └────────────────────────────────────────┘
                              │
                              │ OpenAI API (Optional)
                              ▼
        ┌────────────────────────────────────────┐
        │  analyzed_storybox_videos.csv          │
        │  (Transcripts + Summary + Sentiment)   │
        └────────────────────────────────────────┘
                              │
                              │ Manual Curation / Formatting
                              ▼
        ┌────────────────────────────────────────┐
        │  Helsinki_Opus_Transcript.xlsx         │
        │  (Main Dataset for Clustering)         │
        └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING PHASE                       │
│                  ÖZELLİK MÜHENDİSLİĞİ AŞAMASI                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  Text Preprocessing                    │
        │  • Stopword Removal                    │
        │  • Normalization                       │
        │  • Cleaning                            │
        └────────────────────────────────────────┘
                              │
                              │ Sentence-BERT (all-MiniLM-L6-v2)
                              ▼
        ┌────────────────────────────────────────┐
        │  Text Embeddings (384D)                │
        │  (Metin Gömmesi)                       │
        └────────────────────────────────────────┘
                              │
                              │ UMAP Dimensionality Reduction
                              ▼
        ┌────────────────────────────────────────┐
        │  Reduced Embeddings (2D)               │
        │  (İndirgenmis Gömme)                   │
        └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CLUSTERING PHASE                              │
│                    KÜMELEME AŞAMASI                              │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐     ┌───────────┐
    │  K-Means  │      │  HDBSCAN  │     │    SOM    │
    │ (3,4,5)   │      │  (3,5,7)  │     │  (4,6,9)  │
    └───────────┘      └───────────┘     └───────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  Cluster Assignments + Quality Scores  │
        │  (Küme Atamaları + Kalite Skorları)    │
        └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  INTERPRETATION PHASE                            │
│                  YORUMLAMA AŞAMASI                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  Keyword Extraction (TF-IDF)           │
        │  (Anahtar Kelime Çıkarma)              │
        └────────────────────────────────────────┘
                              │
                              │ Flan-T5 / Summarization Pipeline
                              ▼
        ┌────────────────────────────────────────┐
        │  LLM-Generated Cluster Labels          │
        │  (LLM Tarafından Oluşturulan Etiketler)│
        └────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  Visualization (UMAP Scatter Plots)    │
        │  (Görselleştirme)                      │
        └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT PHASE                                │
│                      ÇIKTI AŞAMASI                               │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
  ┌──────────────┐   ┌──────────────┐  ┌──────────────┐
  │ kmeans_*.xlsx│   │hdbscan_*.xlsx│  │  som_*.xlsx  │
  │              │   │              │  │              │
  │ • Clusters   │   │ • Clusters   │  │ • Clusters   │
  │ • Labels     │   │ • Labels     │  │ • Labels     │
  │ • Themes     │   │ • Themes     │  │ • Themes     │
  │ • Keywords   │   │ • Keywords   │  │ • Keywords   │
  └──────────────┘   └──────────────┘  └──────────────┘
```

---

## Configuration Parameters / Konfigürasyon Parametreleri

### API Configuration / API Konfigürasyonu

```python
# YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
OPENAI_MAX_TOKENS = 2000
OPENAI_TEMPERATURE = 0.5
```

### Clustering Parameters / Kümeleme Parametreleri

#### K-Means
```python
KMEANS_CONFIG = {
    'n_clusters_range': [3, 4, 5],
    'random_state': 42,
    'n_init': 'auto',
    'max_iter': 300,
    'algorithm': 'lloyd'
}
```

#### HDBSCAN
```python
HDBSCAN_CONFIG = {
    'min_cluster_size_range': [3, 5, 7],
    'min_samples': 1,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
    'gen_min_span_tree': True
}
```

#### SOM
```python
SOM_CONFIG = {
    'grids': [(2, 2), (3, 2), (3, 3)],
    'sigma': 0.5,
    'learning_rate': 0.5,
    'num_iterations': 100,
    'neighborhood_function': 'gaussian',
    'random_seed': 42
}
```

### Text Processing Parameters / Metin İşleme Parametreleri

```python
# Turkish Stopwords
TURKISH_STOPWORDS = {
    've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'gibi',
    'olarak', 'sonra', 'önce', 'yılında', 'artık', 'çok',
    'tüm', 'her', 'ise', 'daha', 'ben', 'sen', 'o', 'biz',
    'siz', 'onlar', 'şu', 'şey', 'var', 'yok', 'mi', 'mu', 'mü'
}

# Keyword Extraction
KEYWORD_CONFIG = {
    'max_features': 100,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.85
}
```

### UMAP Parameters
```python
UMAP_CONFIG = {
    'n_components': 2,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': 42
}
```

---

## Troubleshooting / Sorun Giderme

### Common Errors / Sık Karşılaşılan Hatalar

#### 1. Encoding Errors / Kodlama Hataları

**Problem**: `UnicodeDecodeError` when reading CSV
**Solution / Çözüm**:
```python
# Try UTF-8 with BOM
df = pd.read_csv(file, encoding='utf-8-sig')

# Or Windows Turkish
df = pd.read_csv(file, encoding='windows-1254')
```

#### 2. API Rate Limit

**Problem**: `429 Too Many Requests` error
**Solution / Çözüm**:
```python
import time

for video in videos:
    process_video(video)
    time.sleep(1)  # Add 1 second delay
```

#### 3. Transcript Not Available

**Problem**: `TranscriptsDisabled` exception
**Solution / Çözüm**:
```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr', 'en'])
except TranscriptsDisabled:
    print(f"No transcript for {video_id}")
    transcript = None
```

#### 4. Memory Issues

**Problem**: Out of memory when processing large datasets
**Solution / Çözüm**:
```python
# Process in batches
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    process_batch(batch)
```

#### 5. Model Download Issues

**Problem**: Slow or failed model downloads
**Solution / Çözüm**:
```python
# Set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# Use offline mode if models are already downloaded
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/path/to/cache')
```

---

## Usage Example / Kullanım Örneği

### Complete Workflow / Tam İş Akışı

```python
import os
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
import hdbscan

# 1. Set API Keys
os.environ['YOUTUBE_API_KEY'] = 'your_youtube_api_key'
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

# 2. Fetch YouTube Playlist (Cells 1-3)
playlist_id = 'YOUR_PLAYLIST_ID'
# ... (run playlist fetching code)
# Output: storybox_videos_utf8_bom.csv

# 3. Extract Transcripts and Analyze (Cells 7-9)
CSV_FILE = 'storybox_videos_utf8_bom.csv'
OUTPUT_FILE = 'analyzed_storybox_videos_v2.csv'
# ... (run transcript extraction code)

# 4. Load Data for Clustering (Cells 10-14)
df = pd.read_excel('Helsinki_Opus_Transcript.xlsx')

# 5. Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['transcript'].tolist())

# 6. Dimensionality Reduction
reducer = umap.UMAP(n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)

# 7. Clustering
# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster_kmeans'] = kmeans.fit_predict(reduced_embeddings)

# HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df['cluster_hdbscan'] = clusterer.fit_predict(reduced_embeddings)

# 8. LLM Labeling
# ... (run cluster labeling code)

# 9. Export Results
df.to_excel('kmeans_4_labeled_llm.xlsx', index=False)
df.to_excel('hdbscan_min5_labeled_llm.xlsx', index=False)
```

---

## Data Security & Privacy / Veri Güvenliği ve Gizlilik

### API Key Security / API Anahtarı Güvenliği

**DO NOT / YAPMAYIN**:
```python
api_key = "AIzaSyXXXXXXXXXXXXXXXXXX"  # ❌ Never hardcode!
```

**DO / YAPIN**:
```python
import os

# From environment variable
api_key = os.getenv("YOUTUBE_API_KEY")

# Or from .env file
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
```

### .gitignore Settings

```gitignore
# API keys
.env
*.env
*_api_key.txt

# Large data files
*.xlsx
*.csv
analyzed_*.csv
*_labeled_*.xlsx
Helsinki_Opus_Transcript.xlsx

# Notebook checkpoints
.ipynb_checkpoints/
__pycache__/

# Model cache
.cache/
models/
```

---

## License & Terms / Lisans ve Şartlar

### YouTube Data

YouTube API'den alınan veriler için:
- YouTube'un [Terms of Service](https://www.youtube.com/t/terms)'ine uyulmalıdır
- API usage limits must be respected (API kullanım limitlerine uyulmalıdır)
- Do not redistribute data without permission (verileri izinsiz dağıtmayın)
- Respect copyright (telif haklarına saygı gösterin)

### OpenAI Data

OpenAI API ile oluşturulan içerik için:
- OpenAI'ın [Usage Policies](https://openai.com/policies/usage-policies)'ine uyulmalıdır
- Check license before commercial use (ticari kullanım öncesi lisans kontrol edin)

---

## References / Referanslar

### Academic Papers / Akademik Makaleler

1. **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

2. **HDBSCAN**: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD 2013.

3. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

4. **SOM**: Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE, 78(9), 1464-1480.

### API Documentation / API Dokümantasyonu

- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [OpenAI API](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)

---

**Last Updated / Son Güncelleme**: 2026-01-04
**Version / Versiyon**: 2.0
**Notebook**: YouTube Video Comprehensive Analyzer.ipynb
**Author / Yazar**: [Your Name]
**Contact / İletişim**: [Your Email]

---

## Summary / Özet

This document provides comprehensive documentation for all datasets used in the YouTube Video Comprehensive Analyzer notebook, including:

Bu doküman, YouTube Video Kapsamlı Analiz notebook'unda kullanılan tüm veri setleri için kapsamlı dokümantasyon sağlar:

- **3 Main Data Sources**: YouTube Playlist, Transcripts, Analysis Results
- **4 API Services**: YouTube Data API, YouTube Transcript API, OpenAI API, Translation API
- **3 Clustering Methods**: K-Means, HDBSCAN, SOM
- **2 ML Models**: Sentence-BERT, Flan-T5
- **Multiple Output Formats**: CSV, Excel (XLSX)
- **Turkish & English Support**: Bilingual processing and analysis

For questions or issues, please refer to the troubleshooting section or contact the maintainers.

Sorular veya sorunlar için lütfen sorun giderme bölümüne bakın veya bakımcılarla iletişime geçin.
