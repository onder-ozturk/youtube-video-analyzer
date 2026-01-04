# Veri Setleri ve Veri Kaynakları Dokümantasyonu

Bu doküman, "YouTube Video Comprehensive Analyzer" notebook'unda kullanılan tüm veri setlerini, veri kaynaklarını ve bunların nasıl kullanıldığını detaylı olarak açıklar.

## İçindekiler

1. [Kullanılan Veri Setleri](#kullanılan-veri-setleri)
2. [Veri Kaynakları ve API'ler](#veri-kaynakları-ve-apiler)
3. [Veri Yükleme ve Ön İşleme](#veri-yükleme-ve-ön-işleme)
4. [Çıktı Veri Dosyaları](#çıktı-veri-dosyaları)
5. [Veri Akışı Diyagramı](#veri-akışı-diyagramı)
6. [Kullanılan Kütüphaneler ve Modeller](#kullanılan-kütüphaneler-ve-modeller)
7. [Konfigürasyon Parametreleri](#konfigürasyon-parametreleri)

---

## Kullanılan Veri Setleri

### 1. StoryBox YouTube Playlist Veri Seti

**Kaynak Tipi**: YouTube API v3 Verisi
**Format**: CSV Dosyaları (birden fazla encoding varyantı)
**Köken**: YouTube Playlist (API üzerinden sağlanan playlist_id)

#### Oluşturulan Dosyalar:
- `storybox_videos_utf8_bom.csv` - UTF-8 with BOM (Excel uyumlu)
- `storybox_videos_utf8.csv` - Saf UTF-8 formatı
- `storybox_videos_win1254.csv` - Windows-1254 Türkçe encoding
- `storybox_videos_manual.csv` - Manuel encoding kontrolü

#### Veri Yapısı:
```csv
title,url
"Video Başlığı 1","https://www.youtube.com/watch?v=VIDEO_ID1"
"Video Başlığı 2","https://www.youtube.com/watch?v=VIDEO_ID2"
```

**Sütunlar**:
- `title`: Video başlığı (string)
- `url`: YouTube video URL'si (string)

**Kaynak**: YouTube Playlist Items API endpoint
**İstek başına maksimum sonuç**: 50 öğe
**Sayfalandırma**: nextPageToken ile pagination kullanılır

---

### 2. Helsinki Opus Transcript Veri Seti

**Kaynak Tipi**: Excel Dosyası (`.xlsx`)
**Dosya Adı**: `Helsinki_Opus_Transcript.xlsx`
**Format**: Yapılandırılmış sütunlar içeren Excel

#### Gerekli Sütunlar:
- `Video URL` - Analiz edilen videoların URL'leri
- `Transcript` - YouTube'dan çıkarılan tam video transkriptleri
- `Summary` - Video içeriğinin metin özetleri

#### Veri Yapısı Örneği:
```
Video URL                                    | Transcript                  | Summary
https://www.youtube.com/watch?v=VIDEO_ID    | [Transkript metni...]       | [Özet metni...]
```

**Kullanım Amacı**:
- Kümeleme analizi için hazır transkript ve özet verisi sağlar
- Farklı kümeleme algoritmalarıyla karşılaştırma yapmak için kullanılır
- LLM tabanlı etiketleme için kaynak veri

---

## Veri Kaynakları ve API'ler

### A. YouTube Data API v3

**Amaç**: Playlist videolarını ve metadata'yı çıkarmak
**Kimlik Doğrulama**: API anahtarı (ortam değişkeninden: `YOUTUBE_API_KEY`)

**Kullanılan Endpoint'ler**:
```python
youtube.playlistItems().list(
    part='snippet,contentDetails',
    playlistId=playlist_id,
    maxResults=50
)
```

**Dönen Veri**:
- `videoId` - Video kimliği
- `title` - Video başlığı
- `snippet` - Video meta bilgileri
- `contentDetails` - İçerik detayları

**Veri Çıkarımı**: Video başlıkları ve URL'leri CSV oluşturma için kullanılır

**API Limitleri**:
- Günlük quota: 10,000 birim (varsayılan)
- playlistItems.list çağrısı: 1 birim

---

### B. YouTube Transcript API

**Amaç**: YouTube videolarından altyazı/transkript çıkarmak
**Kütüphane**: `youtube_transcript_api` (YouTubeTranscriptApi)

**Kullanım**:
```python
from youtube_transcript_api import YouTubeTranscriptApi

transcript = YouTubeTranscriptApi.get_transcript(
    video_id,
    languages=['tr', 'en']
)
```

**Çıktı**: Zaman damgalı metin transkriptleri

**Özellikler**:
- Otomatik ve manuel altyazıları destekler
- Çoklu dil desteği
- Zaman damgası bilgisi içerir

**Limitasyonlar**:
- Tüm videolarda transkript bulunmayabilir
- Bazı videolar transkript erişimini engelleyebilir

---

### C. OpenAI API

**Amaç**: Video analizi, özetleme ve topic etiketleme
**Versiyon**: OpenAI v1.0+ (ClientAPI uyumlu)

**Kullanım Alanları**:

1. **Video İçeriği Analizi**
```python
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Video özetleyici"},
        {"role": "user", "content": f"Özetle: {transcript}"}
    ]
)
```

2. **Video Özetleri Oluşturma**
- Transkriptlerden kısa özetler
- Anahtar noktaları çıkarma
- İçerik kategorilendirme

3. **Küme Etiketleme (LLM Labeling)**
```python
# Her kümeden örnek metinler
sample_texts = [cluster_samples...]

# LLM ile etiket oluşturma
prompt = f"""
Aşağıdaki video özetlerine dayanarak bu grubun ana temasını
2-3 kelime ile tanımla:
{sample_texts}
"""
label = get_llm_response(prompt)
```

4. **Dil Çevirisi (Yardımcı)**
- Özetlerin İngilizce çevirisi
- Çok dilli içerik normalizasyonu

**API Parametreleri**:
- Model: `gpt-3.5-turbo` veya `gpt-4`
- Maksimum token: 2000-4000
- Temperature: 0.3-0.7 (özetleme için)

---

### D. Google Translate / Deep Translator

**Amaç**: Video transkriptlerini ve özetlerini çevirmek

**Kullanılan Kütüphaneler**:

1. **Deep Translator** (Tercih edilen):
```python
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='tr', target='en')
translated = translator.translate(text)
```

2. **Googletrans** (Yedek):
```python
from googletrans import Translator

translator = Translator()
result = translator.translate(text, src='tr', dest='en')
```

**Desteklenen Diller**:
- Türkçe → İngilizce
- İngilizce → Türkçe
- Otomatik dil tespiti

**Kullanım Senaryoları**:
- Türkçe transkriptleri İngilizce'ye çevirme
- Çok dilli kümeleme için veri normalizasyonu
- Özet metinlerini standart dile çevirme

---

## Veri Yükleme ve Ön İşleme

### Aşama 1: Playlist Veri Toplama

```python
# Adım 1: YouTube API ile playlist videolarını çek
playlist_id = "PLnAF4npbrTwzO2W6v07ktEjBU6885k5hB"  # Örnek
api_key = os.getenv("YOUTUBE_API_KEY")
videos = get_all_playlist_videos(playlist_id, api_key)
# Döner: Liste [{'title': '...', 'url': '...'}, ...]

# Adım 2: Encoding işleme ile CSV'ye kaydet
# Türkçe karakterler için birden fazla encoding
save_to_csv_multiple_encodings(videos, base_filename="storybox_videos")
```

**Çıktı Dosyaları**:
- UTF-8 BOM: Excel'de doğrudan açılabilir
- UTF-8: Genel programlama kullanımı
- Windows-1254: Eski Windows sistemleri için

---

### Aşama 2: Video Analizi

```python
# Giriş: CSV dosyası (storybox_videos_utf8_bom.csv)
df = pd.read_csv(csv_file, encoding='utf-8-sig')

# Her video URL'si için:
for index, row in df.iterrows():
    video_url = row['url']

    # 1. Transkript çıkar
    transcript = extract_transcript(video_url)

    # 2. OpenAI ile özetle
    summary = summarize_with_openai(transcript)

    # 3. Google Translate ile çevir
    summary_en = translate_to_english(summary)
    transcript_en = translate_to_english(transcript)

    # 4. Sonuçları kaydet
    save_to_csv(analyzed_data)
```

**Çıktı**: `analyzed_storybox_videos.csv` veya `v2` versiyonu

**Hata Yönetimi**:
- Transkript bulunamazsa: `Status = "No Transcript"`
- API hatası: `Status = "Error"`
- Başarılı: `Status = "Success"`

---

### Aşama 3: Transkript Verisi Yükleme

```python
# Excel dosyasını yükle
df = pd.read_excel("Helsinki_Opus_Transcript.xlsx")

# Gerekli sütunları doğrula
required_columns = ["Video URL", "Transcript", "Summary"]
validate_columns(df, required_columns)

# Büyük/küçük harf duyarlılığı ve sütun eşleştirme
df.columns = df.columns.str.strip()
```

**Metin Ön İşleme**:
```python
# Küçük harfe çevir
df['text'] = df['text'].str.lower()

# Fazla boşlukları temizle
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

# Başında/sonundaki boşlukları sil
df['text'] = df['text'].str.strip()

# Özel karakterleri encoding ile işle
# UTF-8 encoding garantisi
```

---

### Aşama 4: Metin Embedding ve Vektörleştirme

#### Sentence Embedding
```python
from sentence_transformers import SentenceTransformer

# Model yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Metinleri vektörlere dönüştür
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32
)
# Çıktı: numpy array (n_samples, 384)
```

**Model Detayları**:
- Model: all-MiniLM-L6-v2
- Embedding boyutu: 384
- Dil: Çok dilli (İngilizce ağırlıklı)
- Performans: Hızlı ve hafif

#### Keyword Çıkarma

**İngilizce için**:
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    stop_words='english',
    max_features=50,
    ngram_range=(1, 2)  # unigram ve bigram
)
keyword_matrix = vectorizer.fit_transform(texts)
```

**Türkçe için**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

TURKISH_STOPWORDS = {
    've', 'bir', 'bu', 'da', 'de', 'ile', 'için',
    'gibi', 'olarak', 'sonra', 'önce', 'yılında',
    'artık', 'çok', 'tüm', 'her', 'ise', 'daha',
    'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
    'şu', 'şey', 'var', 'yok', 'mi', 'mu', 'mü'
}

vectorizer = TfidfVectorizer(
    stop_words=list(TURKISH_STOPWORDS),
    max_features=100,
    ngram_range=(1, 2),
    min_df=2  # En az 2 dokümanda geçmeli
)
```

---

### Aşama 5: Boyut İndirgeme (UMAP)

```python
import umap

# UMAP reducer
reducer = umap.UMAP(
    n_components=2,      # 2D görselleştirme için
    n_neighbors=15,      # Yerel komşuluk
    min_dist=0.1,        # Minimum nokta mesafesi
    metric='cosine',     # Cosine similarity
    random_state=42      # Tekrarlanabilirlik
)

# Boyut indirgeme
reduced_embeddings = reducer.fit_transform(embeddings)
# Çıktı: numpy array (n_samples, 2)
```

**UMAP Parametreleri**:
- `n_neighbors`: Küçük değer → detaylı yapı, Büyük değer → genel yapı
- `min_dist`: Küçük değer → sıkı kümeler, Büyük değer → dağılmış kümeler
- `metric`: 'cosine', 'euclidean', 'manhattan'

---

### Aşama 6: Kümeleme

#### 1. K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Farklı küme sayıları test et
for n_clusters in [3, 4, 5]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    clusters = kmeans.fit_predict(embeddings)

    # Kalite metriği
    silhouette_avg = silhouette_score(embeddings, clusters)
    print(f"K={n_clusters}, Silhouette: {silhouette_avg:.3f}")
```

**Çıktı Dosyası**: `kmeans_{n_clusters}_labeled_llm.xlsx`

**Avantajlar**:
- Hızlı ve basit
- Küme sayısını belirleyebilme
- Yorumlanması kolay

**Dezavantajlar**:
- Küre şeklinde kümeler varsayar
- Küme sayısını önceden belirlemek gerekir

---

#### 2. HDBSCAN (Hierarchical Density-Based Clustering)

```python
import hdbscan

# Farklı minimum küme büyüklükleri test et
for min_size in [3, 5, 7]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    clusters = clusterer.fit_predict(embeddings)

    # Gürültü noktaları: label = -1
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"Min size={min_size}, Clusters={n_clusters}, Noise={n_noise}")
```

**Çıktı Dosyası**: `hdbscan_min{size}_labeled_llm.xlsx`

**Avantajlar**:
- Küme sayısını otomatik belirler
- Gürültüyü (outlier) tespit eder
- Farklı yoğunluktaki kümeleri bulabilir

**Dezavantajlar**:
- Parametrelere duyarlı
- Küçük veri setlerinde zayıf performans

---

#### 3. Self-Organizing Maps (SOM)

```python
from minisom import MiniSom

# Farklı grid boyutları test et
configs = [
    (2, 2, 4),   # 2x2 grid, 4 küme
    (3, 2, 6),   # 3x2 grid, 6 küme
    (3, 3, 9)    # 3x3 grid, 9 küme
]

for x_dim, y_dim, n_clusters in configs:
    som = MiniSom(
        x=x_dim,
        y=y_dim,
        input_len=embedding_dim,
        sigma=1.0,
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )

    # Eğitim
    som.train_random(embeddings, num_iteration=1000)

    # Küme ataması
    clusters = []
    for emb in embeddings:
        winner = som.winner(emb)
        cluster_id = winner[0] * y_dim + winner[1]
        clusters.append(cluster_id)
```

**Çıktı Dosyası**: `som_{n_clusters}_labeled_llm.xlsx`

**Avantajlar**:
- Topolojik yapı korur
- Görselleştirmesi kolay
- Boyut indirgeme + kümeleme birlikte

**Dezavantajlar**:
- Grid boyutu seçimi kritik
- Eğitim süresi uzun olabilir

---

### Aşama 7: LLM ile Küme Etiketleme

```python
def label_clusters_with_llm(df, cluster_column, text_column, n_samples=5):
    """Her küme için LLM ile anlamlı etiket oluştur"""

    cluster_labels = {}

    for cluster_id in df[cluster_column].unique():
        # Her kümeden örnek metinler
        cluster_texts = df[df[cluster_column] == cluster_id][text_column]
        samples = cluster_texts.sample(min(n_samples, len(cluster_texts)))

        # LLM'e gönder
        prompt = f"""
        Aşağıdaki video özetleri aynı temalı bir gruptan geliyor.
        Bu grubun ana temasını 2-3 kelime ile özetle:

        {chr(10).join(samples.tolist())}

        Etiket (sadece 2-3 kelime):
        """

        label = get_openai_response(prompt)
        cluster_labels[cluster_id] = label

    # Etiketleri DataFrame'e ekle
    df['Cluster_Label'] = df[cluster_column].map(cluster_labels)

    return df
```

**Örnek Etiketler**:
- Küme 0: "Eğitim ve Öğretim"
- Küme 1: "Teknoloji Haberleri"
- Küme 2: "Sağlık ve Yaşam"
- Küme 3: "Müzik ve Sanat"

---

## Çıktı Veri Dosyaları

### CSV Dosyaları

#### 1. Playlist CSV'leri
```
storybox_videos_utf8_bom.csv
├── Sütunlar: title, url
├── Encoding: UTF-8 with BOM
└── Kullanım: Excel'de direkt açılabilir
```

#### 2. Analiz Sonuçları
```
analyzed_storybox_videos.csv
├── Video_URL: YouTube video linki
├── Video_Title: Video başlığı
├── Transcript: Tam transkript metni
├── Summary: Video özeti
├── Summary_English: İngilizce özet
├── Transcript_English: İngilizce transkript
└── Status: İşlem durumu (Success/Error/No Transcript)
```

```
analyzed_storybox_videos_v2.csv
└── Geliştirilmiş versiyon (ek alanlar ve iyileştirilmiş özetler)
```

---

### Excel Dosyaları

#### 1. K-Means Sonuçları
```
kmeans_labeled_llm.xlsx
kmeans_3_labeled_llm.xlsx
kmeans_4_labeled_llm.xlsx
kmeans_5_labeled_llm.xlsx
```

**Sütunlar**:
- Video URL
- Transcript / Summary
- Cluster (küme numarası)
- Cluster_Label (LLM etiketi)
- X_umap, Y_umap (görselleştirme koordinatları)

---

#### 2. HDBSCAN Sonuçları
```
hdbscan_min3_labeled_llm.xlsx
hdbscan_min5_labeled_llm.xlsx
hdbscan_min7_labeled_llm.xlsx
```

**Sütunlar**:
- Video URL
- Transcript / Summary
- Cluster (küme numarası, -1 = gürültü)
- Cluster_Label (LLM etiketi)
- Outlier_Score (gürültü skoru)

---

#### 3. SOM Sonuçları
```
som_4_labeled_llm.xlsx
som_6_labeled_llm.xlsx
som_9_labeled_llm.xlsx
```

**Sütunlar**:
- Video URL
- Transcript / Summary
- Cluster (küme numarası)
- Cluster_Label (LLM etiketi)
- SOM_X, SOM_Y (grid koordinatları)

---

#### 4. İyileştirilmiş Sonuçlar
```
kmeans_labeled_llm_improved.xlsx
```

**Ek Özellikler**:
- Daha detaylı LLM etiketleri
- Küme kalite metrikleri
- Anahtar kelimeler (keywords)
- Küme özet istatistikleri

---

## Veri Akışı Diyagramı

```
┌─────────────────────────────────────────────────────────────────┐
│                      YouTube Playlist                           │
│                   (Playlist ID: PLnAF...)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              YouTube Data API v3 (playlistItems.list)           │
│              Kimlik Doğrulama: YOUTUBE_API_KEY                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CSV Dosyaları                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  storybox_videos_utf8_bom.csv  (Excel uyumlu)           │   │
│  │  storybox_videos_utf8.csv       (Standart UTF-8)        │   │
│  │  storybox_videos_win1254.csv    (Windows Türkçe)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│           Sütunlar: [title, url]                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Her Video için İşlem Döngüsü                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. YouTube Transcript API → Transkript Çıkarma         │   │
│  │  2. OpenAI API → Özet Oluşturma                         │   │
│  │  3. Google Translate → İngilizce Çeviri                 │   │
│  │  4. Durum Kaydı (Success/Error/No Transcript)           │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Excel: Helsinki_Opus_Transcript.xlsx               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Video URL  │  Transcript  │  Summary                    │   │
│  │  ───────────┼──────────────┼─────────────                │   │
│  │  youtube... │  [metin...]  │  [özet...]                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           SentenceTransformer Embedding (all-MiniLM-L6-v2)      │
│              Metin → 384 boyutlu vektörler                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UMAP Boyut İndirgeme                         │
│                   384 boyut → 2 boyut (X, Y)                    │
│              Görselleştirme ve kümeleme için                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
         ┌───────────┐ ┌──────────┐ ┌─────────┐
         │  K-Means  │ │ HDBSCAN  │ │   SOM   │
         │ (n=3-5)   │ │(min=3-7) │ │ (2x2-   │
         │           │ │          │ │  3x3)   │
         └─────┬─────┘ └────┬─────┘ └────┬────┘
               │            │            │
               └────────────┼────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │    OpenAI LLM Küme Etiketleme        │
         │  - Her kümeden 5-10 örnek            │
         │  - LLM'e gönder                      │
         │  - 2-3 kelimelik etiket al           │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │      Excel Çıktı Dosyaları           │
         │  ┌────────────────────────────────┐  │
         │  │ kmeans_labeled_llm.xlsx        │  │
         │  │ hdbscan_min5_labeled_llm.xlsx  │  │
         │  │ som_6_labeled_llm.xlsx         │  │
         │  └────────────────────────────────┘  │
         │  Sütunlar:                           │
         │  - Video URL                         │
         │  - Transcript/Summary                │
         │  - Cluster (ID)                      │
         │  - Cluster_Label (LLM etiketi)       │
         │  - X_umap, Y_umap                    │
         └──────────────────────────────────────┘
```

---

## Kullanılan Kütüphaneler ve Modeller

### Kütüphane Kategorileri

| Kategori | Kütüphaneler | Versiyon | Amaç |
|----------|-------------|----------|------|
| **YouTube Entegrasyonu** | `google-api-python-client` | ≥2.0.0 | YouTube Data API erişimi |
| | `youtube-transcript-api` | ≥0.6.0 | Transkript çıkarma |
| **LLM & NLP** | `openai` | ≥1.0.0 | GPT API erişimi |
| | `deep-translator` | ≥1.11.0 | Çeviri (tercih edilen) |
| | `googletrans` | 4.0.0rc1 | Çeviri (yedek) |
| **Embedding** | `sentence-transformers` | ≥2.2.0 | Metin embedding |
| | `transformers` | ≥4.30.0 | Transformer modelleri |
| | `torch` | ≥2.0.0 | PyTorch backend |
| **Kümeleme** | `scikit-learn` | ≥1.3.0 | K-Means, metriler |
| | `hdbscan` | ≥0.8.29 | HDBSCAN algoritması |
| | `minisom` | ≥2.3.0 | Self-Organizing Maps |
| **Boyut İndirgeme** | `umap-learn` | ≥0.5.3 | UMAP algoritması |
| **Veri İşleme** | `pandas` | ≥2.0.0 | DataFrame işlemleri |
| | `numpy` | ≥1.24.0 | Numerik hesaplamalar |
| | `openpyxl` | ≥3.1.0 | Excel okuma/yazma |
| **Görselleştirme** | `matplotlib` | ≥3.7.0 | Grafik oluşturma |
| | `seaborn` | ≥0.12.0 | İstatistiksel görselleştirme |
| **Ek NLP** | `nltk` | ≥3.8.0 | Stopwords, tokenization |

---

### Kullanılan ML/DL Modelleri

#### 1. Sentence Transformer: all-MiniLM-L6-v2

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Özellikler**:
- **Model Tipi**: Sentence-BERT
- **Embedding Boyutu**: 384
- **Parametre Sayısı**: 22.7M
- **Diller**: Çok dilli (İngilizce ağırlıklı, Türkçe kısmen destekler)
- **Performans**: Hızlı çıkarım (~3000 cümle/saniye CPU'da)
- **Kullanım Alanı**: Semantik benzerlik, kümeleme, sınıflandırma

**Avantajlar**:
- Hafif ve hızlı
- Pre-trained, fine-tuning gerektirmez
- Yüksek kaliteli embedding

**Limitasyonlar**:
- Türkçe için optimize edilmemiş
- Maksimum token: 256

---

#### 2. UMAP (Uniform Manifold Approximation and Projection)

```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
```

**Özellikler**:
- **Algoritma Tipi**: Manifold öğrenme
- **Amaç**: Yüksek boyutlu veriyi düşük boyuta indirgeme
- **Matematiksel Temel**: Topolojik veri analizi

**Parametreler**:
- `n_components`: 2 (2D görselleştirme için)
- `n_neighbors`: 15 (yerel komşuluk)
- `min_dist`: 0.1 (minimum nokta mesafesi)
- `metric`: 'cosine' veya 'euclidean'

**t-SNE ile Karşılaştırma**:
- UMAP daha hızlı
- UMAP global yapıyı daha iyi korur
- UMAP deterministik (random_state ile)

---

#### 3. K-Means Clustering

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
```

**Algoritma**: Lloyd'un iteratif K-Means algoritması

**Parametreler**:
- `n_clusters`: Küme sayısı (3-5 arası test edilir)
- `n_init`: 10 (farklı başlangıç denemeleri)
- `max_iter`: 300 (maksimum iterasyon)
- `random_state`: 42 (tekrarlanabilirlik)

**Optimizasyon**: Küme içi toplam kare hataları minimize eder (WCSS)

---

#### 4. HDBSCAN

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
```

**Algoritma**: Hierarchical Density-Based Spatial Clustering

**Parametreler**:
- `min_cluster_size`: Minimum küme büyüklüğü (3-7 arası)
- `min_samples`: 1 (minimum core noktalar)
- `metric`: 'euclidean'
- `cluster_selection_method`: 'eom' (Excess of Mass)

**Özellikler**:
- Gürültüyü (outlier) otomatik tespit (label=-1)
- Küme sayısını otomatik belirler
- Farklı yoğunluklardaki kümeleri bulabilir

---

#### 5. Self-Organizing Maps (SOM)

```python
from minisom import MiniSom
som = MiniSom(x=3, y=2, input_len=384, sigma=1.0, learning_rate=0.5)
```

**Algoritma**: Kohonen Self-Organizing Map

**Parametreler**:
- `x, y`: Grid boyutları (örn. 3x2)
- `input_len`: 384 (embedding boyutu)
- `sigma`: 1.0 (komşuluk fonksiyonu genişliği)
- `learning_rate`: 0.5 (öğrenme hızı)
- `neighborhood_function`: 'gaussian'

**Eğitim**: 1000 iterasyon rastgele başlatma

---

## Konfigürasyon Parametreleri

### API Konfigürasyonu

```python
# YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"  # veya "gpt-4"
OPENAI_MAX_TOKENS = 2000
OPENAI_TEMPERATURE = 0.5
```

---

### Encoding Konfigürasyonu

```python
# CSV Encoding Seçenekleri
ENCODINGS = {
    'utf8_bom': 'utf-8-sig',      # Excel için önerilen
    'utf8': 'utf-8',                # Standart
    'win1254': 'windows-1254',      # Windows Türkçe
    'latin5': 'iso-8859-9'          # ISO Türkçe
}

# Varsayılan
DEFAULT_ENCODING = 'utf-8-sig'
```

---

### Kümeleme Parametreleri

#### K-Means
```python
KMEANS_CONFIG = {
    'n_clusters_range': [3, 4, 5],
    'random_state': 42,
    'n_init': 10,
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
    'prediction_data': True
}
```

#### SOM
```python
SOM_CONFIG = {
    'grids': [(2, 2), (3, 2), (3, 3)],
    'sigma': 1.0,
    'learning_rate': 0.5,
    'num_iterations': 1000,
    'neighborhood_function': 'gaussian',
    'random_seed': 42
}
```

---

### Metin İşleme Parametreleri

```python
# Keyword Extraction
KEYWORD_CONFIG = {
    'max_features': 50,           # İngilizce
    'max_features_tr': 100,       # Türkçe
    'ngram_range': (1, 2),        # unigram ve bigram
    'min_df': 2,                  # Min document frequency
    'max_df': 0.85                # Max document frequency (% olarak)
}

# Türkçe Stopwords
TURKISH_STOPWORDS = {
    've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'gibi',
    'olarak', 'sonra', 'önce', 'yılında', 'artık', 'çok',
    'tüm', 'her', 'ise', 'daha', 'ben', 'sen', 'o', 'biz',
    'siz', 'onlar', 'şu', 'şey', 'var', 'yok', 'mi', 'mu', 'mü',
    'den', 'dan', 'ten', 'tan', 'nin', 'nın', 'nun', 'nün'
}
```

---

### LLM Etiketleme Parametreleri

```python
LLM_LABELING_CONFIG = {
    'samples_per_cluster': 5,     # Her kümeden kaç örnek
    'max_samples': 10,            # Maksimum örnek sayısı
    'label_max_words': 3,         # Etiket maksimum kelime
    'temperature': 0.3,           # LLM creativity (düşük = tutarlı)
    'max_tokens': 50,             # Etiket için token limiti
    'model': 'gpt-3.5-turbo'      # Kullanılacak model
}
```

---

### UMAP Parametreleri

```python
UMAP_CONFIG = {
    'n_components': 2,            # 2D görselleştirme
    'n_neighbors': 15,            # Yerel komşuluk boyutu
    'min_dist': 0.1,              # Minimum nokta mesafesi
    'metric': 'cosine',           # Mesafe metriği
    'random_state': 42            # Seed
}
```

---

### Embedding Parametreleri

```python
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
    'show_progress_bar': True,
    'normalize_embeddings': True,
    'device': 'cpu'  # veya 'cuda' GPU için
}
```

---

## Veri Kalitesi ve Validasyon

### Veri Kalitesi Kontrolleri

```python
# 1. Eksik Veri Kontrolü
def check_missing_data(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    return missing_percent[missing_percent > 0]

# 2. Duplicate Kontrolü
def check_duplicates(df, column):
    duplicates = df[df.duplicated(column, keep=False)]
    return duplicates

# 3. Encoding Doğrulama
def validate_encoding(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

# 4. Transkript Kalitesi
def check_transcript_quality(transcript):
    if len(transcript) < 50:
        return "Too short"
    if transcript.count('[') > 5:  # Çok fazla otomatik tag
        return "Low quality auto-generated"
    return "OK"
```

---

### Kümeleme Kalite Metrikleri

```python
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Silhouette Score (-1 ile 1 arası, yüksek = iyi)
silhouette_avg = silhouette_score(embeddings, clusters)

# Davies-Bouldin Index (düşük = iyi)
db_score = davies_bouldin_score(embeddings, clusters)

# Calinski-Harabasz Index (yüksek = iyi)
ch_score = calinski_harabasz_score(embeddings, clusters)

print(f"Silhouette: {silhouette_avg:.3f}")
print(f"Davies-Bouldin: {db_score:.3f}")
print(f"Calinski-Harabasz: {ch_score:.3f}")
```

---

## Veri Güvenliği ve Gizlilik

### API Anahtarı Güvenliği

```python
# YANLIŞ - Kodda hardcode
api_key = "AIzaSyXXXXXXXXXXXXXXXXXX"  # ASLA YAPMAYIN!

# DOĞRU - Ortam değişkenlerinden yükleme
import os
api_key = os.getenv("YOUTUBE_API_KEY")

# DOĞRU - .env dosyasından
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
```

### .gitignore Ayarları

```gitignore
# API anahtarları
.env
*.env
*_api_key.txt

# Büyük veri dosyaları
*.xlsx
*.csv
analyzed_*.csv
*_labeled_*.xlsx

# Notebook checkpoint'leri
.ipynb_checkpoints/
__pycache__/

# Model cache
.cache/
models/
```

---

## Veri Seti Lisansı ve Kullanım Şartları

### YouTube Verileri

YouTube API'den alınan veriler için:
- YouTube'un [Kullanım Şartları](https://www.youtube.com/t/terms)'na uyulmalıdır
- API kullanım limitlerini aşmayın
- Verileri izinsiz yeniden dağıtmayın
- Telif haklarına saygı gösterin

### OpenAI Verileri

OpenAI API ile oluşturulan içerik için:
- OpenAI'ın [Kullanım Politikası](https://openai.com/policies/usage-policies)'na uyulmalıdır
- Oluşturulan içeriği ticari amaçla kullanmadan önce lisans kontrol edin

---

## Sorun Giderme ve SSS

### Sık Karşılaşılan Hatalar

#### 1. Encoding Hataları

```python
# Hata: UnicodeDecodeError
# Çözüm:
df = pd.read_csv(file, encoding='utf-8-sig')  # BOM ile UTF-8
# veya
df = pd.read_csv(file, encoding='windows-1254')  # Windows Türkçe
```

#### 2. API Rate Limit

```python
# Hata: 429 Too Many Requests
# Çözüm: Gecikme ekleyin
import time
for video in videos:
    process_video(video)
    time.sleep(1)  # 1 saniye bekle
```

#### 3. Transkript Bulunamadı

```python
# Hata: TranscriptsDisabled
# Çözüm: Try-except ile işleyin
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
except TranscriptsDisabled:
    transcript = None
    status = "No Transcript"
```

---

## Referanslar ve Kaynaklar

### Akademik Referanslar

1. **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

2. **HDBSCAN**: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. Pacific-Asia conference on knowledge discovery and data mining (pp. 160-172).

3. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

4. **SOM**: Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE, 78(9), 1464-1480.

### API Dokümantasyonu

- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [OpenAI API](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)

---

**Son Güncelleme**: 2026-01-04
**Versiyon**: 1.0
