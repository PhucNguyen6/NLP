# NLP Sentiment + RAG System

Du an phan tich cam xuc binh luan (vi/en) theo kien truc:

- Preprocessing
- Sentiment Module (SVM voi 3 encoder: `tfidf`, `doc2vec`, `xlmroberta`)
- RAG Pipeline (tim ngu canh tu vector database)
- LLM Service (giai thich ket qua)

## 0) Quy trinh chay nhanh (phien ban moi nhat)

Chay lan luot tu thu muc **root** cua repo (`NLP/`):

| Buoc | Lenh / hanh dong | Ghi chu |
|------|-------------------|---------|
| 1 | `python -m venv .venv` + kich hoat venv | Muc 2.1 |
| 2 | `pip install -r requirements.txt` | Muc 2.1 |
| 3 | Tao / cap nhat `.env` | Muc 2.2 (DB, LLM, **YouTube key** neu crawl) |
| 4 | (Tuy chon) Crawl comment YouTube | Muc 2.3 — can `YOUTUBE_API_KEY` + file ID video |
| 5 | (Bat buoc neu co raw) Tien xu ly raw -> clean | Muc 2.4 — tao `clean_comment.csv` |
| 6 | (Tuy chon) Tu dien `dictionary_tools.py` | Muc 7 — ho tro sentiment / mo rong tu |
| 7 | `python sentence_list/experiments.py all` | Encode + train SVM + bao cao/plot |
| 8 | `python sentence_list/setup_database.py` | RAG / DB |
| 9 | `python sentence_list/populate_embeddings.py` | Nap vector vao DB |
| 10 | Mo LM Studio (neu dung giai thich LLM) | Trung model voi `LLM_MODEL` trong `.env` |
| 11 | `streamlit run ...` hoac `main.py` | Suy luan |

**Chi can sentiment + SVM (khong RAG):** sau buoc **7** co the chay `main.py` (khong can buoc 8–9 neu khong dung RAG).

**Streamlit + RAG + LLM day du:** buoc **1 → 11** (bo qua 4 neu da co `clean_comment.csv` thu cong; bo qua 6 neu khong can cap nhat tu dien).

---

## 1) Cau truc chinh

- `sentence_list/crawl_sentence.py`: crawl binh luan YouTube -> `raw_data/raw_comment.csv` (can `YOUTUBE_API_KEY`, file `youtube_id_music.txt`).
- `sentence_list/preprocessing_comment.py`: tien xu ly raw -> `clean_data/clean_comment.csv` (cot `lang`).
- `sentence_list/main.py`: entrypoint suy luan (single/batch/interactive, compare encoder).
- `sentence_list/inference_pipeline.py`: pipeline theo architecture.
- `sentence_list/embeddings_manager.py`: tao embedding + fallback rebuild model/vectorizer.
- `sentence_list/experiments.py`: **toan bo** encode (TF-IDF / Doc2Vec / XLM-RoBERTa) + train SVM + danh gia train/val/test + xuat CSV/TXT/JSON + luu bieu do (`plots/`).
- `sentence_list/database.py`, `sentence_list/rag_retriever.py`: tang du lieu + truy hoi.
- `sentence_list/dictionary_tools.py`, `sentence_list/dictionary_expander.py`: xu ly tu dien (`clean_dict`).

Khong con cac script tach rieng `encode_label_*.py` hay `train_svm_models.py`.

---

## 2) Cai dat

### 2.1 Python environment

Khuyen nghi Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu ban khong dung `requirements.txt`, can toi thieu:

- `numpy`, `pandas`, `scikit-learn`, `joblib`, `tqdm`
- `torch`, `transformers`, `sentencepiece`
- `gensim`
- `matplotlib`, `seaborn` (bat buoc cho `experiments.py train`)
- `psycopg2-binary`, `python-dotenv` (RAG / DB)
- `google-api-python-client` (crawl YouTube), `emoji`, `langdetect`; **`nltk`** + tai WordNet neu dung `preprocessing_comment.py` (muc 2.4)

### 2.2 Bien moi truong

Tao file `.env` (tai root project), vi du:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_sentiment_db
DB_USER=postgres
DB_PASSWORD=postgres

LM_STUDIO_URL=http://localhost:1234/v1
# Ten model dung trong LM Studio (phai trung ten model dang load)
LLM_MODEL=Vistral-7B-ChatML-GGUF
# full = chuan hoa 4 muc (mac dinh); minimal = giu gan dung output LLM (tot cho Vistral tieng Viet)
# LLM_POSTPROCESS=minimal
LOG_LEVEL=INFO

# Chi can neu crawl YouTube (muc 2.3)
YOUTUBE_API_KEY=your_youtube_data_api_v3_key
```

---

### 2.3 Crawl binh luan YouTube

Dung **YouTube Data API v3** de lay binh luan theo tung video.

1. **API key:** trong Google Cloud Console bat **YouTube Data API v3**, tao key va them vao `.env`: `YOUTUBE_API_KEY=...` (xem mau muc 2.2).
2. **Danh sach video:** tao / chinh file `sentence_list/youtube_id_music.txt` — **moi dong mot** `video_id` (chuoi sau `v=` trong URL YouTube).
3. **Chay crawl** (tu thu muc goc repo):

```bash
cd sentence_list
python crawl_sentence.py
```

4. **Dau ra:** `sentence_list/raw_data/raw_comment.csv` — cot toi thieu: `video_id`, `comment` (van ban tho, chua lam sach).

**Luu y:** API co **quota**; script co retry va tam dung giua cac video. So binh luan toi da moi video co the chinh trong code crawl neu can.

---

### 2.4 Tien xu ly (raw -> clean)

Chuyen `raw_comment.csv` thanh `clean_comment.csv` phuc vu encode / train / (tuy chon) nen du lieu RAG.

1. **Dau vao mac dinh:** `sentence_list/raw_data/raw_comment.csv` (dung format sau crawl: `video_id`, `comment`).
2. **Dau ra mac dinh:** `sentence_list/clean_data/clean_comment.csv` — them cot **`lang`** (`vi`, `en`, `mix`, …) sau khi phat hien ngon ngu.

**Cac buoc xu ly chinh (tom tat):**

- Loai / rut gon link, emoji, ky tu dac biet; chuan hoa khoang trang va token.
- Phat hien ngon ngu (uu tien tieng Viet / tieng Anh).
- Stopword + **stemming / lemmatize** theo ngon ngu (tieng Anh co stem; tieng Viet giu tu sau loc stopword).
- Loai hang spam / dong trong / khong ho tro ngon ngu.

**Chay mac dinh:**

```bash
cd sentence_list
python preprocessing_comment.py
```

Ket thuc terminal se in so dong **giu lai** vs **loai**. Neu can duong dan khac, trong Python goi `process_file(input_path, output_path)` tu module tien xu ly.

**NLTK:** neu gap loi thieu corpus (vi du WordNet), cai `nltk` va tai du lieu mot lan, vi du:

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 3) Chuan bi du lieu va model (`experiments.py`)

### 3.1 Du lieu comment da lam sach

File dau vao encode / train: `sentence_list/clean_data/clean_comment.csv` (**tao tu muc 2.4**, hoac ban tu tao dung dinh dang tuong duong).

Can toi thieu cot:

- `comment`
- `lang` (khuyen nghi — duoc tao san khi chay tien xu ly mac dinh)
- `video_id` (khuyen nghi neu nguon tu YouTube, de trace lai nguon)

### 3.2 Chia du lieu khi train (7 / 2 / 1)

Khi chay `experiments.py train` (hoac `all`), du lieu da ma hoa duoc chia **stratified**:

- **Train 70%** — fit `StandardScaler` + fit SVM
- **Validation 20%** — dieu chinh/danh gia trong bao cao
- **Test 10%** — danh gia cuoi, confusion matrix, classification report

Moi tap giu ty le lop gan dung dong de (stratify theo nhan).

### 3.3 Lenh `experiments.py`

Tat ca encode + train + xuat ket qua nam trong **mot** file: `sentence_list/experiments.py`.

**Encode** (tao file trong `sentence_list/encoded_data/`):

```bash
# Ca 3 encoder
python sentence_list/experiments.py encode --method all

# Tung encoder
python sentence_list/experiments.py encode --method tfidf
python sentence_list/experiments.py encode --method doc2vec
python sentence_list/experiments.py encode --method xlmroberta
```

**Train** (can file `.pkl` da encode tuong ung; neu thieu phuong phap se bo qua va log canh bao):

```bash
python sentence_list/experiments.py train
```

**Encode + train mot len:**

```bash
python sentence_list/experiments.py all
```

**Luu y thoi gian:** buoc `train` tinh **learning curve** (5-fold CV tren tap train) cho moi encoder — lan dau co the lau, dac biet tren CPU.

### 3.4 Dau ra sau khi train

| Duong dan | Noi dung |
|-----------|----------|
| `sentence_list/encoded_data/*.pkl` | Ma tran dac trung + nhan + metadata |
| `sentence_list/results/svm_models/svm_model_tf_idf.pkl` (va `doc2vec`, `xlm_roberta`) | SVM + scaler + encoder metrics |
| `sentence_list/results/svm_comparison_metrics.csv` | Accuracy, Precision, Recall, F1 (macro) cho **train / val / test** + thong so mo hinh |
| `sentence_list/results/svm_comparison_summary.txt` | Bao cao van ban, phan bo nhan tung tap, classification report (test) |
| `sentence_list/results/training_run_report.json` | Tham so SVM/scaler/split/learning curve + ket qua chi tiet theo phuong phap |
| `sentence_list/plots/01_metrics_train_val_test.png` | So sanh metric 3 tap |
| `sentence_list/plots/02_time_params.png` | Thoi gian encode/SVM + tham so |
| `sentence_list/plots/03_confusion_matrices_test.png` | Confusion matrix (test) |
| `sentence_list/plots/04_learning_curves.png` | Learning curve F1-macro |
| `sentence_list/plots/05_metrics_heatmap_test.png` | Heatmap metric (test) |
| `sentence_list/plots/06_accuracy_splits.png` | Accuracy train / val / test |

---

## 4) Database va RAG (bat buoc truoc inference day du)

De pipeline suy luan dung kien truc (comment -> encode -> truy van DB embedding/RAG -> tong hop sentiment -> LLM),
ban can khoi tao DB va nap embedding truoc.

Khoi tao schema:

```bash
python sentence_list/setup_database.py
```

Nap embedding vao database:

```bash
python sentence_list/populate_embeddings.py
```

---

## 5) Chay he thong inference (CLI)

### 5.1 Phan tich 1 comment

```bash
python sentence_list/main.py "Bai hat nay rat hay!"
```

### 5.2 Batch tu file

File input: moi dong la 1 comment.

```bash
python sentence_list/main.py --batch input.txt --output results.json
```

### 5.3 Interactive mode

```bash
python sentence_list/main.py --interactive
```

### 5.4 Compare hieu nang 3 encoder

```bash
python sentence_list/main.py --compare-encoders input.txt --no-llm
```

Tra ve:

- du doan cho moi encoder tren tung sample
- `avg_encode_ms` de so sanh toc do

---

## 6) Chay bang Streamlit (khuyen nghi)

Ung dung UI da co san tai: `sentence_list/streamlit_app.py`

### 6.1 Cai them Streamlit (neu chua co)

```bash
pip install streamlit
```

(Hoac da co trong `requirements.txt`.)

### 6.2 Chuan bi du lieu/model truoc khi mo UI

Neu chua tao artifacts, chay:

```bash
python sentence_list/experiments.py all
python sentence_list/setup_database.py
python sentence_list/populate_embeddings.py
```

### 6.3 Chay app

Tu thu muc root du an:

```bash
streamlit run sentence_list/streamlit_app.py
```

Neu gap loi import tren mot so moi truong, thu chay:

```bash
cd sentence_list
streamlit run streamlit_app.py
```

Mac dinh app se:

- Nap `inference_pipeline` (co RAG + sentiment module)
- Cho phep bat/tat giai thich LLM o sidebar
- Phan tich tung binh luan theo giao dien chat

---

## 7) Tu dien (`clean_dict`)

Tat ca file tu dien da duoc chuan hoa vao:

- `sentence_list/clean_dict/vi_dict.csv`
- `sentence_list/clean_dict/en_dict.csv`

Tao/cap nhat tu dien:

```bash
python sentence_list/dictionary_tools.py
```

---

## 8) Luong suy luan dung theo kien truc

Khi phan tich 1 comment, he thong xu ly theo thu tu:

1. Preprocessing comment
2. Encode comment (tfidf/doc2vec/xlmroberta)
3. Truy van embedding trong database (RAG retrieve context)
4. Tong hop ket qua sentiment (ML + RAG context)
5. Dua ket qua + context vao LLM de sinh giai thich ngu nghia
6. Tra output cuoi (cam xuc + giai thich)

---

## 9) Luu y quan trong

- Neu gap `ImportError: cannot import name 'triu' from 'scipy.linalg'` khi import `gensim`: do **gensim 4.3.1 (va mot so ban cu)** khong tuong thich **SciPy 1.12 tro len**. Chay `pip install -r requirements.txt` (gensim da len **4.3.3**) hoac: `pip install "gensim>=4.3.3"`.
- Model SVM hien tai co the duoc train bang version `scikit-learn` khac version dang infer.
  Neu gap warning `InconsistentVersionWarning`, nen dong bo version sklearn giua moi truong train/infer.
- Tren Windows terminal cp1252, output Unicode co the gay loi; he thong da duoc dieu chinh output JSON an toan ASCII cho cac lenh benchmark.
- Lan dau benchmark co the cham neu can rebuild `tfidf_vectorizer.pkl` hoac `doc2vec_model.model`.
- Sau khi doi `clean_comment.csv`, nen chay lai `experiments.py all` de dong bo encoded data + SVM + bieu do.

---

## 10) Checklist day du (tu dau den cuoi)

**A. Moi truong**

1. Tao venv, `pip install -r requirements.txt`.
2. File `.env`: PostgreSQL (neu RAG), `LM_STUDIO_URL` + `LLM_MODEL` (neu LLM), `YOUTUBE_API_KEY` (neu crawl).

**B. Du lieu**

3. (Neu can) Chinh `youtube_id_music.txt` -> `cd sentence_list` -> `python crawl_sentence.py` -> co `raw_data/raw_comment.csv`.
4. `python preprocessing_comment.py` (trong `sentence_list`) -> co `clean_data/clean_comment.csv`.

**C. Huan luyen & artifact ML**

5. `python sentence_list/experiments.py all` (tu root) -> encoded data, SVM, metrics, plots, JSON.

**D. RAG / co so du lieu**

6. `python sentence_list/setup_database.py` (xac nhan / tao bang).
7. `python sentence_list/populate_embeddings.py` (nap vector; can buoc 5 da chay).

**E. Tu dien (tuy chon)**

8. `python sentence_list/dictionary_tools.py` (cap nhat `clean_dict/`).

**F. Suy luan**

9. Mo LM Studio, load dung model trung `LLM_MODEL`.
10. `streamlit run sentence_list/streamlit_app.py` hoac `python sentence_list/main.py ...`.

**G. Sau khi doi du lieu**

11. Lap lai **B.4** (tien xu ly) -> **C.5** -> **D.7** neu muon DB RAG dong bo voi comment moi.
