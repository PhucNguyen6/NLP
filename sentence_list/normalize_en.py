import pandas as pd
import re
import os

def normalize_pos(pos_value):
    """
    Chuẩn hóa loại từ (part of speech)
    
    Ánh xạ các định dạng POS phức tạp thành các dạng chuẩn hóa:
    - verb: động từ
    - noun: danh từ
    - adjective: tính từ
    - adverb: trạng từ
    - preposition: giới từ
    - conjunction: liên từ
    - pronoun: đại từ
    - interjection: thán từ
    - past_participle: phân từ quá khứ
    - present_participle: phân từ hiện tại
    - other: khác
    
    Args:
        pos_value: giá trị POS từ CSV
        
    Returns:
        Giá trị POS đã chuẩn hóa
    """
    if pd.isna(pos_value) or pos_value == '':
        return ''
    
    pos_lower = str(pos_value).lower().strip()
    
    # Xóa dấu chấm và khoảng trắng thừa
    pos_clean = re.sub(r'[.\s&/]+', ' ', pos_lower).strip()
    
    # Past participle: p p, pp, imp p p, p p a
    if 'p p' in pos_clean or pos_clean == 'pp':
        return 'past_participle'
    
    # Present participle: p pr, pr vb n, pres part
    if 'p pr' in pos_clean or 'pr vb' in pos_clean or 'pres' in pos_clean:
        return 'present_participle'
    
    # Verb: v, v t, v i, vb n, verb
    if re.search(r'\bv\b|verb|vb', pos_clean):
        return 'verb'
    
    # Noun: n, noun, pl, pl of
    if re.search(r'\bn\b|noun|pl', pos_clean):
        return 'noun'
    
    # Adjective: a, adj, adjective
    if re.search(r'^a\b|adj|adjective|superl', pos_clean):
        return 'adjective'
    
    # Adverb: adv, adverb
    if 'adv' in pos_clean:
        return 'adverb'
    
    # Preposition: prep, preposition
    if 'prep' in pos_clean:
        return 'preposition'
    
    # Conjunction: conj, conjunction
    if 'conj' in pos_clean:
        return 'conjunction'
    
    # Pronoun: pron, pronoun
    if 'pron' in pos_clean:
        return 'pronoun'
    
    # Interjection: interj, interjection
    if 'interj' in pos_clean:
        return 'interjection'
    
    # Impersonal: imp
    if pos_clean == 'imp':
        return 'verb'
    
    # Nếu không khớp với bất kỳ danh mục nào
    if pos_clean:
        return 'other'
    return ''


def normalize_pos_column(csv_path, output_path=None):
    """
    Đọc CSV, làm sạch dữ liệu và chuẩn hóa cột pos
    
    Các bước làm sạch:
    1. Xóa dòng có word trống
    2. Làm sạch khoảng trắng thừa
    3. Chuẩn hóa Unicode
    4. Lọc từ quá ngắn hoặc quá dài
    5. Xóa dòng có definition_en trống
    6. Xóa dòng trùng lặp
    7. Chuẩn hóa POS values
    
    Args:
        csv_path: đường dẫn file CSV đầu vào
        output_path: đường dẫn file CSV đầu ra (mặc định ghi đè file gốc)
    """
    import unicodedata
    
    # Đọc CSV
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("LAM SACH DU LIEU TU DIEN TIENG ANH")
    print("=" * 70)
    
    print("\nTruoc khi lam sach:")
    print(f"Tong so tu: {len(df)}")
    print(f"Dong co pos trong: {df['pos'].isna().sum()}")
    print(f"Dong co word trong: {df['word'].isna().sum()}")
    
    # Bước 1: Xóa dòng có word trống
    initial_count = len(df)
    df = df.dropna(subset=['word'])
    df['word'] = df['word'].str.strip()
    df = df[df['word'] != '']
    removed = initial_count - len(df)
    print(f"\nBuoc 1: Xoa dong co word trong")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 2: Làm sạch khoảng trắng
    df['pos'] = df['pos'].fillna('').str.strip()
    df['definition_en'] = df['definition_en'].fillna('').str.strip()
    print(f"\nBuoc 2: Xoa khoang trang thua")
    print(f"Hoan tat")
    
    # Bước 3: Chuẩn hóa Unicode
    def normalize_unicode(text):
        if pd.isna(text) or text == '':
            return text
        return unicodedata.normalize('NFC', str(text))
    
    df['word'] = df['word'].apply(normalize_unicode)
    df['definition_en'] = df['definition_en'].apply(normalize_unicode)
    print(f"\nBuoc 3: Chuan hoa Unicode")
    print(f"Hoan tat")
    
    # Bước 4: Lọc từ quá ngắn (< 1 ký tự) hoặc quá dài (> 100 ký tự)
    before_filter = len(df)
    df = df[(df['word'].str.len() >= 1) & (df['word'].str.len() <= 100)]
    removed = before_filter - len(df)
    print(f"\nBuoc 4: Loc tu qua ngan hoac qua dai")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 5: Xóa dòng có definition_en trống
    before_def = len(df)
    df = df[df['definition_en'] != '']
    removed = before_def - len(df)
    print(f"\nBuoc 5: Xoa dong co definition_en trong")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 6: Xóa dòng trùng lặp (giữ lại dòng đầu tiên)
    before_dup = len(df)
    df = df.drop_duplicates(subset=['word', 'definition_en'], keep='first')
    removed = before_dup - len(df)
    print(f"\nBuoc 6: Xoa dong trung lap")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 7: Chuẩn hóa POS values
    df['pos'] = df['pos'].apply(normalize_pos)
    print(f"\nBuoc 7: Chuan hoa POS values")
    print(f"Hoan tat")
    
    # Thống kê sau chuẩn hóa
    print(f"\nSau khi lam sach:")
    print(f"Tong so tu: {len(df)}")
    print(f"Dong co pos trong: {(df['pos'] == '').sum()}")
    print(f"\nPhan bo loai tu:")
    for pos, count in df['pos'].value_counts().items():
        if pos:
            print(f"  {pos:<20}: {count:>6} dong")
        else:
            print(f"  (trong)             : {count:>6} dong")
    
    # Lưu kết quả
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False)
    print(f"\nDa luu ket qua vao: {output_path}")
    
    return df


if __name__ == "__main__":
    # Chuẩn hóa file en_dict.csv
    csv_file = "raw_dict/en_dict.csv"
    df = normalize_pos_column(csv_file, "clean_dict/en_dict.csv")
