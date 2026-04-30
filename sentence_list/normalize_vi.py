import pandas as pd
import numpy as np
import re

def normalize_vi_pos(pos_value):
    """
    Chuẩn hóa loại từ tiếng Việt
    
    Args:
        pos_value: giá trị part_of_speech từ CSV
        
    Returns:
        Giá trị part_of_speech đã chuẩn hóa
    """
    if pd.isna(pos_value) or str(pos_value).strip() == '':
        return ''
    
    pos_str = str(pos_value).strip()
    pos_lower = pos_str.lower()
    
    # Danh từ
    if 'danh tu' in pos_lower or 'danh từ' in pos_lower or 'noun' in pos_lower:
        return 'danh_tu'
    
    # Động từ
    if 'dong tu' in pos_lower or 'động từ' in pos_lower or 'verb' in pos_lower:
        return 'dong_tu'
    
    # Tính từ
    if 'tinh tu' in pos_lower or 'tính từ' in pos_lower or 'adj' in pos_lower:
        return 'tinh_tu'
    
    # Adverb / Phó từ / Phụ từ
    if 'phu tu' in pos_lower or 'phụ từ' in pos_lower or 'pho tu' in pos_lower or 'phó từ' in pos_lower or 'adv' in pos_lower:
        return 'phu_tu'
    
    # Conjunction / Kết từ / Liên từ
    if 'ket tu' in pos_lower or 'kết từ' in pos_lower or 'lien tu' in pos_lower or 'liên từ' in pos_lower or 'conj' in pos_lower:
        return 'ket_tu'
    
    # Pronoun / Đại từ
    if 'dai tu' in pos_lower or 'đại từ' in pos_lower or 'pron' in pos_lower:
        return 'dai_tu'
    
    # Auxiliary / Trợ từ
    if 'tro tu' in pos_lower or 'trợ từ' in pos_lower or 'aux' in pos_lower:
        return 'tro_tu'
    
    # Interjection / Cảm từ / Thán từ
    if 'cam tu' in pos_lower or 'cảm từ' in pos_lower or 'than tu' in pos_lower or 'thán từ' in pos_lower or 'interj' in pos_lower:
        return 'cam_tu'
    
    # Prefix / Tiền tố
    if 'tien to' in pos_lower or 'tiền tố' in pos_lower or 'prefix' in pos_lower:
        return 'tien_to'
    
    # Suffix / Hậu tố / Phụ tố
    if 'phu to' in pos_lower or 'phụ tố' in pos_lower or 'hau to' in pos_lower or 'hậu tố' in pos_lower or 'suffix' in pos_lower:
        return 'hau_to'
    
    # Nếu không khớp, trả về giá trị gốc (để kiểm tra)
    if pos_str:
        return pos_str
    return ''


def clean_vi_dict(input_path, output_path=None):
    """
    Làm sạch dữ liệu từ điển Việt
    
    Các bước làm sạch:
    1. Xóa dòng có word trống
    2. Làm sạch khoảng trắng thừa
    3. Chuẩn hóa Unicode và ký tự đặc biệt
    4. Lọc từ quá ngắn hoặc quá dài
    5. Xóa dòng có meaning trống
    6. Xóa dòng trùng lặp
    7. Chuẩn hóa POS values
    
    Args:
        input_path: đường dẫn file CSV đầu vào
        output_path: đường dẫn file CSV đầu ra
    """
    import unicodedata
    
    # Đọc CSV
    df = pd.read_csv(input_path)
    
    print("=" * 70)
    print("LAM SACH DU LIEU TU DIEN TIENG VIET")
    print("=" * 70)
    
    print("\nTruoc khi lam sach:")
    print(f"Tong so dong: {len(df)}")
    print(f"Dong co part_of_speech trong: {df['part_of_speech'].isna().sum()}")
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
    df['meaning'] = df['meaning'].fillna('').str.strip()
    df['example'] = df['example'].fillna('').str.strip()
    df['part_of_speech'] = df['part_of_speech'].fillna('')
    print(f"\nBuoc 2: Xoa khoang trang thua")
    print(f"Hoan tat")
    
    # Bước 3: Chuẩn hóa Unicode
    def normalize_unicode(text):
        if pd.isna(text) or text == '':
            return text
        return unicodedata.normalize('NFC', str(text))
    
    df['word'] = df['word'].apply(normalize_unicode)
    df['meaning'] = df['meaning'].apply(normalize_unicode)
    df['example'] = df['example'].apply(normalize_unicode)
    print(f"\nBuoc 3: Chuan hoa Unicode")
    print(f"Hoan tat")
    
    # Bước 4: Lọc từ quá ngắn (< 1 ký tự) hoặc quá dài (> 100 ký tự)
    before_filter = len(df)
    df = df[(df['word'].str.len() >= 1) & (df['word'].str.len() <= 100)]
    removed = before_filter - len(df)
    print(f"\nBuoc 4: Loc tu qua ngan hoac qua dai")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 5: Xóa dòng có meaning trống
    before_meaning = len(df)
    df = df[df['meaning'] != '']
    removed = before_meaning - len(df)
    print(f"\nBuoc 5: Xoa dong co meaning trong")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 6: Xóa dòng trùng lặp (giữ lại dòng đầu tiên)
    before_dup = len(df)
    df = df.drop_duplicates(subset=['word', 'meaning'], keep='first')
    removed = before_dup - len(df)
    print(f"\nBuoc 6: Xoa dong trung lap")
    print(f"Xoa: {removed} dong, Con lai: {len(df)} dong")
    
    # Bước 7: Chuẩn hóa POS values
    df['part_of_speech'] = df['part_of_speech'].apply(normalize_vi_pos)
    print(f"\nBuoc 7: Chuan hoa POS values")
    print(f"Hoan tat")
    
    # Thống kê sau chuẩn hóa
    print(f"\nSau khi lam sach:")
    print(f"Tong so dong: {len(df)}")
    print(f"Dong co part_of_speech trong: {(df['part_of_speech'] == '').sum()}")
    print(f"\nPhan bo loai tu:")
    for pos, count in df['part_of_speech'].value_counts().items():
        if pos:
            print(f"  {pos:<15}: {count:>6} dong")
        else:
            print(f"  (trong)         : {count:>6} dong")
    
    # Lưu kết quả
    if output_path is None:
        output_path = input_path
    
    df.to_csv(output_path, index=False)
    print(f"\nDa luu ket qua vao: {output_path}")
    return df


if __name__ == "__main__":
    # Làm sạch file vi_dict.csv
    input_file = "raw_dict/vi_dict.csv"
    output_file = "clean_dict/vi_dict.csv"
    
    df = clean_vi_dict(input_file, output_file)
