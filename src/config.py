import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, "student_scores.xlsx")

# Danh sách môn tính điểm chính
CORE_SUBJECTS = ["Toán", "Lý", "Hóa", "Sinh", "Tin học", "Văn", "Địa", "Sử", "Anh"]

# Danh sách môn tự chọn đề xuất (Key tiếng Anh)
OPTIONAL_SUBJECTS = [
    "physics", "chemistry", "biology", "informatics",
    "history", "geography", "english"
]

# Ánh xạ Key tiếng Anh -> Tên môn tiếng Việt
SUBJECT_MAP_VN = {
    "physics": "Lý",
    "chemistry": "Hóa",
    "biology": "Sinh",
    "informatics": "Tin học",
    "history": "Sử",
    "geography": "Địa",
    "english": "Anh",
    "math": "Toán",
    "literature": "Văn"
}

# Tên nhóm cụm mờ năng lực
CLUSTER_NAMES = {
    0: "Tự nhiên",
    1: "Xã hội",
    2: "Ngoại ngữ"
}

# Trọng số tính Final Score tư vấn môn
WEIGHT_SUBJECT_SCORE = 0.6  # w1: Điểm trung bình môn
WEIGHT_CLUSTER_FIT = 0.3    # w2: Mức độ thuộc cụm (Membership)
WEIGHT_TREND_SCORE = 0.1    # w3: Xu hướng tiến bộ

FCM_FUZZINESS = 2.0
MEMBERSHIP_SCORE_SCALE = 4.0

# Danh mục tất cả khối thi THPT Quốc gia (Dùng 3 môn tiếng Việt)
ALL_EXAM_COMBINATIONS = {
    'A00': ['Toán', 'Lý', 'Hóa'],
    'A01': ['Toán', 'Lý', 'Anh'],
    'A02': ['Toán', 'Lý', 'Sinh'],
    'A03': ['Toán', 'Lý', 'Sử'],
    'A04': ['Toán', 'Lý', 'Địa'],
    'A05': ['Toán', 'Hóa', 'Sử'],
    'A06': ['Toán', 'Hóa', 'Địa'],
    'B00': ['Toán', 'Hóa', 'Sinh'],
    'B01': ['Toán', 'Sinh', 'Sử'],
    'B02': ['Toán', 'Sinh', 'Địa'],
    'B03': ['Toán', 'Sinh', 'Văn'],
    'B08': ['Toán', 'Sinh', 'Anh'],
    'C00': ['Văn', 'Sử', 'Địa'],
    'C01': ['Văn', 'Toán', 'Lý'],
    'C02': ['Văn', 'Toán', 'Hóa'],
    'C03': ['Văn', 'Toán', 'Sử'],
    'C04': ['Văn', 'Toán', 'Địa'],
    'C07': ['Văn', 'Sử', 'Lý'],
    'C08': ['Văn', 'Hóa', 'Sinh'],
    'C10': ['Văn', 'Sử', 'Hóa'],
    'C11': ['Văn', 'Địa', 'Hóa'],
    'D01': ['Toán', 'Văn', 'Anh'],
    'D07': ['Toán', 'Hóa', 'Anh'],
    'D08': ['Toán', 'Sinh', 'Anh'],
    'D09': ['Toán', 'Sử', 'Anh'],
    'D10': ['Toán', 'Địa', 'Anh'],
    'D11': ['Văn', 'Lý', 'Anh'],
    'D12': ['Văn', 'Hóa', 'Anh'],
    'D13': ['Văn', 'Sinh', 'Anh'],
    'D14': ['Văn', 'Sử', 'Anh'],
    'D15': ['Văn', 'Địa', 'Anh'],
    'X26': ['Toán', 'Anh', 'Tin học'],
    'X02': ['Toán', 'Văn', 'Tin học'],
    'X06': ['Toán', 'Lý', 'Tin học'],
    'X14': ['Toán', 'Sinh', 'Tin học'],
    'X10': ['Toán', 'Hóa', 'Tin học'],
    'X22': ['Toán', 'Địa', 'Tin học'],
    'X71': ['Văn', 'Sử', 'Tin học'],
}