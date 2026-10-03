# Student Career Fuzzy

Hệ thống phân tích năng lực học sinh và tư vấn môn thi THPT dựa trên điểm số. Pipeline kết hợp tiền xử lý, feature engineering, Fuzzy C-Means (FCM), mô hình suy luận mờ Sugeno theo cấu trúc MANFIS, recommender môn học và ánh xạ tổ hợp xét tuyển.

Ứng dụng cung cấp hai cách sử dụng:

- **Tra cứu theo danh sách lớp:** xem kết quả đã tính cho từng học sinh.
- **Nhập điểm trực tiếp:** nhập điểm ba giai đoạn của 9 môn, xem membership FCM và kết quả xếp hạng môn từ MANFIS.

## 1. Mục tiêu nghiệp vụ

Hệ thống hỗ trợ trả lời ba câu hỏi:

1. Học sinh có xu hướng mạnh hơn ở miền nào: **Tự nhiên**, **Xã hội** hay **Ngoại ngữ**?
2. Học sinh có mức độ giao thoa giữa nhiều miền hay không?
3. Hai môn tự chọn và các tổ hợp xét tuyển nào phù hợp với hồ sơ điểm hiện tại?

FCM không gán mỗi học sinh vào đúng một nhóm cứng. Mỗi học sinh có một vector membership gồm ba giá trị trong khoảng `[0, 1]`, và tổng ba giá trị bằng `1`.

Ví dụ:

```text
Nhóm Tự nhiên : 0.20
Nhóm Xã hội   : 0.35
Nhóm Ngoại ngữ: 0.45
```

Hồ sơ này nghiêng về Ngoại ngữ nhưng vẫn có giao thoa đáng kể với Xã hội.

## 2. Kiến trúc xử lý

Pipeline chính nằm trong `src/main.py` và gồm sáu bước:

```text
Excel đầu vào
    |
    v
Tiền xử lý và xử lý giá trị thiếu
    |
    v
Tính điểm trung bình, xu hướng và điểm 3 miền
    |
    v
Chuẩn hóa, FCM và đánh giá mô hình
    |
    v
Membership, hồ sơ cụm và đề xuất môn bằng baseline
    |
    v
Huấn luyện/đánh giá MANFIS, đề xuất tổ hợp + giao diện Streamlit
```

### 2.0. Luồng chạy tổng quát

Lệnh `src/main.py` là điểm vào chính và chạy tuần tự toàn bộ pipeline:

1. Đọc Excel và làm sạch điểm.
2. Tạo điểm trung bình, xu hướng và ba điểm miền năng lực.
3. Chuẩn hóa dữ liệu, chạy FCM và tính các chỉ số phân cụm.
4. Xuất membership, centroid, hồ sơ cụm và đề xuất Top 2 baseline.
5. Tạo tập nhãn huấn luyện từ kết quả baseline, huấn luyện/đánh giá MANFIS và sinh dự đoán riêng.
6. Ánh xạ dự đoán MANFIS thành tối đa bốn tổ hợp xét tuyển cho mỗi học sinh.

Các file trong `data/processed/` là dữ liệu trung gian và kết quả sinh tự động. Không sửa thủ công các file này vì lần chạy pipeline tiếp theo sẽ ghi đè chúng.

### 2.1. Tiền xử lý

Module: `src/data/preprocessor.py`

Hệ thống đọc file Excel bằng `openpyxl`, bỏ hai dòng header đầu và trích xuất:

- Mã học sinh.
- Họ tên.
- Lớp.
- Điểm Toán, Lý, Hóa, Sinh, Tin học, Văn, Địa, Sử, Anh ở ba giai đoạn:
  - Lớp 10.
  - Lớp 11.
  - Học kỳ 1 lớp 12.

Giá trị thiếu được thay bằng median của chính cột điểm. Nếu cả cột không có giá trị hợp lệ, hệ thống dùng `0.0`.

Các điểm `0` khi tính trung bình môn được xem là dữ liệu thiếu để không kéo giảm điểm do chưa có dữ liệu.

#### Schema Excel đầu vào

Sau khi bỏ hai dòng header, chương trình dùng vị trí cột cố định:

- Cột thông tin: lớp ở index `1`, họ tên ở index `2`, mã học sinh ở index `3`.
- Điểm lớp 10 bắt đầu ở index `4`.
- Điểm lớp 11 bắt đầu ở index `17`.
- Điểm học kỳ 1 lớp 12 bắt đầu ở index `30`.
- Trong mỗi giai đoạn, thứ tự 9 môn là: Toán, Vật lý, Hóa học, Sinh học, Tin học, Ngữ văn, Địa lý, Lịch sử, Tiếng Anh.

Vì vậy, dữ liệu sạch có 30 cột: 3 cột thông tin và 27 cột điểm. Nếu thay đổi thứ tự hoặc số lượng cột trong Excel, cần cập nhật `subject_offsets` và `periods_start_idx` trong `src/data/preprocessor.py`.

### 2.2. Feature engineering

Module: `src/features/feature_engineering.py`

Với mỗi môn, hệ thống tính:

```text
subject_avg = mean(điểm lớp 10, điểm lớp 11, điểm lớp 12 HK1)
subject_trend = điểm lớp 12 HK1 - mean(điểm lớp 10, điểm lớp 11)
```

Ba miền năng lực được tính như sau:

```text
natural_score  = mean(Toán, Lý, Hóa, Sinh, Tin học)
social_score   = mean(Văn, Sử, Địa)
english_score  = Anh
```

Các điểm miền được chia cho `10` để đưa về `[0, 1]`. Sau đó, hệ thống trừ trung bình ba miền của từng học sinh để tập trung vào **miền nổi trội tương đối**:

```text
relative_domain = scaled_domain - mean(natural, social, english)
```

Do đó, vector ba miền có tổng gần bằng `0`. Đây là chủ ý thiết kế: học sinh có điểm tuyệt đối cao ở cả ba miền vẫn được xem là cân bằng nếu không miền nào nổi trội hơn.

File kết quả là `data/processed/normalized_scores.csv`.

## 3. Mô hình Fuzzy C-Means

Module: `src/clustering/fcm.py`

### 3.1. Huấn luyện FCM

FCM được chạy với:

- Số cụm: `3`.
- Hệ số mờ: `m = 2.5`, khai báo tại `src/config.py`.
- Sai số hội tụ: `0.005`.
- Số vòng lặp tối đa: `1000`.
- Seed: `42`.

FCM gốc tự học các centroid từ dữ liệu. Ma trận membership raw do `scikit-fuzzy` trả về được dùng cho FPC, FPE, Xie-Beni và các chỉ số phân cụm.

### 3.2. Nhãn nghiệp vụ

Ba nhãn hiển thị cố định là:

- `Nhóm Tự nhiên`.
- `Nhóm Xã hội`.
- `Nhóm Ngoại ngữ`.

FCM bản chất là không giám sát nên thứ tự cluster `0, 1, 2` không có ý nghĩa nghiệp vụ. Code ánh xạ các centroid sang ba nhãn dựa trên miền trội tương đối, sau đó bảo đảm mỗi nhãn được sử dụng một lần.

File `cluster_label_diagnostics.csv` ghi lại:

- Chỉ số cluster gốc.
- Nhãn được gán.
- Miền có giá trị centroid cao nhất.
- Độ chênh so với miền đứng thứ hai.
- Tọa độ centroid ở ba chiều.

Nếu `Dominance_Margin` nhỏ, nhãn cụm cần được diễn giải thận trọng vì centroid không có miền trội mạnh.

### 3.3. Membership dùng cho tư vấn

Để nhãn nghiệp vụ không bị sai do centroid học có độ lớn khác nhau, giao diện và file membership sử dụng ba prototype hướng miền:

```text
Tự nhiên  = ( 2, -1, -1)
Xã hội    = (-1,  2, -1)
Ngoại ngữ = (-1, -1,  2)
```

Các vector được chuẩn hóa độ dài trước khi đo khoảng cách theo hướng. Hàm `calculate_membership_matrix` dùng softmax với `MEMBERSHIP_SCORE_SCALE` để chuyển điểm tương đồng thành membership. Vì vậy:

- Điểm ba miền bằng nhau cho membership gần `33.33%` mỗi nhóm.
- Tăng điểm Anh làm membership Ngoại ngữ tăng.
- Hai nhóm còn lại giảm dần thay vì bị gán cứng vào `0%` ngay lập tức.
- Tổng membership luôn bằng `1`.

Membership được làm mềm bằng softmax trên điểm miền tương đối với `MEMBERSHIP_SCORE_SCALE = 4.0`. File `membership.csv` và giao diện dùng membership theo prototype nghiệp vụ; `evaluation_metrics.csv` dùng membership raw do FCM sinh ra để đánh giá đúng thuật toán.

Trong giao diện, ngoài biểu đồ membership còn có:

- Membership cao nhất.
- Membership đứng thứ hai.
- Mức độ giao thoa: `1 - membership_lớn_nhất`.
- Entropy chuẩn hóa của phân bố membership.
- Khoảng cách theo hướng tới từng prototype.

## 4. Đánh giá mô hình

Các chỉ số trong `evaluation_metrics.csv` được tính trên **membership raw của FCM**, không phải membership prototype dùng cho lớp tư vấn. Điều này giữ đúng ý nghĩa đánh giá thuật toán FCM.

### Fuzzy Partition Coefficient

```text
FPC = sum(u_ij^2) / n
```

Với `c` cụm, FPC nằm trong khoảng `[1/c, 1]`. Với ba cụm:

- Gần `1`: phân cụm rõ, membership tập trung.
- Gần `1/3`: phân cụm rất mờ hoặc các cụm khó tách.

### Fuzzy Partition Entropy

```text
FPE = -sum(u_ij * log(u_ij)) / n
```

FPE raw nằm trong `[0, ln(c)]`. Code cũng xuất:

```text
fpe_normalized = fpe / ln(c)
```

Giá trị chuẩn hóa nằm trong `[0, 1]` và dễ đọc hơn:

- Gần `0`: membership ít mờ.
- Gần `1`: membership phân tán đều giữa các cụm.

Các chỉ số bổ sung:

- **Xie-Beni:** đánh giá độ chặt trong cụm và khoảng cách giữa centroid; càng thấp thường càng tốt.
- **Silhouette:** mức phù hợp của điểm với cụm được gán cứng; càng cao càng tốt.
- **Davies-Bouldin:** độ tương đồng giữa các cụm; càng thấp càng tốt.
- **Calinski-Harabasz:** tỷ lệ phân tán giữa cụm và trong cụm; thường càng cao càng tốt.

Không nên dùng một chỉ số duy nhất để kết luận mô hình tốt hay xấu. Cần xem đồng thời chỉ số, centroid, hồ sơ cụm và các ca kiểm thử nghiệp vụ.

## 5. Tư vấn môn học

Module: `src/counseling/subject_recommender.py`

Các môn tự chọn được xét:

- Vật lý.
- Hóa học.
- Sinh học.
- Tin học.
- Lịch sử.
- Địa lý.
- Tiếng Anh.

Điểm cuối mỗi môn được tính:

```text
Final Score =
    0.6 × Subject Score
  + 0.3 × Cluster Fit
  + 0.1 × Trend Score
```

Trong đó:

- `Subject Score`: điểm trung bình môn chia cho `10`.
- `Cluster Fit`: membership của miền tương ứng. Nếu dữ liệu không có cột membership đúng tên miền, hệ thống dùng điểm miền chuẩn hóa làm fallback.
- `Trend Score`: xu hướng được clip trong khoảng `[-2, 2]` rồi đưa về `[0, 1]`.

Kết quả được sắp xếp giảm dần và lấy hai môn đầu tiên.

### 5.1. MANFIS tích hợp FCM

Module: `src/counseling/manfis.py`

MANFIS dùng cấu trúc Sugeno bậc nhất. FCM khởi tạo các luật mờ; mỗi luật có một hàm đầu ra tuyến tính được ước lượng bằng weighted least squares. Pipeline thử số luật `2`, `3` và `4`, chọn cấu hình theo validation, đánh giá một lần trên test set, sau đó huấn luyện model triển khai trên toàn bộ dữ liệu.

Đầu vào gồm 17 đặc trưng:

- Điểm ba miền `natural_score`, `social_score`, `english_score`.
- Điểm trung bình và xu hướng của bảy môn tự chọn.

Hai đầu ra phân loại là `top1_subject` và `top2_subject`; hai đầu ra hồi quy là điểm tương ứng. Khi suy luận, Top 1 và Top 2 luôn là hai môn khác nhau, và điểm Top 1 không thấp hơn Top 2. Các tổ hợp không phải đầu ra trực tiếp của MANFIS: hệ thống tính lại chúng bằng `combination_mapper.py` từ hai môn MANFIS đề xuất và bảng điểm.

#### Nguồn nhãn và đánh giá

`src/data/normalize_counseling_labels.py` chuyển kết quả recommender hiện tại thành `data/raw/manfis_pseudo_labels.csv`. MANFIS dùng các nhãn môn và điểm Top 1/Top 2 trong file này; dữ liệu được ghép với `features.csv` bằng `student_id`. Đây là nguồn nhãn huấn luyện của phiên bản hiện tại, vì vậy kết quả thể hiện mức độ mô hình học được quy tắc tư vấn đang dùng. Các metric hold-out đo khả năng tái tạo những nhãn này; chúng chưa phải thước đo dự báo điểm thi hoặc kết quả tuyển sinh.

Tập được chia xấp xỉ 50:20:30 theo Top 1, với seed `42`. Do một số môn có rất ít nhãn, số lượng từng lớp có thể khiến tỷ lệ chia thực tế chênh nhẹ. Chỉ số test được ghi trong `manfis_metrics.csv`; pipeline cũng lưu model huấn luyện toàn bộ dữ liệu để phục vụ dự đoán trong ứng dụng.

## 6. Ánh xạ tổ hợp xét tuyển

Module: `src/counseling/combination_mapper.py`

Hệ thống lấy Toán, Văn và hai môn tự chọn được đề xuất để tìm các tổ hợp phù hợp trong danh mục cấu hình tại `src/config.py`.

Ngoài các tổ hợp A, B, C, D hiện có, hệ thống hỗ trợ các tổ hợp có Tin học:

| Mã | Tổ hợp môn |
| --- | --- |
| `X26` | Toán, Tiếng Anh, Tin học |
| `X02` | Toán, Ngữ văn, Tin học |
| `X06` | Toán, Vật lí, Tin học |
| `X14` | Toán, Sinh học, Tin học |
| `X10` | Toán, Hóa học, Tin học |
| `X22` | Toán, Địa lí, Tin học |
| `X71` | Ngữ văn, Lịch sử, Tin học |

Trong code, các tên môn rút gọn được dùng khi tính toán là `Toán`, `Văn`, `Lý`, `Hóa`, `Sinh`, `Tin học`, `Địa`, `Sử` và `Anh`.

Với mỗi tổ hợp hợp lệ, hệ thống tính tổng điểm trung bình các môn và sắp xếp giảm dần. Kết quả cuối cùng chứa tối đa bốn tổ hợp đề xuất hàng đầu.

Đây là gợi ý kỹ thuật dựa trên điểm số, không thay thế điều kiện tuyển sinh chính thức của từng trường đại học.

## 7. Cấu trúc thư mục

```text
.
├── app.py                              # Giao diện Streamlit
├── requirements.txt                    # Dependency Python
├── README.md                           # Tài liệu dự án
├── src/
│   ├── main.py                         # Pipeline FCM, tư vấn và MANFIS
│   ├── config.py                       # Cấu hình, trọng số và tổ hợp
│   ├── data/
│   │   ├── preprocessor.py             # Đọc và làm sạch Excel
│   │   └── normalize_counseling_labels.py # Chuẩn hóa nhãn tư vấn cho MANFIS
│   ├── features/
│   │   └── feature_engineering.py      # Tạo đặc trưng và chuẩn hóa
│   ├── clustering/
│   │   ├── fcm.py                      # Huấn luyện FCM và membership
│   │   └── evaluation.py               # FPC, FPE và chỉ số đánh giá
│   ├── counseling/
│   │   ├── subject_recommender.py      # Chọn Top 2 môn
│   │   ├── combination_mapper.py       # Ánh xạ tổ hợp
│   │   └── manfis.py                   # FCM khởi tạo luật và suy luận Sugeno
│   └── visualization/
│       ├── radar.py                     # Radar năng lực
│       └── membership_chart.py          # Biểu đồ membership
├── data/
│   ├── raw/                             # Excel đầu vào và nhãn huấn luyện MANFIS đã chuẩn hóa
│   └── processed/                       # CSV sinh bởi pipeline
├── results/                             # Hình ảnh và kết quả phân tích
└── notebooks/                           # Notebook nghiên cứu
```

Các thư mục `results/figures/`, `results/clustering/` và `results/radar/` dành cho hình ảnh hoặc kết quả phân tích bổ sung; pipeline chính không bắt buộc phải có file trong các thư mục này.

## 8. Cài đặt trên Windows

Mở PowerShell tại thư mục gốc dự án:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy trực tiếp executable trong `venv` mà không cần activate.

## 9. Chuẩn bị dữ liệu đầu vào

Đặt file dữ liệu tại:

```text
data/raw/student_scores.xlsx
```

File Excel hiện tại cần giữ cấu trúc cột mà `preprocessor.py` đang đọc:

- Hai dòng đầu là header.
- Thông tin học sinh nằm ở các cột tên, lớp và mã học sinh theo cấu trúc hiện tại.
- Mỗi giai đoạn có các cột môn theo thứ tự Toán, Lý, Hóa, Sinh, Tin học, Văn, Địa, Sử, Anh.
- Có đủ ba giai đoạn: lớp 10, lớp 11 và lớp 12 học kỳ 1.

Nếu thay đổi bố cục Excel, cần cập nhật `subject_offsets` và `periods_start_idx` trong `src/data/preprocessor.py`.

## 10. Chạy pipeline

Từ thư mục gốc:

```powershell
.\venv\Scripts\python.exe src\main.py
```

Hoặc nếu đã activate môi trường:

```powershell
python src/main.py
```

Pipeline sẽ tạo hoặc cập nhật các file trong `data/processed/`.

Có thể chạy từng module để kiểm tra riêng từng giai đoạn, nhưng thứ tự nên là:

```powershell
python src\data\preprocessor.py
python src\features\feature_engineering.py
python src\clustering\fcm.py
python src\counseling\subject_recommender.py
```

Trong sử dụng thông thường, nên chạy `python src\main.py` để bảo đảm mọi file đầu ra được đồng bộ cùng một lần xử lý dữ liệu.

### 10.1. Grid search FCM

Để chạy kiểm tra nhiều cấu hình FCM và lưu kết quả so sánh:

```powershell
python src\clustering\grid_search_fcm.py
```

Kết quả được lưu tại `data/processed/fcm_grid_search_results.csv`.

Grid search hiện thử các số cụm `[2, 3, 4, 5]`, các giá trị hệ số mờ `[1.2, 1.3, 1.5, 1.7, 2.0]` và bốn seed `[10, 42, 100, 2024]`. Kết quả được sắp xếp theo Silhouette giảm dần, sau đó Xie-Beni tăng dần.

## 11. Chạy ứng dụng

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ Streamlit hiển thị trong terminal, thường là:

```text
http://localhost:8501
```

Nếu đã chạy pipeline sau khi mở ứng dụng, hãy tải lại trang hoặc khởi động lại Streamlit để giao diện đọc các CSV mới nhất.

Giao diện có hai tab:

- **Tra cứu theo danh sách lớp:** lọc lớp, chọn học sinh, xem baseline và MANFIS, các tổ hợp tương ứng, biểu đồ radar và membership ba nhóm năng lực.
- **Nhập điểm trực tiếp:** nhập điểm lớp 10, lớp 11 và học kỳ 1 lớp 12 cho 9 môn, sau đó xem radar, membership FCM và xếp hạng MANFIS theo thời gian thực.

Tab nhập điểm yêu cầu `centroids.csv`, vì vậy cần chạy pipeline ít nhất một lần trước khi mở ứng dụng.

## 12. Các file đầu ra

| File | Nội dung |
| --- | --- |
| `cleaned_scores.csv` | Điểm sau tiền xử lý và xử lý missing values |
| `features.csv` | Điểm trung bình, trend và điểm ba miền |
| `normalized_scores.csv` | Vector đặc trưng tương đối đưa vào FCM |
| `centroids.csv` | Tọa độ centroid và nhãn ba nhóm |
| `membership.csv` | Membership dùng cho tư vấn và tra cứu |
| `cluster_label_diagnostics.csv` | Đối soát nhãn và miền trội của centroid |
| `cluster_profile_summary.csv` | Thống kê môn học theo nhóm |
| `cluster_class_distribution.csv` | Phân bố nhóm theo lớp |
| `evaluation_metrics.csv` | FPC, FPE, Xie-Beni và chỉ số bổ sung |
| `fcm_grid_search_results.csv` | Kết quả so sánh các cấu hình FCM |
| `top2_recommendations.csv` | Hai môn tự chọn được đề xuất |
| `final_counseling_results.csv` | Kết quả cuối cùng kèm tổ hợp xét tuyển |
| `manfis_pseudo_labels.csv` | Nhãn Top 1/Top 2 và điểm recommender đã chuẩn hóa để huấn luyện MANFIS |
| `manfis_model.pkl` | Model MANFIS dùng trong Streamlit |
| `manfis_metrics.csv` | Metric hold-out cho phân loại môn và sai số điểm trên nhãn huấn luyện hiện tại |
| `manfis_recommendations.csv` | Dự đoán Top 1/Top 2 từ MANFIS trên danh sách học sinh |
| `manfis_counseling_results.csv` | Dự đoán MANFIS sau khi ánh xạ sang tổ hợp xét tuyển |
| `minmax_scaler.pkl` | Scaler được lưu để tái sử dụng |

## 13. Kiểm thử nhanh

Kiểm tra cú pháp các module chính:

```powershell
python -m py_compile app.py src\clustering\fcm.py src\clustering\evaluation.py src\counseling\manfis.py src\data\normalize_counseling_labels.py src\main.py
```

Kiểm tra đầy đủ các module chính:

```powershell
python -m py_compile app.py src\config.py src\data\preprocessor.py src\data\normalize_counseling_labels.py src\features\feature_engineering.py src\clustering\fcm.py src\clustering\evaluation.py src\counseling\subject_recommender.py src\counseling\combination_mapper.py src\counseling\manfis.py src\main.py
```

Kiểm tra membership có tổng bằng `1`:

```powershell
.\venv\Scripts\python.exe -c "import pandas as pd, numpy as np; m=pd.read_csv('data/processed/membership.csv'); c=[x for x in m.columns if x.startswith('membership_')]; print(np.max(np.abs(m[c].sum(axis=1)-1)))"
```

Kiểm thử nghiệp vụ nên bao gồm:

- Ba miền điểm bằng nhau: membership gần `33.33%` mỗi nhóm.
- Tăng riêng điểm Anh: membership Ngoại ngữ tăng dần.
- Tăng riêng nhóm Toán-Lý-Hóa-Sinh-Tin học: membership Tự nhiên tăng dần.
- Tăng riêng nhóm Văn-Sử-Địa: membership Xã hội tăng dần.
- Hồ sơ lai giữa hai miền: membership của hai nhóm cùng ở mức đáng kể.

## 14. Giới hạn và diễn giải kết quả

- FCM là mô hình không giám sát; nhãn cụm phụ thuộc dữ liệu và cách tạo đặc trưng.
- Membership là độ thuộc mờ, không phải xác suất đỗ đại học hay xác suất thống kê.
- Prototype nghiệp vụ giúp nhãn nhất quán nhưng không thay thế việc kiểm định trên dữ liệu thực tế.
- MANFIS hiện được huấn luyện từ nhãn môn/điểm do recommender baseline tạo ra; báo cáo `manfis_metrics.csv` vì thế đánh giá khả năng tái tạo nhãn này. Để đánh giá dự báo kết quả thi hoặc tuyển sinh, cần bổ sung kết quả thi thật làm nhãn độc lập.
- Excel và form nhập liệu hiện hỗ trợ ba giai đoạn điểm: lớp 10, lớp 11 và học kỳ 1 lớp 12. Nếu yêu cầu nghiên cứu cần đủ năm học kỳ, cần cập nhật schema Excel và tiền xử lý trước khi huấn luyện lại.
- Nếu dữ liệu thiếu nhiều học sinh Xã hội nổi trội, cụm Xã hội sẽ kém ổn định dù hệ thống vẫn phải hiển thị đủ ba nhóm theo yêu cầu nghiệp vụ.
- Điểm đề xuất và tổ hợp chỉ mang tính tham khảo; cần đối chiếu quy chế tuyển sinh hiện hành.

## 15. Xử lý lỗi thường gặp

- **Thiếu `student_scores.xlsx`:** đặt đúng file tại `data/raw/student_scores.xlsx`.
- **Thiếu cột điểm bắt buộc:** kiểm tra số dòng header, vị trí ba nhóm cột điểm và thứ tự 9 môn trong Excel.
- **Ứng dụng báo thiếu CSV:** chạy pipeline trước bằng `python src\main.py`, sau đó tải lại trang Streamlit.
- **Đã thay đổi dữ liệu hoặc code feature:** luôn chạy lại pipeline để cập nhật `features.csv`, `centroids.csv`, `membership.csv` và các kết quả tư vấn.
- **Không có tổ hợp đề xuất:** tổ hợp chỉ được tạo khi Toán, Văn và các môn cần thiết nằm trong bốn môn mà bộ recommender cung cấp cho học sinh.

## 16. Cấu hình quan trọng

Các tham số chính nằm trong `src/config.py`:

```python
FCM_FUZZINESS = 2.5
MEMBERSHIP_SCORE_SCALE = 4.0
WEIGHT_SUBJECT_SCORE = 0.6
WEIGHT_CLUSTER_FIT = 0.3
WEIGHT_TREND_SCORE = 0.1
```

Khi thay đổi tham số FCM hoặc trọng số tư vấn, cần chạy lại `src/main.py` để cập nhật toàn bộ file đầu ra trước khi xem trên Streamlit.

## 17. Lưu ý sử dụng

Luôn chạy pipeline sau khi thay file Excel hoặc thay đổi logic feature engineering. Không chỉnh thủ công `centroids.csv` hoặc `membership.csv`, vì các file này sẽ bị ghi đè ở lần chạy tiếp theo. Kết quả là công cụ hỗ trợ phân tích và tư vấn, không thay thế đánh giá của giáo viên, chuyên gia hướng nghiệp hoặc thông tin tuyển sinh chính thức.
