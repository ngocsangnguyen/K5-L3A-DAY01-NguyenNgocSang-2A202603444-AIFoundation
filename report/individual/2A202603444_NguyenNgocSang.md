# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603444
- **Họ và tên:** Nguyễn Ngọc Sang
- **SEED** (in ở ô đầu notebook): 7431

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

 ## Bảng so sánh các mô hình

| Model | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 1.000 | 0.863 | 0.927 | 0.974 | 9.867 | 22912.742 | 0.045 | 0.067 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.978 | 0.937 | 0.957 | 1.000 | 13.580 | 27327.811 | 0.309 | 0.669 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.874 | 0.947 | 0.909 | 0.974 | 83.397 | 31.573 | 55.761 | 106.828 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.958 | 0.958 | 0.958 | 0.947 | 83.953 | 50.109 | 28.177 | 43.036 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.968 | 0.947 | 0.957 | 0.947 | 84.405 | 49.537 | 28.551 | 37.465 | 0.0 |

### Ghi chú

- Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
- train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
1. Dự đoán F1 của Logistic Regression

- **Tôi đoán:** F1 khoảng 0.85, Precision cao hơn Recall vì muốn hạn chế phân loại nhầm tin nhắn HAM thành SPAM.
- **Kết quả thực tế:** F1 test = 0.927, Precision = 1.000 và Recall = 0.863.
- **Vì sao lệch:** F1 thực tế cao hơn dự đoán. Tôi đã đánh giá thấp hiệu quả của Logistic Regression. Tuy nhiên, nhận định Precision cao hơn Recall là đúng.

2. So sánh Logistic Regression và MLP

- **Tôi đoán:** MLP có thể đạt F1 cao hơn nhờ lớp ẩn và ReLU, nhưng chưa chắc vì có nguy cơ overfitting.
- **Kết quả thực tế:** MLP đạt F1 = 0.957, cao hơn Logistic Regression (0.927).
- **Vì sao lệch:** Dự đoán đúng về xu hướng nhưng chưa xác định được mức cải thiện. Trong thí nghiệm này, MLP học được các tổ hợp từ tốt hơn và cho kết quả cao hơn.

3. So sánh LLM với MiniLM + LogReg

- **Tôi đoán:** LLM có thể vượt MiniLM + LogReg nhờ khả năng hiểu ngữ cảnh tốt hơn.
- **Kết quả thực tế:** Chưa có số liệu F1 của hai LLM trong dữ liệu bạn gửi để xác nhận dự đoán này. MiniLM + LogReg đạt F1 test = 0.958 và F1 SUB40 = 0.947.
- **Vì sao lệch:** Tôi đã dựa nhiều vào khả năng hiểu ngữ cảnh của LLM mà chưa tính đến ảnh hưởng của prompt, cách xử lý đầu ra và chất lượng dữ liệu. Cần có F1 thực tế của LLM trên cùng tập SUB40 mới kết luận được.
## 4. Phân tích 50 case khó

Trong 50 case khó, kết quả trước đó của MiniLM + MLP head (3c) cho thấy mô hình sai 19/50 case, tương đương accuracy 62%.

Các nhóm dễ gây nhầm lẫn:
- natural_spam: Spam sử dụng ngôn ngữ tự nhiên, lịch sự, giống tin nhắn thông thường.
- manipulation: Tin nhắn sử dụng tâm lý thúc giục, giả dạng người quen hoặc thông báo đáng tin cậy.
- talks_about_spam: Tin ham có nhắc đến từ khóa spam, khiến mô hình có thể phân loại nhầm.
Hai case cụ thể:

1. natural_spam: Tin nhắn giả dạng người quen, sử dụng lời chào thân thiện và đề nghị truy cập đường link. Mô hình có thể dự đoán HAM vì nội dung không chứa nhiều từ khóa spam rõ ràng.

2. talks_about_spam: Tin nhắn bình thường thảo luận về spam. Mô hình có thể dự đoán SPAM vì nhận diện từ khóa mà chưa hiểu ngữ cảnh.

Kết luận: Mô hình gặp khó khăn khi phải phân biệt ý nghĩa thực sự của câu thay vì chỉ nhận diện từ khóa.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

Đề #4: Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?

Solution: k quá nhỏ khiến mô hình nhạy với nhiễu và dễ overfit; k quá lớn làm mất ảnh hưởng của các điểm lân cận thực sự gần, khiến mô hình dự đoán theo lớp chiếm đa số và dễ underfit. Vì vậy cần thử nhiều giá trị k và chọn giá trị có F1 tốt nhất trên tập validation.

Output:
|index|k|F1|
|---|---|---|
|0|1|0\.901|
|1|5|0\.9091|
|2|25|0\.8812|

k = 1: Chỉ xét hàng xóm gần nhất nên dễ bị ảnh hưởng bởi nhiễu hoặc dữ liệu bất thường.

k = 5: Xét 5 hàng xóm gần nhất, giúp cân bằng giữa thông tin cục bộ và giảm nhiễu nên đạt F1 tốt nhất.

k = 25: Xét quá nhiều hàng xóm nên có thể bị ảnh hưởng bởi các điểm thuộc lớp đa số, làm giảm khả năng phân biệt spam và ham.

-> Trong ba giá trị đã thử, k = 5 là lựa chọn tốt nhất với F1 = 0.9091. K quá nhỏ dễ overfit, còn k quá lớn dễ underfit.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Lý do:
- p95 chỉ 0.669 ms, thấp hơn nhiều so với giới hạn 50 ms.
- F1 test đạt 0.957, cao hơn BoW + LogReg.
- F1 SUB đạt 1.000 trên tập SUB40.
- Chi phí dự đoán theo bảng là 0 USD/1.000 tin.
Đánh đổi: BoW + MLP có thời gian huấn luyện cao hơn LogReg và vẫn có thể gặp khó khăn với tin nhắn spam tinh vi hoặc cách viết mới. F1 SUB = 1.000 cũng chưa đảm bảo hiệu quả tương tự trên dữ liệu thực tế.

Phương án nâng cao: Kết hợp BoW + MLP với LLM theo Router. Mô hình MLP xử lý tin có xác suất rõ ràng, còn LLM chỉ xử lý những tin có xác suất nằm trong khoảng 0.3–0.7. Cách này giúp giảm chi phí gọi LLM nhưng vẫn tận dụng khả năng hiểu ngữ cảnh của LLM.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
### Lần 1: Logistic Regression
- **Tôi hỏi:** Cách xây dựng mô hình Logistic Regression để phân loại SMS spam.
- **AI trả lời:** Hướng dẫn vector hóa văn bản, tính sigmoid, huấn luyện và đánh giá mô hình.
- **Tôi sửa:** Kiểm tra code, hoàn thiện các phần TODO và chạy thử để xác nhận kết quả.

### Lần 2: Learning Rate
- **Tôi hỏi:** Learning rate ảnh hưởng thế nào đến quá trình huấn luyện?
- **AI trả lời:** Learning rate nhỏ khiến loss giảm chậm, learning rate lớn có thể gây mất ổn định.
- **Tôi sửa:** Đối chiếu với biểu đồ loss thực tế và điều chỉnh nhận xét theo kết quả quan sát.

### Lần 3: MLP và ReLU
- **Tôi hỏi:** So sánh Logistic Regression với MLP khi có và không có ReLU.
- **AI trả lời:** Giải thích vai trò của hàm kích hoạt phi tuyến và nguy cơ overfitting.
- **Tôi sửa:** Đối chiếu F1 và validation loss để đánh giá hiệu quả của từng mô hình.

### Lần 4: Embedding và kNN
- **Tôi hỏi:** Embedding, cosine similarity và kNN hoạt động như thế nào?
- **AI trả lời:** Hướng dẫn biểu diễn ngữ nghĩa văn bản và tìm các tin nhắn tương tự.
- **Tôi sửa:** Kiểm tra kết quả trên các tin nhắn khó và so sánh với Bag-of-Words.

### Lần 5: LLM
- **Tôi hỏi:** Cách xây dựng prompt và sử dụng LLM để phân loại SMS.
- **AI trả lời:** Đề xuất prompt, cách xử lý nhãn và phương pháp Few-shot.
- **Tôi sửa:** Kiểm tra đầu ra, điều chỉnh cách phân tích nhãn và so sánh chất lượng với chi phí API.

### Lần 6: Benchmark và lựa chọn mô hình
- **Tôi hỏi:** Phân tích kết quả benchmark và lựa chọn mô hình đáp ứng yêu cầu p95 dưới 50 ms.
- **AI trả lời:** So sánh F1, tốc độ xử lý, độ trễ p95 và chi phí của các mô hình.
- **Tôi sửa:** Đối chiếu số liệu thực nghiệm và lựa chọn phương án phù hợp với yêu cầu triển khai thực tế.

## Điều tôi hiểu ra hôm nay

Qua bài thực hành, tôi nhận ra không phải cứ dùng mô hình phức tạp hơn thì kết quả sẽ tốt hơn. Ví dụ, BoW + MLP có F1 khá cao và tốc độ xử lý nhanh hơn nhiều so với các mô hình sử dụng MiniLM. Tôi cũng hiểu rằng khi đưa mô hình vào sử dụng thực tế, ngoài độ chính xác còn phải quan tâm đến tốc độ và chi phí. Quan trọng nhất, những gợi ý từ AI chỉ giúp tôi có hướng giải quyết; tôi vẫn cần tự chạy code, kiểm tra kết quả và điều chỉnh khi kết quả không giống dự đoán ban đầu.