# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603971
- **Họ và tên:** Phan Đào Minh Quân
- **SEED** (in ở ô đầu notebook): 3927

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000    0.84      0.913        0.919   
2. BoW + MLP (deep learning)         1.000    0.91      0.953        0.947   
3a. Encoder (MiniLM) + kNN           0.951    0.97      0.960        0.947   
3b. Encoder (MiniLM) + LogReg        0.979    0.95      0.964        0.947   
3c. Encoder (MiniLM) + MLP head      0.990    0.95      0.969        0.947   
4a. LLM nhỏ zero-shot                0.900    0.90      0.900        0.900   
4c. LLM nhỏ few-shot                 1.000    0.90      0.947        0.947   
4b. LLM lớn zero-shot                0.947    0.90      0.923        0.923   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.109      21556.234       0.065   
2. BoW + MLP (deep learning)         7.433      39391.554       0.162   
3a. Encoder (MiniLM) + kNN          85.372         13.730      51.611   
3b. Encoder (MiniLM) + LogReg       85.856         46.954      31.603   
3c. Encoder (MiniLM) + MLP head     86.344         46.476      33.618   
4a. LLM nhỏ zero-shot                0.000          0.992     495.393   
4c. LLM nhỏ few-shot                 0.000          0.951     329.872   
4b. LLM lớn zero-shot                0.000          1.208     326.179   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.157           0.0  
2. BoW + MLP (deep learning)          0.236           0.0  
3a. Encoder (MiniLM) + kNN           95.744           0.0  
3b. Encoder (MiniLM) + LogReg        63.215           0.0  
3c. Encoder (MiniLM) + MLP head      86.651           0.0  
4a. LLM nhỏ zero-shot              2819.737           0.0  
4c. LLM nhỏ few-shot               2593.633           0.0  
4b. LLM lớn zero-shot              2350.743           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

Dự đoán dl_curve (Thí nghiệm 2.A):

Đoán: Val loss giảm vài epoch đầu rồi đi ngang/tăng lại (overfit). F1 của MLP chỉ nhỉnh hơn LogReg một chút.

Thực tế: Val loss đạt đáy ở epoch 8 rồi tăng lại. F1 tăng từ 0.913 (LogReg) lên 0.953 (MLP).

Vì sao: Khớp với dự đoán. Bag-of-words có tính thưa và độc lập, ranh giới đã gần tuyến tính nên phần phi tuyến của MLP không mang lại đột phá quá lớn, mạng dễ ghi nhớ vẹt (overfit) nhanh chóng.

Dự đoán tradeoff (Biểu đồ Pareto):

Đoán: Trên đường Pareto có BoW+LogReg (nhanh nhất) và Embedding/LLM (ít lỗi). LLM nằm ngoài SLA 50ms.

Thực tế: LLM thực sự có p95 rất cao (vượt xa 50ms). Tuy nhiên, trên đường Pareto của tôi chỉ có BoW + LogReg và BoW + MLP.

Vì sao lệch: Do tập SUB 40 tin quá nhỏ (chênh 1 tin đã làm F1 đổi 2.5%), các phương pháp Embedding (3a, 3b, 3c) cho F1_SUB ngang bằng với BoW+MLP (0.947) nhưng độ trễ lại cao hơn rất nhiều (63-95ms so với 0.2ms), nên chúng bị BoW+MLP lấn át trên biểu đồ.

Dự đoán llm_fewshot (Thí nghiệm 4.B):

Đoán: F1 tăng nhẹ hoặc không đổi. Chi phí tăng 3-5 lần do prompt phình to.

Thực tế: F1 tăng từ 0.900 (zero-shot) lên 0.947 (few-shot).

Vì sao: Việc cung cấp ví dụ (in-context learning) giúp LLM nhỏ cân chỉnh lại ranh giới quyết định tốt hơn thay vì chỉ dựa vào pre-training, tuy nhiên đổi lại latency khá cao.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: Mô hình 1. BoW + LogReg (tụt F1 từ 0.919 xuống 0.378, Accuracy chỉ còn 54%) và 2. BoW + MLP (F1 tụt còn 0.450).

Nhóm case khó nhất với BoW:

Né từ khóa (Obfuscation): Đúng 17%. Ví dụ: Get your l0an appr0ved in 5 m1nutes, n0 credit check. Repl¥ L0AN. Tokenizer cắt sai hoặc sinh ra token lạ, BoW hoàn toàn mù do không có trong vocab.

Thao túng/Lừa đảo (Manipulation): Đúng 12%. Ví dụ: Hi Mum, I dropped my phone in the toilet... Can you send £200.... Không chứa các từ khóa spam cổ điển, BoW nhìn vào tưởng là tin nhắn bạn bè/người thân.

Cơ chế khiến model sai: BoW đánh giá văn bản bằng cách đếm các túi từ rời rạc. Nếu thiếu từ khóa (false negative) hoặc từ khóa nằm trong ngữ cảnh vô hại (false positive), mô hình sẽ hoàn toàn sụp đổ vì không có khả năng hiểu ngữ nghĩa toàn cục của câu.

## 5. Thử thách cá nhân

Đề bài: Đề #4 — Embedding: Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?

Cách làm: Viết vòng lặp for chạy hàm knn_predict trên tập test với các giá trị k khác nhau, sau đó gọi hàm metrics để in ra kết quả.

Kết quả:

k = 1: F1 = 0.955

k = 5: F1 = 0.960

k = 25: F1 = 0.913

Giải thích:

k quá nhỏ (k=1): Mô hình rất nhạy với nhiễu. Chỉ cần 1 tin nhắn rác bị gán nhãn nhầm hoặc 1 tin có vector vô tình nằm gần là quyết định bị đảo lộn (high variance).

k quá lớn (k=25): Mô hình bị "bias" về phía lớp đa số (lớp ham chiếm 87%). Bán kính lân cận quá rộng bao trùm luôn cả các tin ham, khiến spam bị lọt lưới rất nhiều (recall giảm mạnh).
## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 tin/s), SLA p95 < 50 ms/tin.

Phương án đề xuất: Hệ thống lai (Router).

Lớp 1 (Chốt chặn chính): Dùng 3b. Encoder (MiniLM) + LogReg (nếu tối ưu được phần cứng để p95 < 50ms) hoặc 2. BoW + MLP (hiện tại p95 cực thấp: 0.236ms). Các mô hình này cực rẻ, nhanh, sẽ tự động gán nhãn ngay cho các tin có xác suất rất cao (p < 0.2 là ham, p > 0.8 là spam).

Lớp 2 (Deep inspection): Đối với các tin "nhập nhằng" (0.2 ≤ p ≤ 0.8) hoặc nghi ngờ có prompt injection/obfuscation, ta route qua LLM nhỏ (zero-shot/few-shot).

Đánh đổi: LLM rất đắt và độ trễ cao (hàng ngàn ms). Việc dùng Router giúp ta đánh đổi một lượng rất nhỏ chi phí và độ trễ (chỉ chạy LLM trên <5% data) để cứu lại độ chính xác trên các case lừa đảo tinh vi mà BoW hay Embedding chịu thua.



## 7. AI log và điều đã học

4.4

Làm sao xử lý lỗi RateLimitError: 429 - Quota exceeded của API Groq/Gemini?

AI giải thích do Free Tier giới hạn 15 RPM, gọi 40 tin/batch sẽ bị chặn. Đề xuất dùng time.sleep hoặc bọc trong hàm retry_with_backoff.

Chạy qua hàm retry_with_backoff đã cung cấp sẵn để code tự động đợi và thử lại.

Trước đây em chỉ biết đánh giá mô hình bằng Accuracy, hôm nay em đã hiểu với dữ liệu mất cân bằng (Spam chỉ 13%), Accuracy 98% là vô nghĩa và phải dùng F1/Precision/Recall. Ngoài ra, em cũng thấy rõ sự đánh đổi khốc liệt giữa tốc độ, độ chính xác và chi phí của LLM so với các mô hình truyền thống (đường Pareto), LLM rất thông minh nhưng không thể "bê nguyên" vào hệ thống cần xử lý hàng triệu request tốc độ cao nếu không có cơ chế Router phù hợp.
