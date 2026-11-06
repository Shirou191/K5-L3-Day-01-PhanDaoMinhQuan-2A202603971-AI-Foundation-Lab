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

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
