# Lab 21 — Evaluation Report

**Họ tên**: Đinh Đức Long  **MSSV**: 2A202602633  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (p95 đo được là 98 trong *results/token_stats.json*) |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2.0 / 30 optimizer steps |

**Template có giữ khối `<think>` không?** có (*results/template_check.json*)
Template của `unsloth/Qwen3.5-4B` giữ nguyên vẹn khối `<think>`. Chuỗi kiểm tra hiển thị rõ cấu trúc `<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4<|im_end|>`, file json ghi nhận `reasoning preserved — safe to train on traces`. Do template không xóa khối suy luận, tôi giữ nguyên cấu hình mặc định mà không cần chỉnh sửa file jinja.

*Ghi chú về `max_length`:* Phân tích `token_stats.json` cho thấy độ dài token có mean = 93.1, p50 = 93, p95 = 98 và max = 101. Con số gợi ý cho max_length là 256. Tôi giữ nguyên mức 1024 theo mặc định của tier T4 trên Colab để đảm bảo không mẫu nào bị cắt ngắn. Bộ nhớ VRAM khi chạy thực tế chỉ tốn 8.78 GB trên tổng số 14.6 GB của T4, nên việc để max_length = 1024 không gây lãng phí tài nguyên đáng kể.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3151.7 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1019.9 |
| (c) LoRA fine-tune | 0.970 | 0.5889 | 1.000 | 1380.4 |

**(b) có thật sự mạnh hơn (a) không?** có. Baseline (b) cải thiện rõ rệt so với (a): điểm target tăng từ 0.000 lên 0.765, tỷ lệ trả lời đúng định dạng JSON tăng từ 0 lên 1.0, và thời gian sinh kết quả giảm từ 3151.7 ms xuống 1019.9 ms nhờ có các ví dụ mẫu (few-shot) giúp mô hình trả lời thẳng vào nội dung.

Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Tôi giữ nguyên prompt (b) gốc với mã SHA `719e74d3b6232053`. Mục tiêu của lab là so sánh bản fine-tune với một baseline prompt thực sự tốt, nên tôi không sửa prompt để hạ thấp mốc đối chứng nhằm làm kết quả fine-tune trông đẹp hơn.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6264 | 0.970 | 399.5 | 8.78 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5381 | 0.970 | 268.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-05 | 1.5702 | 0.000 | 398.2 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.940 | 455.8 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1: `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

Khi kiểm tra trên tập target ở NB5, `attn_only` đạt 0.970, ngang bằng với `correct` (0.970). Tuy nhiên, train loss của `attn_only` lại thấp hơn (0.5381 so với 0.6264). Điều này cho thấy việc dồn rank lên 283 vào riêng hai khối q và v chỉ giúp mô hình khớp dữ liệu huấn luyện tốt hơn chứ không cải thiện khả năng tổng quát hóa trên tập kiểm thử. Với tác vụ phân loại JSON, việc phủ LoRA lên toàn bộ các lớp tuyến tính (`text-linear`) cho phép mô hình đạt độ chính xác tương đương mà chỉ cần rank 16.

**4.2: `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` giảm learning rate từ 1e-4 xuống mức 1e-5 (thang đo của full fine-tuning). Đường loss của run này gần như đi ngang, bắt đầu ở mức 2.163 và sau 30 step mới chỉ xuống 1.119, khiến mô hình đạt 0.000 trên cả target và format. Nếu chỉ nhìn đường loss giảm chậm mà không biết cấu hình learning rate, người làm thí nghiệm rất dễ đoán mò rằng tập dữ liệu quá khó, số epoch chưa đủ, hoặc rank 16 là quá nhỏ. Trong thực tế, vấn đề nằm hoàn toàn ở việc bước cập nhật quá ngắn khiến trọng số LoRA chưa kịp thích nghi với định dạng đầu ra.

**4.3: `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

Bản `qlora` giảm bộ nhớ VRAM cao nhất từ 8.78 GB xuống 3.86 GB, tiết kiệm khoảng 4.92 GB. Đổi lại, thời gian huấn luyện tăng từ 399.5 giây lên 455.8 giây (chậm hơn khoảng 14%), độ trễ khi sinh mẫu tăng từ 1380.4 ms lên 1845.4 ms (chậm hơn 33.7%), và điểm target giảm nhẹ từ 0.970 xuống 0.940. Kết quả đo được khớp với khuyến cáo của tác giả mô hình: với kiến trúc kết hợp linear attention như Qwen3.5, việc lượng tử hóa 4-bit làm tăng độ trễ tính toán và giảm nhẹ độ chính xác. Do phần cứng T4 16GB dư sức chứa bản 16-bit, việc dùng QLoRA trong trường hợp này mang lại bất lợi nhiều hơn lợi ích.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.202` · `valid_trace_rate = 0.00`

Diễn giải:
Cổng hồi quy đánh giá kết quả là `FAILED` vì điểm số trên tập kiểm tra tổng quát `eval_regression` bị tụt từ 0.7911 xuống 0.5889, tương đương mức giảm 0.202 (vượt xa ngưỡng cho phép là 0.020). Mặc dù điểm target tăng từ 0.765 lên 0.970 và mô hình sinh đúng định dạng JSON 100%, việc huấn luyện chỉ trên 225 mẫu ticket CSKH mà không trộn thêm dữ liệu tổng quát đã gây ra hiện tượng quên kiến thức nền (catastrophic forgetting). Mô hình đã tối ưu toàn bộ trọng số LoRA cho tác vụ trích xuất 4 trường JSON, dẫn đến suy giảm khả năng trả lời các câu hỏi logic và chỉ dẫn thông thường. Ngoài ra, chỉ số `valid_trace_rate` rơi về 0.00 do tập huấn luyện không chứa khối `<think>`, khiến mô hình mất thói quen suy luận từng bước.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | intent: van_chuyen, urgency: thap, product: ốp lưng điện thoại, sentiment: tieu_cuc | Sai cấu trúc hoặc nhầm intent | Đúng cả 4 trường (score 1.0) | FT thắng |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | intent: hoi_thong_tin, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: trung_tinh | Nhầm nhãn sentiment | Đúng cả 4 trường (score 1.0) | FT thắng |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. | intent: hoan_tien, urgency: thap, product: bình giữ nhiệt, sentiment: tich_cuc | Đúng cả 4 trường (score 1.0) | urgency: trung_binh (score 0.75) | FT thua |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. | intent: san_pham_loi, urgency: thap, product: nồi chiên không dầu, sentiment: trung_tinh | Đúng cả 4 trường (score 1.0) | urgency: trung_binh (score 0.75) | FT thua |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop. | intent: san_pham_loi, urgency: thap, product: áo khoác gió, sentiment: tich_cuc | Đúng cả 4 trường (score 1.0) | urgency: trung_binh (score 0.75) | FT thua |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung ở cả 3 ca mô hình fine-tune bị trừ điểm là đều đoán sai trường `urgency`. Các ticket này chứa cụm từ thể hiện khách hàng không vội như "Khi nào tiện", nhưng do nội dung có khiếu nại về tiền bạc hoặc hàng lỗi ("Chưa thấy tiền", "Thiếu phụ kiện", "Bị lỗi"), mô hình fine-tune có xu hướng mặc định gán nhãn `trung_binh`. Trong khi đó, prompt (b) có các ví dụ mẫu hướng dẫn ngữ cảnh tốt hơn nên nhận diện đúng mức độ `thap`.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**

Tôi đánh giá mô hình fine-tune này chưa sẵn sàng để đưa vào làm trợ lý trò chuyện trực tiếp với khách hàng vì cổng hồi quy đã báo FAILED khi điểm năng lực tổng quát giảm hơn 20%. Tuy nhiên, nếu dùng mô hình này như một worker chuyên biệt nằm sau hệ thống phân loại ticket (chỉ nhận đầu vào là text và trả về JSON), thì kết quả fine-tune mang lại hiệu quả vượt trội so với prompt thông thường: độ chính xác tăng từ 76.5% lên 97.0% và tỷ lệ sinh đúng chuẩn JSON đạt 100%. Nếu muốn triển khai cho tác vụ giao tiếp đa năng, bước tiếp theo bắt buộc là phải đưa thêm 3% đến 5% dữ liệu đệm (replay data) vào tập huấn luyện để giữ lại khả năng suy luận nền. Qua các thí nghiệm, yếu tố quyết định sự thành bại của mô hình lần lượt là: chọn đúng thang learning rate (1e-4 thay vì 1e-5), gắn adapter lên toàn bộ các lớp tuyến tính (`text-linear`), và che phần prompt trong hàm loss để mô hình không lặp lại câu hỏi.

**Ba điều tôi học được:**

1. Không dùng train loss làm thước đo duy nhất: Run `attn_only` có train loss 0.5381 (thấp hơn `correct` là 0.6264), nhưng khi kiểm thử trên tập target độc lập thì hai bên hòa nhau ở mức 0.970. Train loss thấp trên tập dữ liệu nhỏ thường chỉ là dấu hiệu của việc mô hình học vẹt.
2. Cần cố định số tham số khi so sánh vị trí gắn adapter: Nếu chỉ giữ nguyên rank r=16 giữa `all-linear` và `attn_only`, số tham số sẽ lệch nhau hơn 17 lần. Sử dụng `matched_rank` để đưa cả hai về cùng mức khoảng 32.4 triệu tham số là cách duy nhất để kiểm chứng xem việc mở rộng vị trí gắn adapter có thực sự mang lại lợi thế hay không.
3. Mô hình bị suy giảm kiến thức tổng quát rất nhanh: Chỉ sau 30 step huấn luyện trên 225 mẫu dữ liệu CSKH, điểm kiểm tra tổng quát đã giảm hơn 20%. Kết quả FAILED này phản ánh đúng thực tế kỹ thuật và có giá trị thực tiễn hơn việc cố tình hạ thấp tiêu chuẩn đánh giá để lấy kết quả đỗ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Thêm khoảng 3% đến 5% dữ liệu chỉ dẫn tổng quát vào tập huấn luyện để kiểm tra xem điểm regression có quay lại ngưỡng an toàn (độ lệch dưới 0.02) hay không.
- Thử nghiệm chế độ `MASK_MODE=response-only` với tập dữ liệu có giữ lại chuỗi suy luận để đánh giá sự thay đổi của chỉ số `valid_trace_rate`.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**:
  - Đã chạy notebook `06_merge_and_serve.py`.
  - Kết quả trong `results/merge_check.json`: Điểm target trước và sau khi merge đều đạt 0.9700 (độ lệch bằng 0, nằm trong ngưỡng cho phép 0.0100). Trọng số sau khi gộp giữ nguyên được độ chính xác của adapter.
  - Đã kiểm tra cơ chế hot-swap 3 adapter (`correct`, `attn_only`, `qlora`) trên cùng một base model `unsloth/Qwen3.5-4B` đang nạp trong VRAM.
  - Trả lời câu hỏi B1:
    - Việc merge trọng số giúp loại bỏ hoàn toàn độ trễ tính toán của adapter khi suy luận, nhưng làm mất tính linh hoạt vì trọng số bị cộng cứng vào mô hình nền. Mỗi tác vụ sau khi merge sẽ tốn dung lượng lưu trữ tương đương toàn bộ mô hình gốc (khoảng 9 GB) thay vì chỉ vài chục MB cho file adapter.
    - Nên giữ adapter riêng khi triển khai hệ thống phục vụ nhiều khách hàng hoặc nhiều tác vụ phụ: chỉ cần nạp 1 base model vào bộ nhớ GPU rồi tráo đổi các adapter nhẹ tùy theo yêu cầu của từng người dùng, giúp tiết kiệm chi phí phần cứng và dễ dàng cập nhật từng adapter độc lập.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
