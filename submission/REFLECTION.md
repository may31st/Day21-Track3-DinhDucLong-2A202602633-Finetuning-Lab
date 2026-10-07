# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hiện tượng quên kiến thức nền (catastrophic forgetting) xảy ra nhanh hơn tôi tưởng: chỉ sau đúng 30 optimizer steps (2 epoch trên 225 mẫu) huấn luyện phân loại JSON, điểm đánh giá tổng quát (regression) đã tụt từ 0.7911 xuống 0.5889 (mất hơn 20% điểm số). Thêm nữa, run `attn_only` dù train loss thấp hơn cả `correct` (0.5381 so với 0.6264) nhưng khi kiểm tra trên tập test target lại chỉ hòa điểm 0.970, cho thấy việc nhìn train loss để đánh giá mô hình dễ gây hiểu nhầm thế nào.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở công đoạn chạy 3 run đối chứng ở NB4 và chấm điểm ở NB5 trên GPU T4 (hơn 30 phút). Ban đầu tôi đoán phần xử lý dữ liệu và cấu hình chat template ở NB1 sẽ tốn thời gian nhất, nhưng thực tế việc chạy suy luận và train từng cấu hình (nhất là QLoRA do phải giải nén trọng số on-the-fly) mới là phần mất thời gian chờ đợi nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab này, tôi từng nghĩ cứ nâng rank LoRA thật cao và gắn vào attention là mô hình sẽ giỏi hơn, miễn là loss giảm đều. Bây giờ tôi thấy việc phủ adapter lên toàn bộ các lớp linear (`all-linear`) ở rank vừa phải (r=16) hiệu quả hơn nhiều so với việc dồn rank lớn vào attention, và loss thấp trên tập train hẹp chỉ phản ánh mô hình đang ghi nhớ dữ liệu chứ không đồng nghĩa với năng lực thực tế.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để hỗ trợ phân tích nguyên nhân các ca đoán sai nhãn (error analysis) và kiểm tra lại cú pháp chat template Jinja. Điểm nó hay sai ban đầu là có xu hướng gợi ý hạ tiêu chuẩn đánh giá hoặc nới lỏng ngưỡng regression gate để cố làm cho kết quả thành PASSED, thay vì phân tích đúng bản chất kỹ thuật rằng kết quả FAILED hoàn toàn hợp lệ và có giá trị thực tế cao hơn.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên là xây dựng một bộ dữ liệu kiểm thử độc lập gồm cả bài test chuyên môn và bài test chống hồi quy các tác vụ thông thường, sau đó đo kỹ năng lực của base model với prompt tối ưu (few-shot). Chỉ khi baseline prompt không đạt yêu cầu và đã chuẩn bị sẵn dữ liệu đệm (replay data) để chống quên kiến thức, tôi mới bắt tay vào fine-tune LoRA.
