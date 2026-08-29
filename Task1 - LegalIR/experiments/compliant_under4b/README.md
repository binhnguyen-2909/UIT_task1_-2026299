# Compliant-under-4B experiment

Đây là khu vực kiểm soát tuân thủ độc lập. Không sửa hoặc ghi đè `v57` hiện
tại. Theo phản hồi của BTC, tổng tham số của toàn bộ hệ thống end-to-end phải
nhỏ hơn 4B; quantization, LoRA, GPTQ, AWQ và GGUF không làm giảm số tham số.

## Kết luận về v57 hiện tại

V57 hiện tại không compliant. Lineage thực tế của nó có:

```text
BQB legal embedding       0.33B
Qwen3-Embedding            0.60B
Vietnamese Reranker        0.76B
Vi-Qwen2-RAG               1.50B
Qwen2.5-3B-Instruct        3.00B+
----------------------------------
tổng                       ≈6.19B+
```

Chỉ hai LLM đã là khoảng 4.5B. Chạy chúng ở các stage khác nhau, dùng cache
frozen hoặc NF4 4-bit không biến pipeline end-to-end thành hệ thống dưới 4B.

## Các profile hợp lệ để nghiên cứu

### Profile A — v47-like

```text
BQB 0.33B + Qwen3-Embedding 0.60B
+ Vietnamese Reranker 0.76B + Vi-Qwen2-RAG 1.50B
= khoảng 3.19B
```

Đây là profile gần nhất với v47: bỏ Qwen2.5-3B, sau đó phải chạy lại legal
rescue và exact10 trên output mới. Không được lấy `parent_legal2.json` của v57
rồi gọi đó là profile A vì parent ấy đã kế thừa Qwen2.5.

### Profile B — một LLM chính

```text
BQB 0.33B + Qwen2.5-3B-Instruct 3.00B+
= khoảng 3.33B
```

Profile B bỏ Vi-Qwen2, Qwen3-Embedding và Vietnamese Reranker. BM25/FTS5,
legal rules và exact selector không có tham số model. Score của profile B chưa
được đo trong workspace; cần chạy một thí nghiệm mới.

### Profile C — pure rule-based

Không có model neural. Holdout BTC-score đã đo được:

- strict lexical rules: Recall `0.6959`;
- thêm lexical question memory: Recall `0.7762`;
- oracle union top-5 candidate hiện tại: Recall `0.8752`.

Vì vậy profile C chưa đạt khoảng `0.90`; oracle cũng chưa chạm `0.90` với
candidate sources hiện tại.

### Profile D — v11 legal hybrid (đang chạy)

```text
BQB legal embedding       0.33B
Vietnamese reranker       0.76B
                          ------
                          1.09B < 4B
```

Profile D tái sử dụng các ý tưởng thuật toán của v57 — hybrid lexical/dense,
candidate pool rộng, legal structure, one-slot evidence rescue và passage
aggregation — nhưng xây index, prediction và reranker từ đầu. Nó không dùng
LLM upstream hay frozen output của v57.

## Khả năng tái tạo từ Git

Git hiện tại chỉ version-control code nền/v10 và scorer. Các artifact v14/v47/v49
như model checkpoint, dense index, rank cache, per-query LLM score và runner
end-to-end đã bị cleanup hoặc chỉ còn ở workspace local không tracked. Vì vậy
không thể khẳng định tái tạo byte-for-byte v47 từ Git hiện tại. Có thể dùng raw
data + code lịch sử để dựng lại, nhưng phải tải/khôi phục model và chấp nhận đó
là một lần chạy mới.
