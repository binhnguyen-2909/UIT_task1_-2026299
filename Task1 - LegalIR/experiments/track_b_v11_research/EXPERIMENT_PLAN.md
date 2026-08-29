# V11 research runbook

Mục tiêu là đo từng thay đổi theo thứ tự ưu tiên của nghiên cứu, không kết luận
model mới thắng khi thực tế chỉ là lỗi truncation hoặc fusion.

V57 chỉ được dùng như nguồn ý tưởng thuật toán. V57 lineage không được dùng
làm model, prediction hay input của v11 vì tổng tham số end-to-end của v57 là
`6.19B+`, vượt `<4B`. V11 dùng registry v10 độc lập: BQB 0.33B + Vietnamese
reranker 0.76B = `1.09B`.

## Bước 0 — Khóa baseline

Giữ nguyên artifact v10:

```powershell
Task1 - LegalIR\experiments\track_b_v10\run_v10_rebuild.ps1 -Device cuda
```

Lưu lại `*.audit.json`, đặc biệt các trường `answers`, `candidates`, `reranker.trace`
và latency. Không ghi đè thư mục `v10_github_rebuild` khi chạy các bước sau.

## Bước 1 — Data-only audit

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11_research.ps1 `
  -Device cuda -NoReranker -IndexName v11_research_256
```

Kiểm tra `legalir.config.json` và manifest trong SQLite:

- `chunk_strategy = legal`;
- `training_quality.enabled = true`;
- không có dữ liệu ngoài `selected-contexts.zip`/`train.json`;
- số chunk và document không giảm bất thường.

## Bước 2 — Stage 1 candidate oracle

So sánh candidate recall của v10 và v11 trước khi bật cross-encoder. Nếu v11
không tăng candidate recall, dừng việc tune reranker và quay lại chunking/RRF.

Stage 1 phải được chạy trên toàn bộ holdout 1.000 câu bằng `parallel_stage1_infer.py`.
Không dùng score của sample100 làm kết luận. Cổng qua tầng là:

- `candidate-oracle@20 >= 0.90` là mục tiêu tối thiểu để Stage 2 có đủ không gian;
- `candidate-oracle@100 >= 0.97` là cổng chẩn đoán để biết lỗi còn nằm ở retrieval;
- Recall@5 của Stage 1 chỉ là tiêu chí phụ, vì Stage 2 vẫn được phép sắp xếp lại
  trong candidate set;
- index và train phải tách holdout: `v11_research_fit_256` dùng train-fit đã loại
  1.000 câu holdout, còn `v11_research_256` chỉ dùng để kiểm tra độ nhạy.

Nhánh chạy ổn định, ít rủi ro nhất hiện tại là `chunk_only`: giữ focused search,
dense retrieval và chunk-phrase rescue; tắt các nhánh OR/question-dense/char-neighbor
để giảm candidate nhiễu trước khi rerank. Lệnh đầy đủ:

```powershell
& .python312-embed\python.exe 'Task1 - LegalIR\scripts\parallel_stage1_infer.py' `
  --index-dir 'Task1 - LegalIR\data\processed\v11_research_fit_256' `
  --input 'Task1 - LegalIR\data\raw\public_test\train.json' `
  --ids-cache 'Task1 - LegalIR\experiments\track_b_v11_research\artifacts\holdout_rank_cache.jsonl' `
  --reference 'Task1 - LegalIR\data\raw\public_test\train.json' `
  --model-registry 'Task1 - LegalIR\model_registry.v10.example.json' `
  --output 'Task1 - LegalIR\output\v11_holdout_stage1_chunk_only_top5.zip' `
  --profile chunk_only --workers 8 --encode-batch-size 64 --top-k 5 `
  --device cuda --checkpoint-every 250
```

```powershell
& .python312-embed\python.exe 'Task1 - LegalIR\scripts\analyze_candidate_oracle.py' `
  --index 'Task1 - LegalIR\data\processed\v11_research_256\legalir.sqlite3' `
  --data 'Task1 - LegalIR\data\raw\public_test\train.json'
```

## Bước 3 — Reranker reachability

Bật Stage 2 bằng runner mặc định. Các thay đổi được đo cùng lúc nhưng audit
phải giữ lại `rerank_candidates=80`, `passages_per_doc=3`, `max_length=768`,
`adaptive_rerank=true`, `aggregation=top3` và `reranker_fp16=true`. FP16 chỉ là
runtime optimization, không giảm số tham số.

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11_research.ps1 -Device cuda
```

Khi GPU thiếu bộ nhớ, giảm `-RerankerBatchSize 4`, không giảm `max_length` trước
khi có một run 512/768 để tránh lặp lại lỗi truncation của v10.

Sau khi Stage 1 qua cổng, dùng batch runner để tách riêng việc hydrate passage
và việc tính CrossEncoder. Với cùng một Stage-1 audit, thử theo thứ tự `p1` rồi
`p3`; chỉ giữ `p3/top3` nếu Recall tăng trên fit và không tụt trên holdout:

```powershell
& .python312-embed\python.exe 'Task1 - LegalIR\scripts\batch_stage2_rerank.py' `
  --stage1-audit 'Task1 - LegalIR\output\v11_holdout_stage1_chunk_only_top5.audit.json' `
  --index-dir 'Task1 - LegalIR\data\processed\v11_research_fit_256' `
  --model 'Task1 - LegalIR\models\thanhtantran_Vietnamese_Reranker' `
  --model-registry 'Task1 - LegalIR\model_registry.v10.example.json' `
  --reference 'Task1 - LegalIR\data\raw\public_test\train.json' `
  --output 'Task1 - LegalIR\output\v11_holdout_stage2_chunk_only_c20_p3_top5.zip' `
  --candidate-limit 20 --passages-per-doc 3 --max-length 768 `
  --batch-size 16 --device cuda --fp16 --neural-weight 1 --rrf-k 10
```

Stage 2 không được bổ sung document ngoài Stage-1 candidate audit. Vì vậy nếu
candidate-oracle thấp thì không được “cứu” bằng reranker; phải quay lại Stage 1.

## Bước 4 — Ablation tối thiểu

Chạy lần lượt trên cùng index v11, mỗi run một thư mục output riêng:

| Run | Thay đổi | Mục đích |
| --- | --- | --- |
| A0 | v10 fixed/256/35×2 | mốc |
| A1 | legal chunks | đo cấu trúc pháp luật |
| A2 | A1 + CE 768 | đo truncation |
| A3 | A2 + adaptive 3 passages | đo candidate reachability |
| A4 | A3 + top3 aggregation | đo document evidence aggregation |
| A5 | A4 + dense weight/RRF mới | đo fusion |

Tiêu chí chọn cấu hình là lexicographic: tối đa `Recall@5`, nếu bằng nhau mới
chọn `Precision@5`; mọi submission phải có đúng tối đa 5 ID duy nhất.

Mỗi tầng phải lưu bốn bằng chứng: cấu hình/model registry, số câu đã chạy,
candidate-oracle ở nhiều k, và latency/exception count. Một cấu hình chỉ được
đẩy sang tầng sau khi chạy lại cùng command trên holdout mà không có lỗi; một
sample run chỉ dùng để loại nhanh cấu hình, không dùng để tuyên bố vượt 0,88.

## Bước 5 — Model challenger sau cùng

Chỉ sau khi A0–A5 có audit ổn định mới thay dense model bằng Qwen3/BGE/Nemotron.
Mỗi challenger cần registry BTC riêng và một thư mục index riêng; không thay
reranker tiếng Việt hiện tại đồng thời với embedding để giữ khả năng quy lỗi.
