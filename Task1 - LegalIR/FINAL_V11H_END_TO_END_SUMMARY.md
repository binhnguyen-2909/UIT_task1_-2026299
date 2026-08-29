# LegalIR — tổng hợp cuối và pipeline v11h end-to-end

Ngày chốt: 2026-08-29.  Bản deploy duy nhất được giữ lại là **v11h**.  Mọi
nhánh thử nghiệm khác chỉ còn lịch sử trong `EXPERIMENT_LOG.md`, các disclosure
và file này; output/checkpoint dư thừa đã được dọn khỏi workspace.

## Kết luận cuối

| Hạng mục | Kết quả |
|---|---:|
| Bản chính | `output/v11h.zip` |
| Public Recall | `0.9183` |
| Locked holdout Recall@5 | `0.9188333333` |
| End-to-end model budget | `1,090,000,000` (`1.09B`) |
| Giới hạn yêu cầu | `<4B` — đạt |
| Public submission thay thế | Không |
| SHA-256 v11h | `57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175` |

`v11h` vẫn là hệ thống duy nhất được xem là production/public. Các kết quả OOF
cao hơn nhưng holdout giảm không được promote.

## Luồng chạy v11h từ đầu đến cuối

```text
raw train/public/context
        |
        +-- tách fit 6,000 câu + locked holdout 1,000 câu
        |
        +-- build v11 research index
        |     legal chunking 1,200 / overlap 150
        |     BTC-only normalize + clean duplicate question
        |     BQB Vietnamese legal embedding ~330M, CUDA ưu tiên
        |     dense max length 256, 200 initial candidates
        |
        +-- Stage 1 hybrid retrieval
        |     focused lexical + dense RRF
        |     dense weight 0.30, RRF k=20
        |     v11h safe-tail PRF/citation completion
        |     khóa cứng top-5, mở rộng tail đến c35
        |
        +-- Stage 2 reranking
        |     Vietnamese cross-encoder ~760M
        |     max length 768, FP16 CUDA nếu có
        |     hydrate tối đa 3 legal passages/document
        |     aggregate best + 0.15*second + 0.05*third
        |     neural RRF weight 1.0, k=10
        |
        +-- selector v11h
        |     v2 query-normalized features
        |     grouped OOF complement, pointwise logistic C=1
        |     candidate pool c35
        |     exact legal-heading rescue >=10 từ
        |     chỉ thay slot 5 khi policy đã khóa
        |
        +-- validate
        |     grouped complement OOF
        |     locked holdout mở một lần sau khi khóa policy
        |     kiểm tra 1,000 QID / 5 string IDs / không duplicate
        |
        +-- package output/v11h.zip
```

Runner tái lập là
`experiments/track_b_v11_research/run_v11h_end_to_end.ps1`. Runner gọi lần
lượt ba stage nội bộ F → G → H trong cùng một file và có cache-aware skip;
không gọi runner chain bên ngoài. Mặc định dùng CUDA, có thể chuyển
`-Device cpu` khi GPU không khả dụng.

## Thành phần được giữ lại

- `data/raw/public_test/`: train, public questions và selected contexts.
- `data/processed/v11_research_fit_256/`: index fit loại holdout.
- `data/processed/v11_research_256/`: public index.
- `data/processed/v11_research_complement_256/`: clean complement index.
- `output/v10_9_prf/`: chỉ các artifact safe-tail c35 dùng trong chain v11h.
- `output/v11_feature_cache/`: feature cache v11h.
- `output/v11h_query_norm_ltr/`: selector và report v11h.
- Các audit trung gian `v11_*_chunk_only_top5` cần cho runner v11f/v11g.
- `models/bqbbao6_vietnamese-legal-embedding/` và
  `models/thanhtantran_Vietnamese_Reranker/` ở workspace root.
- Source, tests và log được giữ để runner v11h còn tái lập được và lịch sử
  nghiên cứu không bị mất.

## Nhật ký toàn bộ hướng đã thử

| Giai đoạn | Ý tưởng | Kết quả / quyết định |
|---|---|---|
| V10/V49/v57 | LLM-assisted RAG, legal heading, citation completion, exact-heading | v57 exact10 đạt public `0.9190`, nhưng lineage vượt 4B; chỉ kế thừa ý tưởng |
| V11 initial | compliant hybrid dense/lexical, legal chunk, reranker | Stage-2 repaired đạt holdout `0.888917`; làm nền cho nhánh compliant |
| V10.3 ablation | dense prefix/no-prefix và sequence length | prefix/no-prefix chỉ là channel khảo sát; không thay v11h |
| V10.9 | safe-tail PRF + citation QE | tăng oracle c35, giữ top-5; trở thành input v11g/v11h |
| V11g | grouped OOF LTR, c35, exact-heading | holdout `0.912000`, public `0.9086`; không phải main cuối |
| V11h | query-normalized v2, pointwise logistic C=1, exact10 | holdout `0.918833`, public `0.9183`; **được giữ làm main** |
| V11i | logistic/selector ablation | không vượt v11h |
| V11j | listwise/ListNet/Lambda-style selector | OOF tăng nhưng holdout `0.915000`; loại |
| V11k | rank-normalized score ensemble | holdout `0.914000`; loại |
| V11l | c50 mở rộng | holdout `0.916833`, public `0.9181`; gần nhưng không thay v11h |
| V11m | dual-dense/prefix-512 | holdout `0.914333`, public `0.9090`; loại |
| V12 | aspect MaxSim, greedy evidence, TF-IDF, LambdaMART | OOF `0.910290`, holdout `0.906114`; dừng |
| V13 router | conditional fusion/router | holdout không tăng `0.918833`; dừng |
| V13 LoRA | hard-negative reranker fine-tune | holdout `0.901083`; loại mạnh |
| V14 replacement | deterministic legal evidence + one-slot selector | complement tăng nhưng không đủ robust; chưa public |
| V14 ColBERT | Jina late interaction, typed MaxSim, evidence union | complement `+0.005013`, holdout `-0.001000`, IP `0.3333`; reject |
| V15 pairwise | direct A-minus-B, exclusive evidence, tournament, cycle gate | best complement delta `-0.001253`, IP `0.40`; không mở holdout/LoRA |

Chi tiết từng run, lỗi đã phát hiện, leakage policy, metric và hash nằm trong
`EXPERIMENT_LOG.md`, `CURRENT_STATE.md`, `V14_COLBERT_DISCLOSURE.md` và
`V15_COUNTERFACTUAL_DISCLOSURE.md`.

## Các bài học đã khóa vào v11h

1. Không dùng holdout/public để chọn policy.
2. Mọi selector phải so với đúng v11h anchor; không trộn mapping top-5 khác
   lineage.
3. Candidate coverage cao không tự động tạo utility; phải đo rescue,
   destructive và transfer theo nhóm query.
4. Chỉ mở candidate tail có kiểm soát; không mở c50/candidate source mới nếu
   selector chưa chứng minh transfer.
5. Exact-heading có tín hiệu, nhưng threshold và số intervention quyết định
   transfer; v11h giữ policy exact10 chặt.
6. GPU chỉ dùng cho dense/reranker forward. Các selector thuần thuật toán
   chạy CPU để tránh overhead; tổng tham số vẫn là 1.09B.

## Cách chạy lại

Từ repository root:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1 -Device cuda
```

Nếu GPU bị chiếm hoặc không khả dụng:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1 -Device cpu
```

Không dùng `-Force` trừ khi muốn rebuild toàn bộ artifact tốn thời gian. Bản
public cần lấy từ `Task1 - LegalIR/output/v11h.zip`.

## Trạng thái dọn dẹp checkpoint

Đã dọn workspace ngày 2026-08-29:

- Xoá 80 file output trung gian, sweep, fallback và submission cũ ở
  `output/` và `output/v10_9_prf/`.
- Xoá các index cũ, thư mục thí nghiệm không thuộc chain v11h và bản model
  trùng trong `Task1 - LegalIR/models/`.
- Giữ nguyên source, tests, raw data, ba index cần thiết, hai model ở workspace
  root, runner `v11f/v11g/v11h`, audit c35 và toàn bộ log/disclosure.
- `v11f.zip` và `v11g.zip` chỉ là checkpoint nội bộ để đối chiếu; artifact nộp
  duy nhất vẫn là `output/v11h.zip`.

Các file đã xoá không còn bản sao cục bộ, nhưng đều có thể dựng lại từ source,
data, model và runner còn giữ. `v11h.zip` được giữ nguyên và hash không đổi.
