# Pure Deep Learning vs LLM dưới constraint `<4B`

## Kết luận ngắn

V57 **không phải cấu hình hợp lệ** cho constraint của BTC. Theo manifest tuân
thủ trong `experiments/compliant_under4b`, lineage của v57 gồm BQB legal
embedding 0.33B, Qwen3-Embedding 0.60B, Vietnamese reranker 0.76B,
Vi-Qwen2-RAG 1.50B và Qwen2.5-3B-Instruct 3.00B+, tổng khoảng **6.19B+**.
Riêng hai LLM upstream đã khoảng 4.5B; chạy khác stage, cache frozen hay
quantization không làm tổng số tham số end-to-end giảm.

Vì vậy, v11 được xây độc lập, chỉ tái sử dụng **ý tưởng thuật toán** đã quan
sát ở v57, không tái sử dụng model, checkpoint, prediction hay parent output
của v57. Profile hiện tại của v11 là:

```text
BQB Vietnamese legal embedding       0.33B
Vietnamese legal reranker             0.76B
                                      ------
                                      1.09B < 4B
```

Khuyến nghị cho BTC: dùng retrieval-specialized Pure DL + lexical hybrid làm
đường chính; chỉ thêm LLM khi nó nằm trong một profile độc lập đã tính đủ
tham số và được chứng minh trên cùng protocol fit/holdout. Không dùng LLM
generator để sinh trực tiếp document ID.

## “Pure DL” và “LLM” không phải hai tập loại trừ

Embedding model và cross-encoder reranker cũng là deep neural networks. Cách
so sánh đúng là theo **mục tiêu huấn luyện và vai trò trong pipeline**:

| Nhánh | Vai trò | Điểm mạnh | Rủi ro với LegalIR |
|---|---|---|---|
| BM25/FTS5/sparse | Stage 1 lexical | Không tham số; rất nhanh; bắt chính xác số Điều, văn bản, mã hiệu | Kém với paraphrase và cách hỏi khác nguyên văn |
| Legal bi-encoder 330M | Dense candidate retrieval | Encode corpus một lần; tốt cho Recall@K và semantic matching | Có thể bỏ sót citation hiếm; index lớn; cần kiểm soát truncation |
| Cross-encoder 760M | Stage 2 reranking | So sánh query–passage trực tiếp; phân biệt điều kiện pháp lý tinh hơn | Chi phí tăng theo số candidate × passage; dễ lãng phí nếu cắt 256 token |
| Late interaction | Token-level retrieval/rerank | Giữ tín hiệu token pháp lý thay vì nén toàn câu; ColBERTv2 cho thấy có thể nén residual index | Phức tạp hơn, tốn storage và cần benchmark riêng |
| Small embedding/reranker LLM | Challenger retrieval | Instruction-aware, đa ngôn ngữ, có thể hiểu paraphrase | Generic-domain mismatch; cần chứng minh trên tiếng Việt pháp lý |
| Generative LLM | Rerank/rescue/teacher | Có thể đọc và giải thích điều kiện, citation, ngoại lệ | Chậm, khó tái lập, hallucination/format risk; sinh ID trực tiếp không an toàn |

DPR cho thấy bi-encoder là cách thực dụng để truy hồi nhanh ở Stage 1
[DPR, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.550/). ColBERTv2
giữ tương tác token-level bằng late interaction và giảm chi phí lưu trữ nhờ
residual compression [ColBERTv2, NAACL 2022](https://aclanthology.org/2022.naacl-main.272/).
BGE-M3 là ví dụ hybrid hiện đại kết hợp dense, sparse và multi-vector trong
một encoder dài ngữ cảnh [BGE-M3](https://arxiv.org/abs/2402.03216).

## Vì sao v11 ưu tiên Pure DL/hybrid

Khoảng 10k câu hỏi không đủ lớn để fine-tune một generator 1.5–3B một cách
ổn định cho nhiệm vụ sinh document ID. Nhưng nó đủ hữu ích cho:

1. hard-negative mining từ BM25 và dense top-k;
2. học một linear/LightGBM rank selector trên feature retrieval;
3. calibrate RRF weight, candidate cutoff, legal-element gate;
4. kiểm tra duplicate question, leakage và độ ổn định theo fold.

Mục tiêu chính của LegalIR là danh sách ID đúng, không phải văn bản trả lời.
Do đó, candidate coverage và exact legal matching thường quan trọng hơn khả
năng viết/giải thích của một generator. RAG nguyên bản cũng tách parametric
memory khỏi non-parametric memory, nhưng không loại bỏ yêu cầu kiểm soát
provenance khi câu trả lời phải dựa trên corpus [RAG](https://arxiv.org/abs/2005.11401).

Các nghiên cứu về prompt/LLM reranking lưu ý rằng chi phí tăng theo số cặp
query–document; prompt-based representations thường chỉ rerank được một số ít
ứng viên, còn dense retrieval cần dữ liệu paired phù hợp
[PromptReps, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.250/).
Nghiên cứu LLM làm ranker cũng cho thấy prompting có thể hữu ích nhưng phải
đánh đổi với chi phí và sensitivity của ranking procedure
[LLMs as Rankers, Findings of NAACL 2024](https://aclanthology.org/2024.findings-naacl.97/).

## Thuật toán v57 được phép tái sử dụng trong v11

V11 chỉ lấy các nguyên tắc sau, không lấy output/model của v57:

- hybrid lexical + dense RRF;
- candidate reach rộng trước khi Stage 2;
- legal-aware chunking theo Điều/Khoản/Điểm;
- query-focused passage hydration;
- aggregate nhiều passage nhưng để passage tốt nhất chi phối;
- one-slot rescue có điều kiện, chỉ nhận phrase dài hơn và có tín hiệu cấu trúc;
- chọn threshold trên fit, holdout chỉ xác nhận, public không dùng để fit.

Các thay đổi này đang được cô lập tại
`experiments/track_b_v11_research/run_v11_research.ps1` và index mặc định
`data/processed/v11_research_256`.

## Profile v11 hiện tại

| Thành phần | Cấu hình |
|---|---|
| Corpus | 8,532 documents; 33,846 legal chunks |
| Chunking | legal structure, 1,200 chars, overlap 150 |
| Lexical | FTS5 + exact/chunk phrase + question memory |
| Dense | BQB legal embedding 330M, max 256 tokens, RRF weight 0.30, k=20 |
| Candidate pool | 200 trước reranking |
| Reranker | Vietnamese legal reranker 760M, tối đa 80 docs × 3 passages |
| Passage aggregation | `best + 0.15*second + 0.05*third` |
| Reranker input | 768 tokens |
| Stage-2 fusion | weight 1.0, RRF k=10 |
| Runtime optimization | FP16 trên CUDA; không thay đổi parameter count |
| Tổng tham số | khoảng 1.09B, compliant với `<4B` |

Bản 256-token của dense branch là lựa chọn runtime trên RTX 3050 6GB; Stage
2 vẫn dùng 768-token để tránh lỗi truncation của v10. Nếu ablation chứng minh
256/512 tốt hơn trên holdout, phải ghi rõ đó là trade-off được đo, không mặc
định rằng ngữ cảnh dài luôn tốt hơn.

## Protocol đánh giá bắt buộc

Mỗi variant v11 cần lưu:

- model registry và tổng tham số unique;
- hash corpus, train split và index config;
- Recall@1/5, Precision@1/5, candidate-oracle Recall@K;
- candidate reachability trước và sau dense;
- reranker input coverage, số passage thực tế, latency p50/p95;
- số câu bị thay đổi bởi rescue và gained/lost/both-hit;
- fit-selected policy và holdout-only confirmation.

Không chọn policy từ public score. Không gọi v57 public score là kết quả của
v11. Khi cần so sánh với v57, chỉ ghi v57 như historical upper reference và
đồng thời đánh dấu **non-compliant**.

## Trả lời trực tiếp câu hỏi BTC

Với compute hạn chế, `<4B` và khoảng 10k data:

1. **Pure DL retrieval chuyên biệt** có xác suất cho Pareto tốt hơn: chi phí
   index hóa một lần, inference ổn định, dễ kiểm soát top-k và provenance.
2. **LLM nhỏ** có thể thắng ở paraphrase hoặc legal reasoning cục bộ, nhưng
   chỉ có ý nghĩa sau khi candidate oracle đã đủ cao. Nó nên là reranker/rescue
   có giới hạn, không phải generator document ID.
3. Một LLM 3B đơn lẻ dưới 4B chưa chắc thắng hệ hybrid 1.09B; số tham số
   không thay thế domain alignment, hard negatives và protocol đánh giá.
4. Nếu thêm LLM, phải chạy A/B cùng corpus, cùng candidate pool, cùng fold và
   cùng latency budget. Nếu không, mọi kết luận “LLM tốt hơn Pure DL” đều không
   có giá trị nhân quả.

**Quyết định triển khai:** giữ v11 legal hybrid làm baseline compliant; chỉ mở
LLM challenger sau khi v11 có audit holdout hoàn chỉnh và sau khi chứng minh
LLM đó làm tăng Recall/Precision mà không phá parameter budget, provenance và
latency.
