# Pipeline v49 — mô tả đầy đủ để công bố

Cập nhật: 2026-08-18  
Kết quả public: **0.9160**  
Vai trò hiện tại: v49 là upstream anchor của `legal2`, và vì vậy tiếp tục ảnh
hưởng đến submission cuối `v57 exact10 = 0.9190`.

## 1. Kết luận ngắn gọn

V49 không phải một lần chạy Qwen2.5 độc lập trên raw corpus. Nó là một phép
**one-slot rescue có kiểm soát** đặt trên v47. Bản thân v47 lại là một phép
one-slot rescue đặt trên v14 và có dùng `AITeamVN/Vi-Qwen2-1.5B-RAG`.

Vì vậy lineage phải công bố là:

```text
Hybrid retrieval/reranking + v14 frozen anchor (0.9135)
        ↓
Vi-Qwen2-1.5B-RAG answerability + protected one-slot gate
        ↓
v47 RAG-protected anchor (0.9150)
        ↓
Qwen2.5-3B-Instruct 4-bit answerability + stricter one-slot gate
        ↓
v49 fixed rescue (0.9160)
        ↓
2 legal rescues → legal2 (0.9175)
        ↓
exact10 deterministic selector → v57 (0.9190)
```

Hai LLM không sinh `document_id` và không sinh câu trả lời pháp lý. Chúng chỉ
cho điểm một tập candidate đã được retrieval tạo trước. Quyết định cuối cùng
được thực hiện bằng rule xác định, chỉ có quyền thay document thứ năm.

## 2. Các thành phần neural đã ảnh hưởng đến v49

| Thành phần | Model | Vai trò |
|---|---|---|
| Dense passage/question retrieval | `bqbbao6/vietnamese-legal-embedding` (330M) | Tạo `dense_passage` và `question_dense` trong họ cache nền |
| Dense candidate độc lập | `Qwen/Qwen3-Embedding-0.6B` | Tạo nguồn `qwen_dense` |
| Cross-encoder | `thanhtantran/Vietnamese_Reranker` (760M) | Tạo nguồn `cross_encoder` |
| RAG scorer tầng v47 | `AITeamVN/Vi-Qwen2-1.5B-RAG` | Tạo `rag_answerability` ở FP16 |
| LLM scorer tầng v49 | `Qwen/Qwen2.5-3B-Instruct` | Tạo `qwen25_answerability` bằng NF4 4-bit |

Các nguồn sparse `production_fusion` và `chunk_phrase` không phải deep
learning. BGE, Harrier, Qwen3-Reranker, LambdaRank và các nhánh kNN không nằm
trong lineage v49 cuối; chúng chỉ là ablation hoặc thử nghiệm bị loại.

> Lưu ý tuân thủ: model lớn nhất được load trong một stage là Qwen2.5 3B, dưới
> 4B. Tuy nhiên, nếu BTC tính **tổng tham số của mọi model từng ảnh hưởng đến
> submission** thay vì model được vận hành tại một thời điểm, tổng unique
> parameters của pipeline sẽ lớn hơn 4B. Đội phải hỏi BTC cách diễn giải và
> không được tự tuyên bố “toàn hệ thống chỉ có 3B tham số”. Lượng tử hóa 4-bit
> giảm VRAM, không làm số lượng tham số của Qwen2.5 từ 3B thành một con số nhỏ
> hơn.

## 3. Tầng candidate retrieval

Với mỗi query, rank cache chứa nhiều ranking độc lập. Candidate để hai RAG
scorer đánh giá được tạo bằng phép union có giữ thứ tự:

```text
top 5 production_fusion
+ top 5 qwen_dense
+ top 5 question_dense
+ top 5 chunk_phrase
+ top 5 cross_encoder
        ↓ loại document_id trùng
candidate universe (tối đa 25 document, thường ít hơn)
```

Ý nghĩa từng nguồn:

- `production_fusion`: ranking hybrid nền từ FTS5/BM25, phrase/legal signals,
  dense passage/question retrieval và fusion.
- `qwen_dense`: dense retrieval bằng Qwen3-Embedding-0.6B.
- `question_dense`: tìm câu hỏi train gần query rồi truyền các document label
  liên quan; khi đánh giá local phải leave-one-out để tránh leakage.
- `chunk_phrase`: ưu tiên document chứa các cụm liên tiếp của query.
- `cross_encoder`: relevance ranking query–passage bằng Vietnamese Reranker.

Candidate union chỉ tạo **phạm vi lựa chọn**. Thứ hạng cuối trong nguồn
answerability được quyết định lại bằng LLM score.

## 4. Passage hydration

Mỗi candidate document được nối với passage trước khi đưa vào LLM:

1. Dùng SQLite retriever tìm passage tốt nhất của document đối với query.
2. Chỉ giữ `1 passage/document`.
3. Context thô bị chặn ở 12.000 ký tự.
4. Prompt cuối được giới hạn `768` token.
5. Với scorer Qwen2.5, phần cuối prompt được giữ để bảo toàn câu hỏi và nhãn
   khi prompt vượt budget.

Do đó LLM không đọc toàn bộ văn bản pháp luật. Nó đọc passage được retrieval
chọn, nên lỗi passage selection vẫn có thể làm score answerability sai.

## 5. Prompt và công thức answerability

Prompt đã dùng:

```text
Chú ý các yêu cầu sau:
- Chỉ đánh dấu Có nếu ngữ cảnh chứa thông tin để trả lời câu hỏi.
- Không suy luận hoặc dùng kiến thức ngoài ngữ cảnh.
- Chỉ trả lời đúng một từ: Có hoặc Không.

### Ngữ cảnh :
{context}

### Câu hỏi :
{question}

### Nhãn :
```

Prompt được bọc bằng chat template của tokenizer. Hệ thống không gọi
`generate()`. Với mỗi nhãn, code tính trung bình log-probability của các token
nhãn rồi lấy hiệu:

```text
answerability(q,d)
  = mean log P(tokens("Có") | prompt(q,d))
  - mean log P(tokens("Không") | prompt(q,d))
```

Score dương nghĩa là model nghiêng về “Có”; score âm nghĩa là nghiêng về
“Không”. Đây là **log-probability classification**, không phải LLM generation.

## 6. Từ v14 đến v47: RAG-protected rescue

### 6.1 Scorer

- Model: `AITeamVN/Vi-Qwen2-1.5B-RAG`.
- Precision runtime: FP16 trên CUDA.
- Candidate cutoff khi tạo scorer: top 5 của năm nguồn ở mục 3.
- Passage/document: 1.
- Max input: 768 token.
- Output source: `rag_answerability`.

### 6.2 Policy đã khóa

```json
{
  "candidate_cutoff": 1,
  "support_cutoff": 20,
  "min_support": 2,
  "margin": 1.0,
  "weak_support_max": 2
}
```

Flow chọn:

1. Lấy top 5 của v14 làm anchor `D1…D5`.
2. Xem `D5` là weak slot; giữ nguyên `D1…D4`.
3. Đếm support của `D5` trong top 20 của ba nguồn độc lập:
   `question_dense`, `chunk_phrase`, `cross_encoder`.
4. Nếu `D5` xuất hiện trong cả ba nguồn (`support > 2`) thì bảo vệ weak slot và
   không thay.
5. Nếu không bị bảo vệ, chỉ xét candidate đứng hạng 1 của
   `rag_answerability`.
6. Candidate phải nằm ngoài top 5 hiện tại, có support từ ít nhất 2/3 nguồn
   trong top 20 và có RAG score lớn hơn score của `D5` ít nhất `1.0`.
7. Nếu qua mọi gate, trả về `D1,D2,D3,D4,C`; nếu không thì abstain và giữ v14.

Policy này thay `93/1000` query public và tạo v47, public score tăng từ
`0.9135` lên `0.9150`. Holdout Recall tăng khoảng `+0.0010`, Precision tăng
`+0.0002`.

## 7. Từ v47 đến v49: Qwen2.5 fixed rescue

### 7.1 Re-anchor đúng v47

Trước khi áp Qwen2.5, rank cache public được re-anchor để
`production_fusion[:5]` chính là mapping v47. Đây là điều quan trọng: selector
luôn so candidate với document thứ năm thực sự của v47, không so với một
ranking nền khác.

### 7.2 Qwen2.5 scorer

- Checkpoint: `Qwen/Qwen2.5-3B-Instruct`.
- Snapshot đã dùng trong cache local:
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Quantization: bitsandbytes NF4 4-bit, double quantization.
- Compute dtype: FP16.
- Device: CUDA.
- Candidate pool: union top 5 của năm nguồn ở mục 3.
- Passage/document: 1.
- Max input: 768 token.
- Batch size: 4 trong lần chạy chính.
- Output source: `qwen25_answerability` và score cho từng document.

### 7.3 Chọn policy trên fit

Grid ban đầu gồm 180 policy:

```text
candidate_cutoff ∈ {1,3,5,10,20}
support_cutoff   ∈ {5,10,20}
min_support      ∈ {0,1,2}
margin           ∈ {0,1,2,3}
```

Fit gồm 800 query và được báo cáo bằng 4 fold. Khóa sắp hạng policy lần lượt là:

1. Recall cao hơn.
2. Precision cao hơn.
3. Ít query bị mất gold hơn.
4. Ít query bị thay hơn.

Policy cuối được refit/chọn trên toàn bộ fit rồi đóng băng trước khi áp public:

```json
{
  "candidate_cutoff": 1,
  "support_cutoff": 5,
  "min_support": 2,
  "margin": 3.0
}
```

### 7.4 Logic inference chính xác

Giả sử anchor v47 là `D1,D2,D3,D4,D5`:

```text
weak = D5
C = document hạng 1 theo Qwen2.5 answerability

if C đã nằm trong {D1..D5}:
    giữ nguyên
elif C xuất hiện trong ít hơn 2/3 nguồn
     {question_dense, chunk_phrase, cross_encoder} ở top 5:
    giữ nguyên
elif qwen_score(C) < qwen_score(D5) + 3.0:
    giữ nguyên
else:
    output = D1,D2,D3,D4,C
```

Vì `candidate_cutoff=1`, selector chỉ xét document mà Qwen2.5 xếp cao nhất.
Vì `min_support=2`, Qwen không được tự ý đưa một candidate semantic đơn độc
vào top 5. Vì `margin=3.0`, model phải tự tin hơn weak slot một khoảng lớn.

### 7.5 Contract output

- Giữ nguyên bốn ID đầu.
- Thay tối đa một ID ở slot 5.
- Có từ 1 đến 5 ID và không duplicate.
- Policy fit đã frozen; không tune bằng nhãn public.
- Public v49 thay đúng `20/1000` query so với v47.

## 8. Kết quả và nghịch lý local/public

| Phạm vi | Thay đổi | Gained/Lost | Recall delta | Precision delta |
|---|---:|---:|---:|---:|
| Fit policy được chọn | 21 | 1/0 | in-sample dương | in-sample dương |
| 4-fold OOF | — | — | `-0.001875` | `-0.0005` |
| Holdout 1.000 | 14 | 0/1 | `-0.0010` | `-0.0002` |
| Public 1.000 | 20 | không có per-QID gold | v47 `0.9150` → v49 `0.9160` | không được công bố riêng |

Không thể suy ra query public nào đúng chỉ từ mức tăng tổng. Kết quả chứng minh
20 thay đổi có utility tổng dương trên public, nhưng không chứng minh cả 20 đều
đúng. Local âm không phải do công thức official scorer khác; nguyên nhân hợp lý
hơn là distribution shift, fit-policy variance và độ nhạy theo anchor.

## 9. Pseudocode end-to-end của v49

```python
for question in questions:
    rankings = retrieve_with_sparse_dense_and_cross_encoder(question)
    candidates = ordered_unique_union(
        rankings["production_fusion"][:5],
        rankings["qwen_dense"][:5],
        rankings["question_dense"][:5],
        rankings["chunk_phrase"][:5],
        rankings["cross_encoder"][:5],
    )

    for doc in candidates:
        passage = hydrate_best_passage(question, doc, passages_per_doc=1)
        rag_score[doc] = logp_yes_minus_no(ViQwen2_1_5B, question, passage)

    v47 = protected_one_slot_replace(
        anchor=v14,
        scores=rag_score,
        candidate_cutoff=1,
        support_cutoff=20,
        min_support=2,
        margin=1.0,
        weak_support_max=2,
    )

    # Cache được re-anchor: production_fusion[:5] == v47
    for doc in candidates:
        passage = hydrate_best_passage(question, doc, passages_per_doc=1)
        qwen25_score[doc] = logp_yes_minus_no(
            Qwen2_5_3B_Instruct_NF4, question, passage
        )

    v49 = one_slot_replace(
        anchor=v47,
        scores=qwen25_score,
        candidate_cutoff=1,
        support_cutoff=5,
        min_support=2,
        margin=3.0,
    )
```

Đoạn trên mô tả logic nghiên cứu; runner v57 hiện tại không còn tái chạy hai
LLM mà đọc frozen parent JSON.

## 10. Giới hạn tái lập sau khi cleanup

Các con số, prompt, model, policy và lệnh runtime ở trên đã được khôi phục từ
audit/log/lịch sử chạy. Tuy nhiên artifact cũ của v14/v47/v49 đã bị xóa khi thu
gọn workspace. Hiện còn các khoảng trống sau:

- Không còn runner một lệnh tái tạo v14 → v47 → v49.
- Không còn rank cache gốc và audit per-query của 20 thay đổi public v49.
- Công thức tạo chính xác frozen v14 chưa được giữ đầy đủ trong pipeline hiện
  tại; chỉ còn provenance của họ retrieval nền.
- Không thể kiểm chứng lại hash toàn bộ model/cache cũ chỉ từ workspace hiện
  tại.

Vì vậy cách phát biểu trung thực là:

> Nhóm có thể giải thích và công bố đầy đủ logic của v49, nhưng bản repository
> đã cleanup hiện chỉ tái lập trực tiếp legal2 → v57. Việc tái lập byte-for-byte
> v14 → v49 cần phục hồi các cache/artifact cũ.

## 11. Cách nói ngắn khi thuyết trình

> Chúng tôi dùng retrieval hybrid để tạo một candidate pool đa nguồn. Một mô
> hình RAG 1.5B trước hết chấm khả năng trả lời và chỉ được thay slot 5 khi có
> đồng thuận từ ít nhất hai nguồn retrieval, tạo v47. Sau đó Qwen2.5-3B-Instruct
> 4-bit chấm lại candidate bằng hiệu log-probability giữa nhãn Có và Không.
> Qwen2.5 cũng chỉ được thay slot 5, với support 2/3 và margin 3.0. Policy được
> khóa trên fit trước khi áp public, thay 20/1000 query và tăng public từ 0.9150
> lên 0.9160. V57 kế thừa các quyết định này rồi thêm legal rescue và exact
> phrase gate, nên submission cuối có sử dụng LLM theo lineage dù runner cuối
> không gọi lại LLM.
