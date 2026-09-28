# Hướng dẫn chạy end-to-end V16b — bản 0.9210 đã khóa

Đây là toàn bộ luồng tái lập cho submission LegalIR đã đạt **0.9210** trên
public leaderboard. Đây là tài liệu release của recipe đã được xác nhận, **không
phải** hướng dẫn tune theo leaderboard hay mở lại locked holdout.

## Kết quả đã khóa

| Hạng mục | Giá trị |
| --- | ---: |
| Public leaderboard snapshot | **0.9210** |
| Recall@5 trên locked holdout | **0.9228333333** |
| V11h anchor trên locked holdout | `0.9188333333` |
| Chênh lệch V16b trên locked holdout | **`+0.0040000000`** |
| Submission hợp lệ | `Task1 - LegalIR/output/v16b_memory_slot5.zip` |
| SHA-256 của ZIP | `B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9` |
| Tổng tham số end-to-end | BQB `330M` + Vietnamese reranker `760M` + selector <= `10K` = `1.09001B` |

Pipeline hợp lệ là **V11h + V16b**, không phải chỉ V16b. V11h tạo candidate
pool, điểm tài liệu và top-5 anchor; V16b chỉ là gate cuối rất hẹp trên anchor
đó.

## Contract bắt buộc của release

- Không dùng nhãn gán tay và không đọc public labels.
- Candidate set là **c35** đã được V11h chấm điểm. V16b không thêm, xóa, parse
  lại hoặc tự thay tài liệu ứng viên.
- Hạng 1--4 của V11h được giữ byte-for-byte. V16b chỉ được phép thay **slot 5**.
- Nếu V16b không duyệt swap, nguyên top-5 của V11h phải được giữ lại.
- QREL-memory được dựng chỉ từ phía train không chứa query đích hoặc query trùng
  sau chuẩn hóa. Đây là feature học có kiểm soát leakage, không phải nhãn thủ
  công.
- Locked holdout là bằng chứng lịch sử cho recipe đã khóa. Không dùng nó để chọn
  threshold, candidate pool hay model mới.
- Các thử nghiệm V18 trở đi như selector/LoRA/E3B không thay thế được release
  này.

## Điều kiện đầu vào và các path cần giữ

Chạy lệnh tại thư mục gốc repository, `D:\projects\UIT DSC`. Release dùng
Windows/PowerShell. Runner sẽ ưu tiên
`D:\projects\UIT DSC\.python312-embed\python.exe` nếu có, nếu không thì dùng
`python` trong `PATH`.

Cài dependency nếu môi trường chưa được chuẩn bị:

```powershell
python -m pip install -r 'Task1 - LegalIR\requirements-core.txt' -r 'Task1 - LegalIR\requirements-neural.txt'
```

Để rebuild đầy đủ, các input sau phải tồn tại:

```text
Task1 - LegalIR/data/raw/public_test/
  train.json
  public-official.json
  selected-contexts.zip

Task1 - LegalIR/data/processed/
  v11_research_fit_256/
  v11_research_256/
  v11_research_complement_256/

Task1 - LegalIR/experiments/track_b_v11_research/artifacts/
  holdout_1000.json
  train_fit_excluding_holdout.json
  complement_split/

models/bqbbao6_vietnamese-legal-embedding/
models/thanhtantran_Vietnamese_Reranker/
```

Danh sách đầy đủ các output/cache cần giữ nằm trong
[MANIFEST.md](MANIFEST.md). Giữ nguyên cả thư mục artifact: runner mặc định tái
sử dụng cache hợp lệ; chỉ `-Force` mới buộc chạy lại.

## Sơ đồ toàn bộ pipeline

```text
Official corpus + questions
        |
        v
V11h stage 1: truy hồi legal lexical/chunk/citation + BQB dense retrieval
        |
        v
Corpus-only PRF + bổ sung legal citation (chỉ safe tail)
        |
        v
V11h stage 2: Vietnamese cross-encoder chấm c35
        |
        v
Feature-v2 chuẩn hóa theo query + tiny LTR grouped-OOF + exact-heading rescue
        |
        v
V11h top-5 anchor đã khóa
        |
        +--> V16a: QREL-memory fit-only cho complement / holdout / public
                    |
                    v
V16b: pointwise-logistic gate grouped-OOF, chỉ được thay slot 5
        |
        v
Kiểm tra JSON và đóng gói submission.json thành v16b_memory_slot5.zip
```

## Giai đoạn 1–2: V11h tạo anchor

V11h là pipeline retrieval/reranking/fusion nền.

1. **Truy hồi ứng viên.** Kết hợp legal FTS, chunk phrase, citation evidence và
   BQB Vietnamese legal dense retrieval.
2. **Bổ sung safe tail.** Corpus-only PRF và legal-citation completion chỉ được
   thêm ứng viên ở phần đuôi, không làm thay đổi thứ tự head của Stage 1.
3. **Chấm c35 bằng cross-encoder.** Vietnamese reranker chấm 35 tài liệu/query,
   dùng tối đa ba legal chunks/tài liệu, `max_length=768`, FP16.
4. **Fusion.** Feature-v2 chuẩn hóa neural score theo query, kết hợp provenance,
   rank agreement, exact phrase/title và tín hiệu tỷ lệ độ dài. Tiny
   pointwise-logistic LTR (`C=1`) được đánh giá grouped-OOF để tạo ranking V11h.
   Feature supervised `document_prior` bị tắt vì đã từng gây label leakage trên
   locked test.
5. **Rescue bảo thủ.** Exact-heading threshold 10 chạy sau LTR.

Output public V11h đã khóa là `Task1 - LegalIR/output/v11h.zip`; đây là
anchor/fallback của V16b. V11h đạt `0.9188333333` trên locked holdout và `0.9183`
trên public snapshot.

Khi chỉ cần rebuild V11h anchor, dùng:

```powershell
& 'Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1' -Device cuda
```

## V16a: dựng feature QREL-memory không leakage

V16a **không thay đổi prediction**. Nó tạo feature đầu vào mà V16b sẽ nối thêm
vào từng document row trong c35 cố định của V11h.

| Split cần dự đoán | Nguồn memory | Query đích |
| --- | --- | --- |
| Complement | các query locked holdout | các query clean complement |
| Locked holdout | `train_fit_excluding_holdout.json` | các query locked holdout |
| Public | `train.json` | `public-official.json` |

Mọi split đều dùng BQB với `max_seq_length=256`, `memory_k=50`,
`temperature=0.07`, `degree_exponent=0.5`, `reciprocal_floor=0.25`,
`anchor_cutoff=35` và `memory_top_k=5`. Các query trùng sau chuẩn hóa bị loại.
V16a tạo ba file feature:

```text
Task1 - LegalIR/output/v16_rgm/complement_memory_features.json
Task1 - LegalIR/output/v16_rgm/holdout_memory_features.json
Task1 - LegalIR/output/v16_rgm/public_memory_features.json
```

Release runner sẽ chạy cả ba. Nếu cần chạy riêng từng split:

```powershell
& 'Task1 - LegalIR\experiments\track_b_v11_research\run_v16a_memory_oracle.ps1' -Split complement -Device cuda
& 'Task1 - LegalIR\experiments\track_b_v11_research\run_v16a_memory_oracle.ps1' -Split holdout -Device cuda
& 'Task1 - LegalIR\experiments\track_b_v11_research\run_v16a_memory_oracle.ps1' -Split public -Device cuda
```

## V16b: gate bảo thủ cho slot 5

V16b bắt đầu bằng top-5 V11h đã khóa, sau đó nối thêm 15 memory features cho
mỗi tài liệu trong c35. Các feature quan trọng gồm
`degree_corrected_memory_score`, memory rank, prototype/neighbor signals,
reciprocal-neighbor score và anchor margin. Selector pointwise logistic hiện có
được học trên complement bằng grouped OOF; các query exact-memory bị loại khỏi
cohort dùng để chọn policy.

Config đã được chọn và khóa:

```text
selector: pointwise_logistic, C=1.0
candidate limit: 35
policy: memory_slot5
memory feature: degree_corrected_memory_score
memory floor: 0.005
memory margin: 0.0
model margin: 0.0
exact-heading rescue: bật
```

Gate giữ nguyên V11h trừ khi challenger qua memory floor và điều kiện selector
không âm của policy đã khóa. Nó không được sửa các hạng 1–4.

Bằng chứng chọn/promotion lịch sử trong
`output/v16_rgm/v16b_memory_ltr_report.json`:

| Đánh giá | Anchor Recall@5 | V16b Recall@5 | Chênh lệch |
| --- | ---: | ---: | ---: |
| Clean-complement grouped OOF | `0.8787524366` | `0.8860623782` | `+0.0073099415` |
| Locked holdout | `0.9188333333` | `0.9228333333` | **`+0.0040000000`** |

Lần materialize public lịch sử đã đổi 165/1.000 top-5 từ V11h. JSON đầu ra duy
nhất của V16b là:

```text
Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json
```

## Rebuild đầy đủ bằng một lệnh

Trước tiên, xác minh artifact release hiện có chưa bị thay đổi:

```powershell
& 'Task1 - LegalIR\release\v16b_0921\verify_release.ps1' -VerifyHashes
```

Sau đó chạy toàn bộ V11h → V16a → V16b → đóng gói:

```powershell
& 'Task1 - LegalIR\release\v16b_0921\run_release.ps1' -Device cuda
```

Runner này thực hiện theo đúng thứ tự:

1. Chạy `run_v11h_end_to_end.ps1`.
2. Dựng feature V16a cho complement, holdout, rồi public.
3. Materialize V11h locked holdout anchor nếu nó bị thiếu.
4. Chạy `run_v16b_memory_slot5.ps1` với c35 và policy đã khóa.
5. Kiểm tra public answer có đúng 1.000 question IDs mong đợi và đóng gói ZIP
   chứa duy nhất `submission.json`.

Chỉ dùng `-Force` khi thật sự muốn rebuild có chủ đích. Nó ghi đè các artifact
V11h/V16 cacheable tại chỗ; vì vậy hãy copy output frozen trước nếu cần so sánh
trước/sau.

## Output cuối và xác minh trước khi nộp

File hợp lệ duy nhất là:

```text
Task1 - LegalIR/output/v16b_memory_slot5.zip
```

Trước khi nộp, chạy:

```powershell
& 'Task1 - LegalIR\release\v16b_0921\verify_release.ps1' -VerifyHashes
Get-FileHash 'Task1 - LegalIR\output\v16b_memory_slot5.zip' -Algorithm SHA256
```

Lệnh thứ hai phải trả về:

```text
B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9
```

Nếu ZIP rebuild có hash khác, hoặc report V16b không còn config/policy ở trên,
hãy xem nó là **thử nghiệm mới**, không phải release 0.9210 đã được chấp nhận.
Giữ nguyên ZIP frozen và review report trước khi đưa ra bất kỳ quyết định nộp
bài nào.

## Không được thay V16b bằng các artifact này

Không dùng V16d+ hoặc bất kỳ artifact V18–V25 nào thay cho ZIP frozen. Các nhánh
đó là nghiên cứu/diagnostic, có candidate hoặc evidence contract khác, hoặc
không vượt bằng chứng release của V16b. Lý do dừng promotion được lưu tại
[LESSONS_AND_BOTTLENECKS.md](LESSONS_AND_BOTTLENECKS.md).
