# LegalIR — huấn luyện lớp chấm điểm trên hidden states Qwen

Ngày lập: 16/09/2026. Đây là **một đề xuất nghiên cứu để tự triển khai**, chưa có kết quả huấn luyện, benchmark GPU hoặc điểm public của nhánh này. Không có cơ sở để bảo đảm public đạt 0,95. Kế hoạch giới hạn chi phí và có đường đóng gói rõ ràng nếu vượt kiểm định; nó không biến một kết quả chưa đo thành lời hứa.

**1. Quyết định duy nhất**

Giữ backbone `Qwen/Qwen3-Reranker-0.6B` nguyên bản, lấy vector cuối của mỗi cặp query–document đã đọc evidence dài, rồi **học một lớp tuyến tính hiệu chỉnh điểm relevance**. Dùng toàn bộ c67 hợp lệ để đề cử slot 5. Giữ top 1–4 V16b và dùng dự đoán E3B đã được kiểm định làm mặc định khi không đủ bằng chứng thay thế.

Chi phí lớn là một lượt trích đặc trưng cho complement và một lượt cho public sau khi pass. Nested OOF chỉ fit các lớp nhỏ trên GPU, không fit lại backbone theo fold. Không mở thêm một nhánh LoRA, verifier sinh A/B hoặc grid fusion lớn trong kế hoạch này.

Đây là lựa chọn theo **chi phí kiểm chứng trong thời gian còn lại**, không phải khẳng định lớp tuyến tính chắc chắn tốt hơn fine-tuning backbone.

**2. Bằng chứng và giả thuyết cần tách biệt**

| Điều đã biết | Ý nghĩa thực tế |
|---|---|
| R1: 0,898037 / 0,902214; E3B lịch sử: 0,898246 / 0,900961 | R1 thay đổi −0,000209 / +0,001253 so với E3B. Gain +0,012–0,016 so với V16b không phải gain mới so với bản mạnh nhất. |
| E/F thường lặp lại cùng challenger sai qua hai seed | Chỉnh ngưỡng đơn thuần khó sửa thứ tự sai; cần thêm tín hiệu có giám sát. |
| So sánh cùng 8 ID: evidence dài tốt hơn ngắn theo báo cáo của bạn | Giữ packer dài đã có kiểm chứng; không đồng thời viết lại cách đóng gói văn bản. |
| Oracle giữ top-4: c35 = 0,949206; c67 = 0,967335; shortlist-8 = 0,947354 trên complement | c67 có dư địa, nhưng chỉ lấy 8 ID cũ lại bỏ mất gần 0,02 oracle. Không dùng oracle làm dự báo public. |

Hai oracle c35/c67 đã được đối chiếu từ dữ liệu complement đính kèm trong phiên làm việc. Các con số E3B, E/F, E4 và R1 mới nhất ở trên dựa vào báo cáo bạn cung cấp, không phải lần chạy lại trên server.

Qwen chấm relevance bằng chênh lệch logits `yes` và `no`. Đó là một phép đọc tuyến tính từ vector cuối. **Giả thuyết mới**: vector này còn giữ các đặc trưng giúp phân biệt đáp án luật đúng mà phép đọc có sẵn chưa sử dụng tốt. Một lớp học từ nhãn LegalIR có thể khai thác chúng. Chưa có bằng chứng rằng giả thuyết này đúng với các lỗi hiện tại. [Model card Qwen chính thức](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

Không được gọi nhánh này là “sửa chắc chắn bottleneck cuối”. Nếu biểu diễn đã mất điều kiện pháp lý cần thiết, đổi lớp đọc không khôi phục được thông tin đó. Nghiên cứu về linear probing cũng cho thấy giữ nguyên biểu diễn có thể hữu ích khi chuyển miền, nhưng kết quả đó không phải bằng chứng tăng Recall cho LegalIR. [Kumar và cộng sự, ICLR 2022](https://arxiv.org/abs/2202.10054).

**3. Preflight trước khi tiêu GPU**

- Kiểm tra S2 cũ nhận những feature nào. Nếu nó đã học chính vector 1024 chiều ở cùng vị trí, cùng loss và cùng OOF như đề xuất này, đây không còn là hướng mới; không chạy lại dưới tên mới.
- Xác nhận checkpoint Qwen là bản gốc được BTC duyệt, chưa fine-tune trên complement. Không dùng E4 deployment weights để tạo feature OOF: chúng đã thấy toàn bộ 1197 query.
- Tái sử dụng chính manifest T/C/V, hai seed và bốn chế độ chia của R1/E3B. Kiểm tra overlap nhóm theo chế độ chia tương ứng, không tự chia random lại.
- Xác nhận có prediction E3B cho C và V theo đúng lineage từng fold. Nhánh `abstain` phải tái hiện prediction đó chính xác. Không thay bằng prediction của hai E4 deployment models trên complement.
- Đọc c67 từ manifest đã khóa; giữ cả incumbent trong pool. Kiểm tra ID thực tế thay vì suy từ tên file. Không tự thêm gold vào candidate universe của C/V/public.
- Chạy `validate_model_plan()` theo registry hiện hành, ghi số tham số thực. R1 báo tổng 3.205.786.512; thêm một vector và bias 1025 tham số vẫn dưới 4B nếu giữ nguyên các thành phần còn lại. Đây là phép cộng, chưa thay cho validator trên server.
- Chốt protocol/release của **nhánh mới** trước khi xem kết quả mới. Không sửa gate R1 đã fail, contract E4, ledger holdout hoặc trạng thái E/F/PV1.

**4. Cache đặc trưng: phần GPU cần triển khai**

Đầu vào là packer dài 4096 đã dùng ở F/R1: cùng instruction, suffix, tokenizer, truncation, special tokens và cách xử lý body rỗng. Freeze toàn bộ Qwen; `eval()` và inference không gradient.

Cho mỗi `(qid, doc_id)`, lưu:

```text
qid, doc_id, candidate_rank, evidence_valid
packed_text_sha256, input_ids_sha256
h_last[hidden_size], native_yes_minus_no
model_sha256, tokenizer_sha256, packer_sha256, pool_sha256
```

`h_last` là hidden state cuối đã qua normalization của decoder, tại token cuối không phải padding — chính vị trí dùng để chấm `yes/no`. Không lấy vector riêng của document, không lấy mean embedding từ embedding model khác. Cấu hình công khai hiện có `hidden_size=1024`; đọc lại cấu hình checkpoint local và assert. [Cấu hình Qwen](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/main/config.json).

Chỉ lấy output tầng cuối, không bật việc giữ toàn bộ hidden states của 28 tầng. Backbone API trả `last_hidden_state`; đường CausalLM có thể hạn chế logits ở token cuối. Phải kiểm tra phiên bản local trước khi dùng API. [Tài liệu Transformers Qwen3](https://huggingface.co/docs/transformers/en/model_doc/qwen3).

Với `w0 = lm_head[yes] - lm_head[no]`, kiểm tra trên một batch rằng `w0 @ h_last` khớp chênh lệch logits native; cộng chênh lệch bias nếu kiến trúc có bias. Sai lệch dtype phải được đo trên pilot và ghi lại. Khóa tolerance trước khi đọc nhãn; không nới tolerance theo kết quả recall. Nếu sai prompt, vị trí token hoặc scale, dừng để sửa parity trước khi cache toàn bộ.

Cache scalar scores hiện tại **không thể suy ngược ra hidden states**. Nếu chưa lưu vector thì cần forward lại. Với 1197 × 67 × 1024 giá trị 2 byte, phần vector khoảng **156,6 MiB**, chưa tính metadata; đây không phải dự toán VRAM khi forward. Kiểm tra sai lệch khi ghi/đọc dtype; có thể giữ float32 nếu cần, khoảng 313,3 MiB.

Tất cả query được cache bằng backbone gốc không sử dụng nhãn. Do đó có thể dùng chung cache này cho các fold. Những thứ có học từ nhãn — head, lựa chọn negative, policy — vẫn phải nằm đúng T/C/V.

**5. Công thức huấn luyện cụ thể**

Một head duy nhất, không thêm MLP ở lần chạy này:

\[
x_{qd}=h_{qd}/\max(\|h_{qd}\|_2,10^{-8}),\qquad
s_{qd}=z^0_{qd}+w^\top x_{qd}+b.
\]

`z0` là raw logit `yes−no`; khởi tạo `w=0, b=0`. Không dùng QID, doc ID, source-family ID hoặc nhãn kiểu luật làm feature. Đây là học cách đọc vector có ngữ cảnh, không phải chỉ hiệu chỉnh một scalar Qwen.

Tạo ví dụ **chỉ từ T**:

1. Với từng query, lấy mọi gold có mặt trong c67 hợp lệ làm positive. Có thể dùng cả gold nằm trong top-4 để học relevance; inference vẫn khóa top-4.
2. Lấy tối đa 8 negative không nằm trong toàn bộ gold list: 4 tài liệu có native Qwen score cao nhất và 4 tài liệu từ phần giữa thứ hạng còn lại, chọn tất định theo seed/QID. Deduplicate ID. Không dùng danh sách destructive của V/stress để tạo batch.
3. Query không có positive trong pool vẫn tham gia pointwise loss với negative. Không bỏ hết các query “không có cơ hội cứu”. Không coi cặp hai tài liệu cùng đúng hoặc cùng sai là cặp rescue/destructive.
4. Không tự gán nhãn câu hay đoạn con từ nhãn document. Không truyền nhãn giữa các document chỉ vì cùng họ luật.

Với `Pq` là positive và `Nq` là negative được lấy mẫu:

\[
L_q=\tfrac12\operatorname{mean}_{p\in P_q,n\in N_q}
\operatorname{softplus}(s_{qn}-s_{qp})
+\tfrac12\operatorname{mean}_{d\in P_q\cup N_q}
\operatorname{BCEWithLogits}(s_{qd},\mathbf1[d\in G_q]).
\]

Nếu không có cặp positive–negative, đặt thành phần pairwise bằng 0. Query không có document hợp lệ được ghi audit và không đưa vào optimizer.

\[
L_T=\frac{\sum_{q\in T}|G_q|^{-1}L_q}{\sum_{q\in T}|G_q|^{-1}}
+\lambda(\|w\|_2^2+b^2).
\]

Trọng số `1/|Gq|` phù hợp mức thay đổi Recall do một tài liệu, nhưng loss vẫn là surrogate, không phải tối ưu chính xác metric. Đánh giá phải dùng scorer macro Recall@5 chính thức, giữ nguyên denominator gold list.

Khóa trước ba mức regularization `lambda = {1e-4, 1e-3, 1e-2}`. Đây là cấu hình khởi đầu đề xuất, chưa phải hyperparameter đã được chứng minh. Fit head bằng full-batch L-BFGS float32 trên GPU, tối đa 200 iterations, kiểm tra finite loss/gradient và ghi convergence. Không early-stop theo V. Nếu implementation L-BFGS chưa có sẵn, việc thay optimizer phải chốt trước lượt đánh giá đầu, không thay sau khi thấy V.

Với feature cố định, loss này lồi theo head. Dùng một lớp giúp việc fit lại nhanh và dễ kiểm soát; không bảo đảm generalization.

**6. Selector và nested OOF**

Gọi `i` là slot 5 của **E3B baseline đúng fold**. Xếp các ứng viên hợp lệ ngoài top-4 bằng `s`; chọn `c` cao nhất. Đề xuất thay khi đồng thời:

```text
c != i
s(c) > 0
s(c) - s(i) > tau
```

Không đủ điều kiện thì giữ `i`. Điều kiện `s(c)>0` chỉ là quy tắc score đã khóa; không diễn giải thành xác suất đúng được bảo đảm. Incumbent thiếu evidence, cache thiếu dòng cần thiết hoặc provenance sai thì abstain/fail preflight theo nguyên nhân; không âm thầm đổi ID để bù.

`tau ∈ {0, 1, 2}`. Trên C chỉ chọn giữa 9 cặp `(lambda,tau)` và một lựa chọn **giữ nguyên E3B toàn bộ**. Tối đa 10 policy, không thêm grid khi kết quả xấu. Chọn theo macro Recall C; hòa thì ít destructive hơn, rồi ít thay thế hơn, rồi regularization mạnh hơn và tau lớn hơn. Chốt quy tắc tie-break trước khi chạy.

Trình tự mỗi fold: fit ba head trên T → chọn một policy trên C → khóa → score V. Tập C không được nhập vào optimizer. Tất cả negative mining có nhãn chỉ dùng T. Mọi fold giữ đúng nhóm loại trừ và seed đã có.

Standard 10 fold và stress 30 fold tương ứng **120 head fits nhỏ** cho ba lambda, không phải 120 lần huấn luyện Qwen. Không dùng weights của head khác fold để “resume” vì đã học nhãn khác; chỉ resume chính fold/config/checkpoint khớp manifest.

**7. Phép đối chứng bắt buộc — biết gain đến từ đâu**

Trên đúng cùng c67 và pack dài, ghi ba kết quả:

- E3B baseline nguyên bản theo fold.
- Qwen native không học head, với cùng họ rule abstain và chọn tau trên C.
- Qwen có head học từ T, với policy đã chọn trên C.

So sánh head với native cùng pool để tách lợi ích học tín hiệu khỏi lợi ích mở pool. Native ở đây là đối chứng được định trước, không tự thành nhánh nộp nếu head thất bại. Không dùng tăng điểm so với V16b để che việc chưa vượt E3B.

Ghi theo query: incumbent, challenger, gold membership dùng riêng khi đánh giá, delta Recall, policy, score, fold và hash. Báo cáo cả rescue/destructive/neutral; đừng xem hàng trăm thay thế trung tính là hàng trăm cải thiện.

Để gọi đây là bước tiến có cơ sở, tối thiểu phải:

- Vượt mọi gate đang áp dụng cho nhánh mới, bao gồm so E3B lịch sử theo từng seed; không hạ gate R1 vì chênh lệch nhỏ.
- Tăng ít nhất 0,003 Recall so E3B ở cả hai standard seed, paired-bootstrap lower bound dương, và không có seed/regime âm ở ba stress regime; đồng thời rescues phải hơn destructive. Nếu gate hiện hành chặt hơn, dùng gate chặt hơn.
- Không gộp cùng QID xuất hiện qua hai seed thành hai quan sát độc lập khi tính độ tin cậy. Stress bootstrap theo nhóm phù hợp. Bootstrap sau nhiều vòng nghiên cứu chỉ là kiểm tra hỗ trợ, không biến complement đã xem nhiều lần thành test mới.
- Duy trì top-4, năm ID khác nhau, đúng pool, đúng provenance và format.

Mức +0,003 là ngưỡng nghiên cứu tối thiểu đề xuất, **không phải tín hiệu đã gần public 0,95**. Nếu muốn gọi là cải thiện lớn, cần thấy cỡ +0,01 trở lên so E3B trên complement và kết quả stress cùng chiều. Ngay cả vậy cũng không có công thức cộng gain này vào 0,9252.

**8. Triển khai cuối và quyền nộp**

Nhánh này dùng c67 và head mới nên **không thừa hưởng pass E4/E5**. Contract mới cần được chốt trước khi chạy: cho phép một candidate release dựa trên nested standard + toàn bộ stress của nhánh mới, không mở lại holdout đã dùng. Nếu quy trình dự án bắt buộc một holdout độc lập mới thì phải giải quyết điều kiện đó trước khi đầu tư GPU; không xóa ledger hoặc gọi lại `MODE=parity` cũ.

Đề xuất recipe final khóa từ đầu: lấy cặp `(lambda,tau)` xuất hiện nhiều nhất trong 10 quyết định C standard; tie-break như trên; nếu `abstain` thắng thì không phát hành head. Fit một head mới trên toàn bộ 1197 complement với lambda đã chọn, giữ tau đó. Không chọn config bằng kết quả V/stress/public. Final refit sử dụng nhiều dữ liệu hơn từng T, vì vậy phải ghi rõ chưa có xác nhận holdout mới cho model final này.

Sau khi đủ gate: cache đặc trưng public label-blind, áp dụng head final trên c67, dùng E3B/E4 deployment pipeline hiện có để tạo incumbent public. Không dùng checkpoint E4 cho việc huấn luyện/đánh giá complement OOF. Lưu candidate ZIP riêng bằng packager đã được BTC chấp nhận; giữ nguyên ZIP từng đạt 0,9252 để dự phòng. Không tự suy tên file trong ZIP từ tài liệu không đầy đủ.

Report cần giữ `new_policy_holdout_validated:false`; ghi rõ căn cứ release là protocol mới, không ghi E4 pass cho model chưa qua E4. Việc user cho phép phát triển nhánh mới không làm thay đổi kết quả các nhánh đã fail.

**9. Deadline: tính cả public, không chỉ tính train**

Đặt `D` là giờ đóng cổng chính xác kèm múi giờ. Tính giờ còn lại thật, không mặc định vẫn đủ 48 giờ vì cuộc trao đổi bắt đầu từ hai ngày trước. Giữ thời gian đóng gói và chờ chấm theo lịch sử thực của BTC.

Đo pilot không nhãn trên một mẫu QID cố định theo hash, gồm các pack dài/ngắn. Đo thời gian wall-clock toàn tuyến: tokenize, forward, ghi cache. Lấy tốc độ bảo thủ từ nhiều batch; nếu shard trên hai GPU thì đo tổng thực, không nhân đôi tốc độ một GPU. Auto-probe chỉ batch và giới hạn VRAM; không tự giảm max length hoặc đổi precision sau khi protocol đã khóa. Giữ headroom cho GPU dùng chung, không dừng process người khác.

Ví dụ sau giả định complement 1197, public 1000 query, 67 document/query. **Public 1000 chỉ là số minh họa, thay bằng số thật.**

| Tốc độ đo được | Cache complement | Cache public | Tổng cache với hệ số dự phòng 1,5 |
|---|---:|---:|---:|
| 2 pairs/s | 11,14 giờ | 9,31 giờ | 30,67 giờ |
| 4 pairs/s | 5,57 giờ | 4,65 giờ | 15,33 giờ |
| 8 pairs/s | 2,78 giờ | 2,33 giờ | 7,67 giờ |

Các số là phép chia khối lượng cho throughput, không phải benchmark nhánh mới. Hệ số 1,5 cũng không bảo đảm chống mọi biến động GPU dùng chung.

Dự trù thêm **12 giờ ngoài cache** làm ngân sách ban đầu: 4 giờ sửa exporter/evaluator và kiểm tra, 2 giờ fit head/standard/stress/report, 6 giờ final refit/đóng gói/upload/chờ chấm. Thay bằng thời gian thực nếu đã có tooling hoặc BTC chấm lâu hơn. Nếu implement mất quá 4 giờ, cập nhật dự toán trước khi tiếp tục.

Điều kiện bắt đầu full cache:

\[
1,5\frac{N_{comp}+N_{public}}{3600r}+B_{ngoài\ cache}
\leq H_{còn\ lại}.
\]

Ví dụ 24 giờ còn lại với ngân sách ngoài cache 12 giờ cần khoảng **5,12 pairs/s** cho 147.199 pairs; 4 pairs/s không đủ theo ngân sách này. Không được cam kết “kịp” chỉ dựa vào tên A100 80GB.

Mốc vận hành: preflight/pilot → cache complement → standard → stress → final refit → cache public → kiểm tra ZIP → nộp trước khoảng đệm dành cho BTC. Cache/fit hết ngân sách mà chưa đủ gate thì nhánh mới không được ghi PASS. Không đánh đổi việc có bài hợp lệ để chạy hết một job nghiên cứu.

**10. Lưu lại để không mất weights lần nữa**

Mỗi cache shard ghi atomically, checksum và coverage `(qid,doc_id)`; không trộn checkpoint hoặc packer. Mỗi head fit lưu `w,b`, lambda, train-QID hash, split/group manifest hash, optimizer state nếu cần resume, convergence, native-checkpoint hash, feature normalization và version code. Lưu riêng policy C của fold và report V, giữ cả head không thắng vì dung lượng rất nhỏ.

Final lưu head, tokenizer/base references kèm hashes, cấu hình packing, policy, model-plan, prediction và ZIP checksum. Tắt mọi cờ `save_weights=false`. Cache vector và head đều phải được sao lưu ra nơi bền vững trước khi dọn working tree.

**11. Việc cần giao ngay cho người triển khai**

> Tạo nhánh mới từ pipeline đang giữ được parity E3B. Thêm exporter lấy `h_last` từ Qwen3-Reranker-0.6B gốc, pack dài 4096, pool c67 manifest cố định. Chứng minh native-logit parity trước. Dùng nested engine R1 để fit residual linear head trên T, chọn 9 config + abstain trên C, đánh giá V theo đúng standard và ba stress regime. So trực tiếp E3B và native Qwen cùng c67. Lưu mọi head/cache/report. Chốt release mới và ngân sách deadline trước khi mở GPU full cache; không mở lại holdout, không sửa các report đã fail, không dùng deployment weights cho OOF. Báo cáo pilot phải có throughput, projected total gồm public, và giờ cuối phải đóng gói.

Kết quả của kế hoạch là một phép kiểm tra rẻ hơn nhiều so với train lại backbone qua mọi fold, nhằm xác định liệu có thể học cách đọc đúng tín hiệu đã tồn tại. **Chỉ điểm chạy thật mới trả lời được nó có tăng và có đạt 0,95 hay không.**
