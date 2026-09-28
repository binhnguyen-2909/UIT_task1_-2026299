# LegalIR: hướng phát triển E3 sau E2 MIL

**Khuyến nghị chính: triển khai E3-B dùng đầy đủ score MIL trong c35, so từng ứng viên với tài liệu đang ở slot 5, và hiệu chỉnh quyết định thay thế bằng nested OOF có kiểm soát phụ thuộc nhãn.** Giữ E1 evidence, kiến trúc và loss E2 cho lần thử tiếp theo. Đây là một nhánh nghiên cứu mới; E3 hiện tại chưa qua gate và E4 vẫn khóa.[^1]

E2 cho tín hiệu đáng giữ lại: cả tám tổ hợp seed/regime đều tăng Recall@5 complement so với V16b. Tuy nhiên, E2 được xếp lại toàn bộ năm vị trí, còn E3 chỉ được sửa vị trí cuối. Điểm tăng của E2 vì thế chưa phải bằng chứng rằng một policy slot 5 sẽ tăng tương ứng. Nút thắt có thể kiểm tra ngay nằm ở cách E3 nén và sử dụng đầu ra E2.[^2][^3]

| Phát hiện | Bằng chứng | Ý nghĩa triển khai |
|---|---|---|
| Tín hiệu E2 bị nén mạnh | E3 nhận top-5 ID, lấy tối đa một đề cử mỗi seed | Xuất score của đủ 35 tài liệu trước khi đổi model |
| Thiếu đối chiếu với slot 5 | Ứng viên có thể được chọn dù incumbent đứng cao hơn trong E2 | Quyết định phải dựa trên chênh lệch ứng viên - incumbent |
| Calibration ít ca quyết định | 16/29 fold chọn policy chỉ có tối đa hai ca cứu trên calibration | Thu nhỏ họ policy; đánh giá độc lập toàn bộ chuỗi |
| Có giới hạn cấu trúc | Oracle slot 5 complement là 0,949206 | Không đặt lời hứa 0,95 trên nhánh giữ nguyên c35/top-4 |

**Phạm vi bằng chứng.** Các thống kê mới là phép đối chiếu code, report và dự đoán complement đã có, trên 1.197 query khớp anchor. Ba hash của anchor, stage 1 và stage 2 khớp contract E3. Không có kết quả huấn luyện E3-B, holdout mới hoặc private trong báo cáo này. Oracle là phép tính dùng nhãn để xác định giới hạn, hoàn toàn không phải thuật toán suy luận.[^3][^4]

Public 0,9210 là snapshot trong kế hoạch bàn giao. Chưa có bằng chứng để dự báo chắc chắn 0,93, 0,94 hay 0,95 trên public/private. Mục tiêu gần nhất có thể kiểm chứng là một E3-B có gain dương và số ca cứu lớn hơn số ca hại ở cả hai inner seed, trong standard và ba stress regime; sau đó mới đến một lần kiểm tra E4 đã khóa.[^1]

Ràng buộc triển khai giữ nguyên: c35 cố định; top-4 V16b sao chép đúng thứ tự; năm ID khác nhau; abstain trả lại V16b; E1-E3 chỉ dùng complement; pipeline suy luận dưới 4B tham số. A100 80GB dùng chung được xử lý bằng probe bộ nhớ thực tế, không mặc định có quyền sử dụng đủ 80GB.

<!-- pagebreak -->

## Luồng dữ liệu và kết quả E2

V16b là anchor E0. Nhánh E1 xây evidence pháp lý cho đúng c35 có sẵn: chia theo Điều/Khoản/Điểm, tạo cửa sổ 384 reranker token có ngữ cảnh cha, hợp nhất đề xuất lexical, BQB và structure/citation, rồi MMR chọn ba block và mở block thứ tư theo điều kiện bất định đã khóa. Đây là xử lý không dùng nhãn truy vấn để chọn đoạn.[^1][^2]

E2 fine-tune reranker tiếng Việt 760M trên các bag gồm 3-4 passage. Score tài liệu là temperature-logsumexp với nhiệt độ 0,5. Loss gồm 0,55 document relevance BCE, 0,30 pairwise boundary và 0,15 regularizer score anchor so với frozen E1 CE. Regularizer này không khóa thành viên top-5. Đầu vào query-passage của E2 có `max_length=768`; không được nhầm với cửa sổ evidence 384 token.[^2]

| Tầng | Đặc điểm đã xác nhận | Quyết định cho E3-B |
|---|---|---|
| Corpus và stage 1/2 | Candidate và anchor có hash đã khóa | Giữ nguyên corpus mapping, c35 và thứ tự anchor |
| E1 evidence | 41.895 document rows; 142.205 passage được chọn; 44 row body rỗng | Tái dùng manifest; giữ empty-mask, không tạo đoạn giả |
| E2 training | 1.150 query trainable, 10.423 document examples | Giữ loss, negative mining và checkpoint khởi tạo |
| E2 export | Report chỉ giữ dự đoán top-5 | Thêm đủ c35 logits, passage logits và lineage |
| E3 selector | Sửa một slot với fallback E0 | Đổi sang chênh lệch score so với incumbent |

| Regime | Gain seed 20260908 | Gain seed 20260909 |
|---|---:|---:|
| Standard | +0,7797 điểm % | +0,7867 điểm % |
| Law/source | +0,5291 điểm % | +0,9886 điểm % |
| Query/template | +1,1626 điểm % | +1,4133 điểm % |
| Near-duplicate | +1,1209 điểm % | +1,6987 điểm % |

Mọi gain trong bảng so với Recall@5 complement anchor **0,8860624**. Đây là kết quả E2 được xếp lại cả top-5, chưa áp ràng buộc top-4 của E3. Những dòng không có positive trong c35 bị bỏ khỏi fit E2 nhưng vẫn phải tồn tại trong đánh giá và đầu ra; không giảm mẫu số để làm đẹp điểm.[^2]

MIL phù hợp với nhãn ở cấp tài liệu vì nhãn dương của một bag không có nghĩa mọi passage đều dương. Nghiên cứu MIL có cơ sở cho cách biểu diễn này, nhưng kết quả trên benchmark khác không cung cấp mức gain dự kiến cho LegalIR.[^10] Việc E1 frozen thấp hơn anchor và E2 tăng là lý do thực nghiệm để giữ khối học sâu hiện có trong lần sửa E3 đầu tiên.

<!-- pagebreak -->

## Vì sao E3 hiện tại không qua gate

Hàm `_propose_slot5()` duyệt top-5 của mỗi seed, bỏ bốn ID đã khóa và bỏ cả incumbent, lấy ứng viên đầu tiên rồi dừng. Sau đó, thiếu memory row là loại; vượt memory-rank threshold là loại; thứ tự ưu tiên là seed support, memory score, rồi E2 rank. Code không kiểm tra liệu ứng viên tốt hơn incumbent theo E2 hay không.[^3]

| Regime | Inner seed | Số câu sửa | Cứu / hại | Gain Recall |
|---|---:|---:|---:|---:|
| Standard | 20260910 | 33 | 5 / 5 | +0,000418 |
| Standard | 20260911 | 37 | 5 / 3 | +0,001671 |
| Law/source | 20260910 | 13 | 1 / 3 | -0,002089 |
| Law/source | 20260911 | 20 | 3 / 3 | -0,000418 |
| Query/template | 20260910 | 124 | 10 / 12 | -0,002228 |
| Query/template | 20260911 | 100 | 12 / 7 | +0,003620 |
| Near-duplicate | 20260910 | 60 | 5 / 5 | 0,000000 |
| Near-duplicate | 20260911 | 43 | 4 / 5 | -0,000418 |

Đơn vị cột cuối là Recall thô: +0,001 tương đương +0,1 điểm phần trăm. Standard seed 20260910 tăng nhẹ nhưng cứu = hại nên không đạt safety gate. Một câu nhiều nhãn đóng góp mỗi ID ít hơn một câu chỉ có một nhãn; vì thế số câu cứu bằng số câu hại vẫn có thể cho gain khác 0.[^3][^5]

Họ policy hiện tại gồm 40 cấu hình và abstain. Calibration chỉ có 158-226 query mỗi fold; 29/40 fold chọn một policy hoạt động, trong đó 16 fold có tối đa hai successful rescues trên calibration. Đây là môi trường dễ chọn nhầm một cấu hình do phương sai lớn. Cawley và Talbot chỉ ra rằng bản thân quá trình chọn mô hình trên một ước lượng hữu hạn cũng có thể overfit.[^4][^6]

Hai seed cũng không phải hai nguồn bằng chứng độc lập: chúng dùng cùng kiến trúc, corpus, evidence và phần lớn dữ liệu học. Seed agreement giúp đo độ ổn định nhưng không tự biến thành xác suất ứng viên đúng. Các stress regime đổi dấu cho thấy mức ổn định cần cho promotion chưa đạt.

Kết luận chẩn đoán là **mất thông tin, thiếu đối chiếu incumbent và calibration ít ca quyết định**. Những dấu hiệu này chưa chứng minh model chỉ học thuộc đáp án. Chúng đủ để bác bỏ việc mở E4 ngay, nhưng cũng chưa bác bỏ khả năng dùng tốt hơn biểu diễn MIL của E2.

<!-- pagebreak -->

## Giới hạn điểm và phần cơ hội bị loại sớm

Với query q, gọi F là bốn ID cố định, i là incumbent slot 5, G là tập nhãn đúng và C là tập ứng viên cho slot cuối. Oracle chọn lựa tốt nhất giữa giữ nguyên anchor và thay một ID từ C ngoài F. Công thức dùng đúng macro Recall của tác vụ:[^4][^5]

```
R_anchor(q) = |(F union {i}) intersect G| / |G|
R_oracle(q,C) = max(R_anchor(q),
                   (|F intersect G| + min(1, |(C - F) intersect G|)) / |G|)
```

| Tập ứng viên xét cho slot 5 | Standard | Law/source | Template | Near-dup. |
|---|---:|---:|---:|---:|
| Hai đề cử E3 hiện tại | 52 câu | 48 câu | 55 câu | 55 câu |
| Hợp toàn bộ top-5 hai seed E2 | 66 câu | 64 câu | 68 câu | 71 câu |
| Toàn bộ c35 | 89 câu | 89 câu | 89 câu | 89 câu |

Các số trên là query có thể tăng Recall nếu được chọn bằng oracle, trước bộ lọc memory/rank. Chỉ việc thôi lấy một đề cử mỗi seed đã mở thêm **13-16 query về mặt khả năng** trong hợp top-5. Chưa có phép đo cho thấy một selector thực tế sẽ cứu được toàn bộ số này; bỏ bộ lọc cũng có thể làm tăng ca hại.[^4]

| Oracle Recall@5 | Standard | Law/source | Template | Near-dup. |
|---|---:|---:|---:|---:|
| Đề cử hiện tại + giữ anchor | 0,92547 | 0,92296 | 0,92658 | 0,92658 |
| Hợp top-5 + giữ anchor | 0,93452 | 0,93326 | 0,93514 | 0,93723 |
| c35 + giữ anchor | 0,94921 | 0,94921 | 0,94921 | 0,94921 |

Có bằng chứng trực tiếp cho vấn đề incumbent: ở query/template seed 20260910, 17/124 ca sửa chọn một ứng viên đứng sau incumbent trong mọi seed E2 có xếp ứng viên đó ở top-5. Trong 17 ca này có năm ca hại và một ca cứu. Một veto suy ra sau khi thấy kết quả không được coi là policy đã kiểm chứng; E3-B phải đánh giá lại từ đầu theo protocol mới.[^4]

Trần **0,94920635** chỉ áp dụng cho complement, với đúng anchor, qrels và c35 đang xét. Nó không phải trần public/private. Với 1.000 câu public, đi từ 0,921 lên 0,950 cần thêm 29 đơn vị recall-query ròng; chỉ khi các câu liên quan có một nhãn mới có thể diễn giải đơn giản thành 29 câu sửa đúng không bị bù trừ bởi ca hại.

Oracle chỉ dùng để quyết định có đáng nghiên cứu nhánh hay không. Không dùng oracle, QID, gold hit hay nhãn ngoài partition được phép làm feature suy luận.

<!-- pagebreak -->

## E3-B: chọn phần thay thế bằng score MIL đầy đủ

**Đề xuất thuật toán, chưa có kết quả thực nghiệm.** Giữ hai reranker E2 760M và cho chúng score đủ c35. Gate CPU dùng chênh lệch score ứng viên - incumbent, độ bất đồng giữa hai seed và một memory prior bị chặn biên. Gate không cần một mạng sâu mới trên số ca cứu ít ỏi; phần học ngữ nghĩa vẫn nằm trong E2 MIL.

Với mỗi seed s, chuẩn hóa score trong cùng query bằng median và IQR của các tài liệu có evidence hợp lệ. Với mỗi ứng viên c ngoài top-4, tính chênh lệch so với i. Memory m(d) là 1/rank nếu có rank hợp lệ trong c35, ngược lại bằng 0. Mọi hằng số dưới đây là cấu hình khởi đầu cần đăng ký trước, không phải giá trị đã tối ưu trên kết quả E3.

```
z_s(d) = (score_s(d) - median_q(score_s)) / max(IQR_q(score_s), 1e-6)
gap_s(c) = z_s(c) - z_s(i)
mean_gap(c) = (gap_1(c) + gap_2(c)) / 2
U(c) = mean_gap(c) - 0.25 * abs(gap_1(c) - gap_2(c))
       + lambda * (m(c) - m(i))
```

Chỉ xét thay thế khi `mean_gap(c) > 0`. Điều kiện này chặn việc memory tự đẩy một ứng viên mà cả hai seed đều đánh giá thấp hơn incumbent. Hệ số 0,25 phạt bất đồng nhưng không bắt buộc hai seed phải cùng đề cử một ID ở top-5. Thành phần memory chỉ là prior mềm; thiếu memory của ứng viên không tự động loại nó.

Với mỗi lambda, chọn c có U lớn nhất trên **toàn bộ c35 ngoài top-4 và incumbent**, rồi áp threshold lên chính ứng viên đã thắng. Giữ nguyên V16b nếu U không vượt threshold, có tie với hành động giữ nguyên, hoặc incumbent không có evidence hợp lệ. Tài liệu rỗng không được đề cử. Missing score hoặc sai c35 là lỗi artifact cần chặn trước chạy; không tự bịa score để hoàn thành file.

| Thành phần | Cấu hình đăng ký cho lần đầu |
|---|---|
| Memory weight | lambda thuộc {0; 0,10} |
| Threshold | tau thuộc {0; 0,25; 0,50; 1,00} |
| Số policy | 8 cấu hình + abstain |
| Chọn nội bộ | Gain macro Recall > 0 và số ca cứu > số ca hại trên calibration |
| Tie giữa policy | Ít ca hại hơn; tau cao hơn; lambda nhỏ hơn |
| Đầu ra | Sao chép nguyên top-4; chỉ thay slot 5 bằng ID khác trong c35 |

Mục tiêu đánh giá mỗi phép thay là `delta_R = (1[c thuộc G] - 1[i thuộc G]) / |G|`. Nhãn chỉ được đọc để chọn policy trên calibration và để đánh giá outer fold. U là **điểm ưu tiên**, không phải xác suất đúng hay ước lượng gain đã hiệu chỉnh.

Selective prediction cung cấp nguyên tắc chấp nhận một phần quyết định và từ chối phần còn lại; SelectiveNet nghiên cứu cách học đồng thời hai việc này.[^7] E3-B chỉ vận dụng nguyên tắc abstention, không tuyên bố triển khai SelectiveNet hay thừa hưởng mức cải thiện từ bài báo. Tất cả lựa chọn c*, lambda và tau phải nằm trong chuỗi được đánh giá outer OOF.

<!-- pagebreak -->

## Nested OOF và phụ thuộc nhãn giữa E2 với E3

**Lỗ hổng cần xử lý.** E3 hiện chia calibration/outer-validation của riêng nó, nhưng sử dụng nguyên hai tập E2 OOF đã ghép toàn complement. Score của một query calibration có thể do model E2 học trên nhãn thuộc outer-validation hiện tại sinh ra. E3 chọn policy qua score ấy, tạo đường phụ thuộc gián tiếp vào nhãn outer-validation. Mỗi query có dự đoán OOF riêng chưa đủ bảo đảm toàn bộ stack có outer evaluation độc lập.[^2][^3][^11]

Code và preflight hiện không kiểm tra tập nhãn phụ thuộc của từng score. Đây là thiếu kiểm soát đã xác nhận; chưa có danh sách fit/early-stop QID của từng model trong handoff để định lượng chính xác số ca bị ảnh hưởng. Kết luận này không phủ nhận E2 được đánh giá riêng: E2 có outer split và early stopping bên trong outer-train. Vấn đề xuất hiện khi ghép các OOF output thành đầu vào chọn policy E3.

**Protocol E3-B đề xuất.** Trong mỗi regime, giữ năm outer group fold. Với từng outer fold và từng inner seed, thực hiện tuần tự:

1. Chia complement thành T = outer-train và V = outer-validation theo group.
2. Chia T theo group thành A = phần xây model và C = calibration E3, khoảng 20% T. Chia tiếp A thành A_fit và A_stop, khoảng 15% A cho early stopping.
3. Với hai neural seed đã khóa, khởi tạo E2 từ checkpoint gốc; gradient chỉ dùng A_fit, chọn epoch chỉ dùng A_stop. Giữ nguyên loss, evidence và cách lấy negative E2.
4. Đóng băng hai model; score đủ c35 của C và V. Chọn một trong tám policy hoặc abstain chỉ bằng C; áp policy đó lên V.
5. Ghép V của năm fold để đánh giá một OOF run. Lặp đúng hai inner seed cho standard và ba stress regime; không dùng kết quả outer để sửa grid trong cùng phiên bản.

Không refit E2 sau khi dùng C chọn gate trong outer fold này: như vậy C và V được score bằng cùng bộ trọng số, tránh dịch chuyển score do đổi model. Protocol này là nested sample splitting với gate nhỏ, không phải học thêm một meta-network cần nhiều tầng neural cross-fitting.

**Các tập phải kiểm tra bằng ID thực.** A_fit, A_stop, C và V không giao nhau về query/group theo regime. Tập nhãn dùng tạo score C không chứa nhãn C hoặc V; nhãn dùng chọn epoch không chứa C hoặc V. Tập nhãn dùng chọn gate không chứa V. Preflight phải kiểm tra cả quan hệ này, không chỉ các hash đầu vào.[^6]

Memory và anchor E0 cũng cần khai báo nguồn nhãn. Handoff không có file memory thực và toàn bộ lịch sử fit E0, nên chưa thể xác nhận tính độc lập của upstream chỉ từ checksum. Nếu memory prior không chứng minh được tách nhãn thì policy có lambda khác 0 không đủ điều kiện chạy. Việc giữ E0 làm baseline cố định cũng không tự chứng minh E0 là ước lượng end-to-end hoàn toàn chưa thấy nhãn.

Toàn complement đã được dùng cho nhiều lần nghiên cứu; các stress regime vì vậy là phép kiểm tra sức chịu đựng có giá trị, nhưng không thay thế một tập private độc lập. Các cải tiến sau khi xem report phải có phiên bản protocol mới và không được gọi outer score cũ là bằng chứng xác nhận cho policy mới.

<!-- pagebreak -->

## A100 80GB dùng chung: auto-probe và ngân sách

E2 hiện đã có `_memory_probe()` chạy forward, backward và AdamW step thật, nhưng chọn query trainable đầu tiên theo QID. Nó chưa phải probe đủ đại diện cho query có nhiều document/passage hoặc token dài. Trong code hiện tại, micro-batch tính theo document, mỗi document mở nhiều passage, và graph của các chunk còn được giữ đến loss của cả query.[^2]

**Thiết kế auto-probe.** Trước một logical run, đọc GPU được cấp và VRAM trống thực tế; khởi đầu với phần dự phòng `max(2 GiB, 15% free)` và áp thêm cap nếu môi trường có cấp. PyTorch `mem_get_info()` cung cấp free/total toàn GPU; allocator cap chỉ giới hạn tiến trình, không đặt chỗ độc quyền trên GPU dùng chung.[^13]

1. Chọn một bộ nhỏ các query nặng nhất trong A_fit theo số passage và token đã encode. Dùng đúng training path, `max_length=768`, FP32 master weights, FP16 autocast, GradScaler và full AdamW của E2; giữ gradient accumulation và gradient checkpointing theo cấu hình đã đăng ký.
2. Thử micro-batch 1, 2, 4, 8, 16 trong subprocess riêng. Mỗi phép thử phải đi qua optimizer step đã khởi tạo optimizer state và thêm bước steady-state; synchronize trước đo peak allocated/reserved. Probe score-batch riêng vì đơn vị batch scoring là passage.
3. Chọn cấu hình nhanh nhất trong ngân sách đo được và phần dự phòng. Nếu batch 1 không fit, chờ tài nguyên; không tự đổi optimizer, quantization, precision, số passage hoặc token limit.
4. Bỏ trạng thái model/optimizer đã cập nhật trong probe; reload checkpoint gốc và reset seed trước fit thật. Đóng băng runtime contract sau probe, lưu GPU UUID, VRAM đo được, peak, phiên bản thư viện và các tham số thực dùng.
5. Khi OOM giữa run vì VRAM dùng chung biến động, kết thúc worker của chính job và giữ các fold đã hoàn tất. Chờ rồi chạy lại fold dở với cùng contract; thay runtime contract phải tạo phiên bản mới. Không trộn score của fold dở với lần retry.

**Chi phí cần tính đúng.** Protocol ở trang trước cần tối đa `4 regime × 5 outer fold × 2 inner seed × 2 neural seed = 80` lượt fit E2, không phải chỉ chạy E3 CPU. Chạy standard trước: 20 lượt; chỉ qua gate mới cấp ngân sách 60 lượt stress. Nếu thiếu trọng số cũ hoặc sai label dependency thì các report top-5 hiện tại không thể thay các lượt fit này.

| Thành phần dùng trong pipeline suy luận cuối | Tham số dự kiến |
|---|---:|
| E0: BQB + reranker cũ + selector nhỏ | Khoảng 1,09B |
| Hai bộ trọng số E2 760M | Khoảng 1,52B |
| Gate E3-B | Không thêm model lớn |
| Tổng | Khoảng 2,61B + selector nhỏ |

Phép cộng tính các bộ trọng số khác nhau dùng khi suy luận, kể cả chạy tuần tự. Không ensemble cả mười model của năm fold và hai seed: riêng cấu hình đó cộng E0 đã khoảng 8,69B. Cần đếm lại số tham số checkpoint thực khi đóng gói.[^1]

Đo thời gian fold đầu để dự báo thời gian còn lại; không suy ra tốc độ từ dung lượng 80GB. Nếu giữ đủ 80 checkpoint FP32 của model 760M, riêng weights xấp xỉ 243GB, chưa tính dữ liệu và trạng thái resume. Kiểm tra dung lượng đĩa trước khi cấp toàn bộ lịch chạy.

<!-- pagebreak -->

## Phạm vi patch và tiêu chí nghiệm thu

**Patch cần xây.** Đây là đặc tả để triển khai, chưa phải một patch đã chạy thử. Handoff hiện có code E2/E3, report và preflight; thiếu actual E1 manifest, memory features, toàn bộ c35 scores và trọng số từng fold. Các artifact trên server chỉ được tái dùng khi đúng hash và đúng phạm vi nhãn.

| Thành phần đề xuất | Nội dung thay đổi |
|---|---|
| `e2_mil_finetune_oof.py` | Tách hàm fit/score; xuất full scores và best weights; giữ loss E2 |
| `e3b_nested_mil_scores.py` | Điều phối T/A/C/V; lưu exact QID/group dependencies |
| `e3b_residual_score_gate.py` | Utility so với incumbent; 8 policy + abstain; top-4 guard |
| `gpu_autoprobe.py` | Đo training step, scoring và VRAM trống; ghi runtime contract |
| `E3B_PROTOCOL.md` | Khóa grid, seed, split, hash, tiêu chí dừng và promotion |

Mỗi score row cần `qid`, `doc_id`, `regime`, `outer_fold`, `inner_seed`, `neural_seed`, document MIL score, passage scores/count và empty-mask. Sidecar chung cần hashes của model weights/tokenizer, E1 manifest/frozen scores, stage1/stage2/anchor/reference/group map; exact fit/early-stop/calibration/validation QID lists; runtime và code version. Hash config model không thay thế được hash trọng số model.

Completed-fold chỉ được ghi atomically sau khi weights, scores và report hoàn chỉnh, kiểm tra đủ query/document rồi hash thành công. Score của C phải mang model hash giống score V tương ứng. Resume kiểm tra mọi dependency; không nhận một progress JSON chỉ có top-5 làm bằng chứng rằng c35 score đã tồn tại.[^2][^3]

**Kiểm chứng CPU trước khi giao GPU.** Các test có giá trị cần tập trung vào top-4/unique/c35; candidate thiếu memory vẫn có thể thắng bằng neural score; ứng viên có cả hai score thấp hơn incumbent không được đổi chỉ vì memory; score tie và evidence rỗng trả V16b; missing logits làm preflight lỗi; các tập nhãn giao nhau bị chặn; và macro Recall xử lý đúng query nhiều nhãn. Dùng dữ liệu giả nhỏ cho invariants, không mở holdout/public để test.

**Gate nghiên cứu giữ nguyên.** Với từng inner seed, standard và cả ba stress regime phải đồng thời có gain Recall > 0 và successful rescues > destructive changes. Báo cáo riêng từng seed/regime, coverage của phép sửa, ca trung tính, số cứu/hại và net query-equivalents. Không lấy trung bình che một regime âm; không lựa seed sau khi thấy điểm.

Bổ sung khoảng tin cậy paired theo group cho gain như thông tin độ bất định, sử dụng đơn vị resample là group tương ứng. Không gọi tám tổ hợp regime/seed là tám bộ test độc lập. Policy có ít ca thay đổi vẫn có thể đạt gate hình thức nhưng chưa đủ cơ sở để hứa chắc về private.

**E4/E5.** Chỉ khi E3-B qua toàn bộ gate mới fit hai model triển khai từ một split A/C complement đã đăng ký trước, chọn gate trên C rồi khóa trọng số và policy. Không chọn final split/seed bằng holdout. E4 đánh giá đúng bộ này một lần; không retrain hoặc đổi threshold sau khi thấy điểm. E5 dùng nguyên bộ đã qua E4 để materialize public. Nếu không qua, tiếp tục giữ V16b.[^1]

<!-- pagebreak -->

## Ưu tiên nghiên cứu và lộ trình theo thời gian còn lại

| Hướng đã xem xét | Cơ sở | Quyết định trong nhánh hiện tại |
|---|---|---|
| Full MIL scores + residual gate | Giải quyết trực tiếp thông tin bị mất và so sánh thiếu incumbent | Ưu tiên duy nhất cho lần thử kế tiếp |
| Pairwise cross-encoder kiểu duoBERT | Có thể học đối chiếu hai tài liệu với query | Dự phòng; thêm training, evidence pairing và protocol mới |
| Attention MIL | Học trọng số passage trong bag | Chưa đổi khi E2 hiện tại đã tăng ở mọi regime |
| Temperature scaling riêng | Có thể cải thiện calibration confidence | Không phục hồi logits bị bỏ; tự nó không đổi thứ tự một model |
| Mở rộng c35 hoặc sửa top-4 | Có thể nâng giới hạn cấu trúc | Là nhánh khác, không hợp contract E3 hiện tại |

Nghiên cứu multi-stage BERT phân biệt pointwise reranking với pairwise document comparison. Nếu E3-B vẫn thất bại dù full scores và nested protocol đúng, một verifier đối chiếu query, incumbent và challenger là giả thuyết tiếp theo hợp lý. Chỉ so một shortlist với incumbent sẽ là thiết kế thích nghi cho LegalIR, không phải cấu hình hay mức gain đã được bài báo chứng minh.[^9]

Temperature scaling là một kỹ thuật calibration có bằng chứng trên classification; với một temperature dương chung, thứ tự score trong một model không đổi. Vì vậy nó không giải quyết riêng được việc E3 bỏ tài liệu ngoài top-5 hoặc bỏ qua incumbent.[^8] Conformal Risk Control có điều kiện về loss và tính trao đổi của mẫu; áp dụng tên kỹ thuật đó không bảo đảm net gain dương hay private pass khi phân phối khác. Mất mát Recall ròng của các phép thay không mặc nhiên là loss đơn điệu cần cho bảo đảm cơ bản.[^12]

**Lịch triển khai theo mốc, không cam kết số giờ chưa đo.**

1. Khóa protocol và patch export/lineage/auto-probe. Kiểm tra CPU các invariants, schema và resume; chưa dùng GPU cho sweep thuật toán.
2. Probe trên A100 shared và hoàn tất fold standard đầu. Dùng thời gian fit + score thực tế để tính tổng 20 standard và 60 stress, cộng thời gian chờ tài nguyên.
3. Hoàn thành standard theo cấu hình đã khóa. Nếu bất kỳ inner seed không qua gate, dừng phiên bản; không tiêu tốn stress hoặc mở E4 để tìm điểm tốt.
4. Nếu standard qua và còn đủ thời gian, chạy ba stress regime. Nếu toàn bộ qua, mới tạo bộ triển khai và làm một lần E4 theo kế hoạch; sau đó mới E5.

Nếu ngân sách thời gian không đủ cho bộ gate đầy đủ, lựa chọn hợp lệ là giữ V16b và lưu nghiên cứu, không tuyên bố một run standard đã đại diện cho private. A100 lớn giúp thực thi được protocol đáng tin hơn; nó không tự sửa candidate capacity, nhãn rò rỉ hoặc policy lệch mục tiêu.

**Mục tiêu điểm số.** 0,95 public vẫn là mục tiêu tham vọng, chưa có dự báo định lượng đáng tin. Dùng +0,005 và +0,010 Recall trên complement như các mốc mô tả mức tiến bộ nếu xuất hiện, không làm lời hứa và không cộng cơ học vào 0,921 public. Dấu hiệu đáng theo đuổi là gain ổn định qua nhóm luật, mẫu câu và near-duplicate, cùng E4 độc lập hơn các vòng complement đã xem. Không có phương pháp nào từ các artifact hiện tại đảm bảo thắng vòng private.

<!-- pagebreak -->

## Nguồn

[^1]: Hồ sơ dự án. `LegalIR_E3_Handoff.zip`: `E3_HANDOFF.md`, `E0_E5_CURRENT_PLAN.md`, các mục E0-E5. Tài liệu bàn giao riêng, không có URL công khai; nguồn của snapshot V16b, ràng buộc c35/top-4 và gate promotion. Thông số 4B được dùng theo contract dự án; PDF mô tả dữ liệu không cung cấp đầy đủ quy chế model.

[^2]: Hồ sơ dự án. Trong `LegalIR_E3_Handoff.zip`: `E1_EVIDENCE_FROZEN_CE_PROTOCOL.md`, `E2_MIL_OOF_PROTOCOL.md`, `scripts/e2_mil_finetune_oof.py` (hằng số E2, `_differentiable_scores`, `_train_fold`, `_memory_probe`, `run`), `scripts/v27_a_common.py` (`load_relevance_model`), `output/e2_mil_oof/e2_mil_oof_report.json` và `e2_mil_oof_stress_report.json`. Nguồn cấu trúc E1/E2, loss, runtime, export và tám kết quả E2.

[^3]: Hồ sơ dự án. Trong `LegalIR_E3_Handoff.zip`: `E3_RESIDUAL_TOP5_PROTOCOL.md`, `scripts/e3_residual_top5_oof.py` (`_propose_slot5`, `_select_inner_policy`, `_run_nested_regime`), `output/e3_residual_top5_oof/e3_residual_top5_preflight.json` và `e3_residual_top5_oof_report.json`. Nguồn policy, provenance contract, nested composition và kết quả E3.

[^4]: Tính lại từ các artifact complement trong `v16b_mmarco_diagnostic_artifacts_20260909.zip`: `anchors/complement_v16b_memory_slot5.json`, `splits/complement_queries_1200.json`, `splits/complement_safe_tail_stage1.audit.json`, `splits/complement_safe_tail_stage2_c35_p3.audit.json`, kết hợp report E2/E3 ở nguồn 2-3. Chỉ dùng 1.197 QID của anchor. Hash anchor/stage1/stage2 khớp preflight E3. Các oracle và thống kê incumbent là phân tích suy ra, không phải metric huấn luyện mới. Kết quả tổng hợp kèm trong `LegalIR_E3B_Audit.json`.

[^5]: UIT Data Science Challenge 2026. `DSC2026_Task1_LegalIR_Data_Overview(2).pdf`, mục 4-5 về macro Recall và giới hạn tối đa năm ID; `scoring(1).py` là scorer được cung cấp. Tài liệu đính kèm, không có URL công khai được xác nhận.

[^6]: Cawley, G. C.; Talbot, N. L. C. (2010). [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html). JMLR 11:2079-2107. Cơ sở về phương sai trong chọn mô hình và sai lệch đánh giá.

[^7]: Geifman, Y.; El-Yaniv, R. (2019). [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html). ICML, PMLR 97:2151-2159. Cơ sở về selective prediction và risk-coverage; không phải thử nghiệm LegalIR.

[^8]: Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML, PMLR 70:1321-1330. Cơ sở cho temperature scaling; nhận xét về thứ tự score là hệ quả toán học của phép chia cho temperature dương chung.

[^9]: Nogueira, R.; Yang, W.; Cho, K.; Lin, J. (2019). [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424). arXiv:1910.14424, phần monoBERT và duoBERT. Cơ sở cho đối chiếu pairwise; chưa xác nhận hiệu quả trên pipeline hiện tại.

[^10]: Ilse, M.; Tomczak, J.; Welling, M. (2018). [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html). ICML, PMLR 80:2127-2136. Cơ sở học nhãn bag và aggregation; E2 hiện dùng logsumexp, không dùng attention của bài báo.

[^11]: Scikit-learn. [StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), mục mô tả và `cv`, truy cập 10/09/2026. Tài liệu phân biệt tạo prediction để fit tầng cuối với đánh giá mô hình; lỗ hổng E3 là suy luận từ code tại nguồn 2-3.

[^12]: Angelopoulos, A. N.; Bates, S.; Fisch, A.; Lei, L.; Schuster, T. [Conformal Risk Control](https://arxiv.org/html/2208.02814v4), arXiv:2208.02814v4, 13/06/2025, mục 1.1, 2.1 và 2.4. Cơ sở về expected risk, tính trao đổi và điều kiện đơn điệu.

[^13]: PyTorch contributors. [CUDA memory implementation](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py), `mem_get_info` và `set_per_process_memory_fraction`, truy cập 10/09/2026. Nguồn API bộ nhớ; các mức dự phòng và chiến lược retry trong báo cáo là đề xuất vận hành cần kiểm chứng trên server.
