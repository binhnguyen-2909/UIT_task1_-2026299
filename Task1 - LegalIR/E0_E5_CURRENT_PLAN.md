# E0–E5 — kế hoạch tiếp tục hiện hành

Mục tiêu của kế hoạch này là tiếp tục từ baseline V16b đã khóa, không thay đổi
submission hiện hành trước khi từng gate độc lập được thông qua.

## E0 — V16b baseline đã khóa

- Artifact đủ điều kiện: `output/v16b_memory_slot5.zip`.
- Public snapshot: `0.9210`; locked holdout Recall@5: `0.9228333`.
- Pipeline: BQB 330M + Vietnamese reranker 760M + tiny selector, dưới 4B.
- E0 luôn là anchor, không được ghi đè bởi experiment E1–E4.

## E1 — evidence pack + frozen CE, đã hoàn tất

- Candidate universe giữ nguyên c35 của V16b.
- Evidence gồm 3–4 block pháp lý label-blind/document; proposer lexical,
  BQB dense, structure/citation; CE 760M chỉ inference frozen.
- E1 frozen CE không được promote trực tiếp: Recall@5 complement `0.8747842`
  thấp hơn V16b anchor `0.8860624`, dù candidate oracle vẫn `0.9503899`.
- Manifest và frozen score được giữ làm input đã khóa cho E2.

## E2 — MIL relevance/boundary OOF

- Standard grouped OOF đã hoàn tất với hai seed: Recall trung bình `0.8938945`,
  delta `+0.0078321` so với V16b complement anchor.
- Ba stress regime đã hoàn tất, không chạy lại standard. Sáu seed/regime đều
  dương: `law_source_family` `+0.0052910`/`+0.0098858`,
  `query_template_family` `+0.0116263`/`+0.0141326`, và
  `near_duplicate_family` `+0.0112086`/`+0.0169869`.
- Mỗi fold checkpoint atomically; resume chỉ được phép khi toàn bộ provenance,
  script, split, seed và hyperparameter trùng khớp.
- RTX 5070 (12GB) phải pass memory probe trước; exact full-AdamW OOF có thể
  không fit. Không thay optimizer/quantization để gọi đó là E2 cùng protocol.

## E3 — residual top-5 policy, chỉ mở sau E2 stress pass

- Dùng V16b memory output và E2 relevance như hai signal đã cross-fit.
- Có nested inner/outer separation riêng; E3 không dùng holdout/public để chọn
  ngưỡng hay policy.
- Default abstain là V16b. Mọi edit phải giới hạn c35, giữ top-4 và chứng minh
  gain cùng safety trên standard lẫn ba stress regime.
- Implementation và nested OOF đã hoàn tất. Gate chưa pass: minimum gain theo
  hai E3 inner seeds lần lượt là standard `+0.0004177`, law/source
  `-0.0020886`, query/template `-0.0022278`, near-duplicate `-0.0004177`.
  Vì vậy E4 vẫn khóa; không có holdout hay public label nào đã được mở.
- Nhánh tiếp theo là **E3-B**: refit hai E2 MIL seed theo A-fit/A-stop/C/V
  group-disjoint trong từng outer fold, export đủ score c35 và chỉ so ứng viên
  slot-5 với incumbent V16b. Policy grid cố định gồm 8 policy + abstain;
  standard cần hoàn tất/pass trước khi cấp GPU cho stress.
- **E3-B đã hoàn tất/pass**: standard và 30 stress fold đều qua gate;
  `promotion_to_e4_allowed:true`. Fixed E4 policy là `tau=0.50`,
  `memory_weight=0.0`.
- **E3-C là nhánh nghiên cứu chạy song song**: LoRA duo comparator đọc query,
  challenger evidence và incumbent slot-5 evidence; báo cáo OOF chỉ trên
  complement. E3-C bắt buộc khởi tạo từ checkpoint E3-B cùng fold đã retain,
  không thể mở holdout, không thay E3-B policy và không cấp quyền E4/E5.

## E4 — locked-holdout parity gate, đã được mở quyền sau E3-B pass

- E3-B standard và cả ba stress regime đã pass; E4 chỉ dùng policy đã khóa
  `tau=0.50`, `memory_weight=0.0`, giữ V16b top-4 và c35.
- `MODE=prepare` tạo holdout evidence label-blind, refit đúng hai E2 deployment
  model từ toàn complement, và materialize prediction; nó không mở qrels.
- `MODE=parity` là lần đọc locked holdout duy nhất, được ledger chặn chạy lại.
  Không retrain, không tune threshold/passage/seed sau khi đọc kết quả.
- Promote E5 chỉ khi gain Recall@5 so với V16b > 0, rescues > destructive,
  provenance/leakage/format đều hợp lệ và pipeline vẫn dưới 4B.

## E5 — public materialization/submission, chỉ mở sau E4 pass

- Materialize public từ policy đã khóa ở E3/E4, kiểm tra 1,000 QID, đúng 5 ID,
  không duplicate và budget <4B.
- Đóng ZIP/hash/release manifest mới; V16b giữ fallback đủ điều kiện.

## Không được làm

- Không dùng public labels, không mở holdout ở E1/E2/E3.
- Không đổi candidate pool khỏi c35 trong nhánh này.
- Không dùng standard OOF lặp lại thay cho stress validation.
