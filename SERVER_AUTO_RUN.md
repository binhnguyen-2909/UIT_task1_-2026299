# Chạy LegalIR E0–E2 trên GPU server

Sau khi giải nén ZIP, bạn chỉ cần chạy ba lệnh này trong GPUHub/Ubuntu có GPU:

```bash
cd /root/autodl-tmp
unzip -q LegalIR_E0_E2_Server_Auto.zip
cd LegalIR_E0_E2_RTX5070_Full
screen -S legalir-e0e2
bash ./run_e0_e2_auto.sh 2>&1 | tee /root/autodl-tmp/legalir-e0-e2.log
```

Script tự cài dependency ngoài PyTorch/CUDA, tự đọc VRAM, chọn batch an toàn,
chạy E0 V16b, kiểm tra E1 đã niêm phong, rồi chạy **memory probe exact AdamW**
trước E2. Nếu probe không fit VRAM thì script dừng trước E2 dài hạn, không âm
thầm đổi optimizer/quantization và không làm sai protocol. GPU từ 40 GiB dùng
profile đã thử nghiệm; GPU nhỏ hơn có profile thận trọng và phải pass probe.

E2 chỉ chạy ba stress regime còn thiếu; nó không chạy lại standard OOF đã hoàn
thành. Các fold đã xong được lưu checkpoint nên chạy lại cùng ZIP/config sẽ
resume.

Nếu thực sự muốn dựng lại E1 từ đầu (không cần cho E2 vì artifact đã có), dùng:

```bash
REBUILD_E1=true bash ./run_e0_e2_auto.sh 2>&1 | tee /root/autodl-tmp/legalir-e0-e2.log
```

Tách screen bằng `Ctrl+A` rồi `D`; xem log bằng:

```bash
tail -f /root/autodl-tmp/legalir-e0-e2.log
```
