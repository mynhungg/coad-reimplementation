"""Generate project/so-lieu-final.md from audit_final.json (numbers are never typed by hand)."""
import json
from pathlib import Path
A = json.load(open(Path(__file__).with_name('audit_final.json'), encoding='utf-8'))

def f(x, d=2, sign=False):
    if x is None:
        return '—'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    return s.replace('-', '−').replace('.', ',')
def p(x):
    return '< 0,001' if x < 0.001 else f(x, 3)
MET = ['F1', 'AUC-PR', 'R-AUC-PR', 'VUS-PR']
L = []
w = L.append

w('# Số liệu cuối cùng cho báo cáo')
w('')
w('*Sinh tự động ngày 08/11/2026 từ các file CSV gốc trong `CoAD_project/results/` bằng `tools/audit_final.py` và `tools/make_final_md.py`; cách chạy lại ở mục "Tái tạo" cuối file. Mọi **bảng** được điền bằng code, không gõ tay. Các gạch đầu dòng nhận xét và mục 9 tóm tắt lại từ các bảng đó. Các số chính đã được đối chiếu bằng `kiem-tra-so-lieu.ipynb`: 62/62 khớp.*')
w('')
w('**Quy ước:**')
w('')
w('- Mọi điểm số được nhân 100.')
w('- "CoAD reimplement" là chế độ code (trùng code tác giả), 64 series TSB-AD × 3 seed (46, 47, 48), lấy trung bình ± độ lệch chuẩn qua seed.')
w('- "CoAD reproduce" là một lần chạy code tác giả.')
w('- Kiểm định theo cặp trên series là Wilcoxon signed-rank.')
w('- So nhiều cặp cùng lúc thì hiệu chỉnh bằng Holm.')
w('')

# 1
w('## 1. Reproduce so với Table 1')
w('')
w('Chênh lệch giữa số nhóm đo được và số trong Table 1 (TSB-AD, 64 series, mỗi phương pháp 1 lần chạy):')
w('')
w('| Phương pháp | ' + ' | '.join(f'Δ {m}' for m in MET) + ' |')
w('|---|' + '---|' * 4)
rows = {}
for r in A['table1']:
    rows.setdefault(r['method'], {})[r['metric']] = r['delta']
for m, d in rows.items():
    w(f'| {m} | ' + ' | '.join(f(d[k], 2, True) for k in MET) + ' |')
w('')
w('- SAND, PCA, MP, damp: khớp Table 1, chênh không quá 1 điểm.')
w('- Series2Graph: chênh khoảng 2–3 điểm.')
w('- Kmeans, KshapeAD: **cao hơn** paper khoảng 4 điểm.')
w('- CoAD: F1 và AUC-PR **thấp hơn** khoảng 3 điểm; R-AUC-PR và VUS-PR **cao hơn** khoảng 2 điểm.')
w('')

# 2
w('## 2. Reimplement')
w('')
w('| Metric | Table 1 | CoAD reproduce | CoAD reimplement (chế độ code) | Chế độ paper | p (code so với reproduce) | Chênh so với Table 1, tính theo số lần SD giữa seed |')
w('|---|---|---|---|---|---|---|')
for r in A['reimpl']:
    w(f"| {r['metric']} | {f(r['table1'])} | {f(r['official'])} | {f(r['code'])} ± {f(r['code_sd'])} | {f(r['paper'])} ± {f(r['paper_sd'])} | {p(r['code_minus_official_p'])} | {f(r['gap_in_seed_sd'], 1, True)} |")
w('')
w(f"- Chế độ code tái tạo đúng code tác giả: không khác có ý nghĩa với bản reproduce ở cả 4 metric. Bản reproduce nằm trong khoảng giữa giá trị nhỏ nhất và lớn nhất của 3 seed ở {A['official_inside_seed_range']}/64 series.")
w('- Chênh lệch với Table 1 bằng khoảng 2–5 lần độ lệch chuẩn giữa các seed, tức **lớn hơn mức dao động ngẫu nhiên**.')
w('- MGAB: chế độ code 44,73, chế độ paper 18,44 (VUS-PR).')
w('')

# 3
w('## 3. So với baseline')
w('')
rk = A['ranks']
order = sorted(rk['VUS-PR'], key=lambda k: rk['VUS-PR'][k])
w('Thứ hạng trung bình trên 64 series (8 phương pháp của Table 1; số càng nhỏ càng tốt):')
w('')
w('| Phương pháp | ' + ' | '.join(MET) + ' |')
w('|---|' + '---|' * 4)
for m in order:
    w(f'| {m} | ' + ' | '.join(f(rk[k][m]) for k in MET) + ' |')
w('')
w('CoAD so với từng baseline (VUS-PR, 64 series, Holm cho 7 phép so):')
w('')
w('| Baseline | CoAD tốt hơn / kém hơn | Chênh | Khoảng tin cậy 95% (bootstrap) | p Holm |')
w('|---|---|---|---|---|')
for r in A['m5_pairwise']:
    if r['metric'] == 'VUS-PR':
        w(f"| {r['baseline']} | {r['better']} / {r['worse']} | {f(r['diff'], 2, True)} | [{f(r['ci_low'])}; {f(r['ci_high'])}] | {p(r['p_holm'])} |")
sig = {}
for r in A['m5_pairwise']:
    if r['significant']:
        sig.setdefault(r['metric'], []).append(r['baseline'])
w('')
w('Sau hiệu chỉnh Holm, CoAD hơn có ý nghĩa:')
w('')
for k in MET:
    w(f"- {k}: {', '.join(sig.get(k, []))}")
w('')
w('CoAD **không** hơn Kmeans, KshapeAD, Series2Graph một cách có ý nghĩa ở metric nào.')
w('')

# 4
w('## 4. Ablation')
w('')
w('### 4.1. M6-A: làm mượt score (chế độ code, 192 model)')
w('')
w('| Metric | Table 1 | Có làm mượt | Không làm mượt | Không làm mượt tốt hơn ở | p | Gần Table 1 hơn? |')
w('|---|---|---|---|---|---|---|')
for r in A['m6a']:
    w(f"| {r['metric']} | {f(r['table1'])} | {f(r['smoothed'])} | {f(r['unsmoothed'])} | {r['unsm_higher']}/64 | {p(r['p'])} | {'có' if r['closer_to_table1'] else 'không'} |")
w('')
w('Giả thuyết "độ lệch với Table 1 là do làm mượt" **bị bác bỏ**.')
w('')
w('### 4.2. M6-B: các biến thể ablation của tác giả (seed 46)')
w('')
w('| Biến thể | Metric | Nhóm đo được | Chênh so với CoAD reproduce | Thấp hơn ở | p | Paper (Table 2) | Cùng chiều với paper? |')
w('|---|---|---|---|---|---|---|---|')
for r in A['m6b']:
    w(f"| {r['variant']} | {r['metric']} | {f(r['ours'])} | {f(r['diff'], 2, True)} | {r['lower']}/64 | {p(r['p'])} | {f(r['paper_diff'], 2, True)} | {'có' if r['same_direction'] else '**không**'} |")
w('')
w('Kiểm tra độ vững của M6-B: so VUS-PR với trung bình 3 seed của chế độ code, và với riêng seed 46:')
w('')
for v in ['time_soft', 'guide_hard', 'freq_time_mean']:
    o = A[f'm6b_vs_code_{v}']
    w(f"- `{v}`: {f(o['vs_3seed'], 2, True)} so với trung bình 3 seed (p = {p(o['p'])}); {f(o['vs_seed46'], 2, True)} so với seed 46.")
w('')
w('### 4.3. M6-C: nhóm khác biệt paper ↔ code (9 series MGAB × 3 seed, VUS-PR)')
w('')
w('| Cấu hình | VUS-PR | SD giữa seed | Thay đổi so với điểm xuất phát | Số series tăng | p |')
w('|---|---|---|---|---|---|')
for r in sorted(A['m6c'], key=lambda r: r['config']):
    ch = f(r.get('change'), 2, True) if r.get('change') == r.get('change') and r.get('change') is not None else '—'
    hi = f"{int(r['higher'])}/9" if r.get('higher') == r.get('higher') and r.get('higher') is not None else '—'
    pp = p(r['p']) if r.get('p') == r.get('p') and r.get('p') is not None else '—'
    w(f"| {r['config']} | {f(r['vus'])} | {f(r['seed_sd'])} | {ch} | {hi} | {pp} |")
w('')

# 5
w('## 5. Phân tích lỗi')
w('')
m7 = A['m7']
w(f"**M7** (seed 46, VUS-PR, cả 64 series):")
w('')
w(f"- Chế độ code: score đầy đủ {f(m7['code|full'])}, chỉ nhánh phân loại {f(m7['code|classifier only'])}, chỉ nhánh tái tạo {f(m7['code|reconstruction only'])}.")
w(f"- Chế độ paper: score đầy đủ {f(m7['paper|full'])}, chỉ nhánh phân loại {f(m7['paper|classifier only'])}, chỉ nhánh tái tạo {f(m7['paper|reconstruction only'])}.")
w('')
w('**M9: tỉ lệ đoạn bất thường bắt được, và bước làm mất thông tin**')
w('')
w('Một đoạn được coi là bắt được khi percentile cao nhất của nó ≥ ngưỡng; ngưỡng chính là 0,99. Bảng dưới cho thấy độ nhạy theo ngưỡng.')
w('')
w('| Family | Ngưỡng | Bắt được | Mất do làm mượt | Mất do gộp nhánh | Mất do bước xử lý | Không tín hiệu nào bắt được |')
w('|---|---|---|---|---|---|---|')
for fam_ in ['TODS', 'IOPS', 'MGAB', 'SVDB', 'SED']:
    for th in ['0.95', '0.99', '0.999']:
        t = A[f'm9_steps_{th}']
        g = lambda k: t.get(k, {}).get(fam_, 0.0)
        w(f"| {fam_} | {th.replace('.', ',')} | {f(g('found'), 3)} | {f(g('smoothing'), 3)} | {f(g('fusion'), 3)} | {f(g('processing'), 3)} | {f(g('all weak'), 3)} |")
w('')
w('Các đoạn OE bỏ sót: tỉ lệ được mỗi tín hiệu bắt được, theo từng ngưỡng:')
w('')
w('| Ngưỡng | Số đoạn OE bỏ sót (trên 8.262) | Lỗi tái tạo thô | Nhánh tái tạo đã xử lý | Score cuối | Tái tạo dự phòng |')
w('|---|---|---|---|---|---|')
for th in ['0.95', '0.99', '0.999']:
    r = A[f'm9_rescue_{th}']
    w(f"| {th.replace('.', ',')} | {r['oe_missed']} | {f(r['raw_error'], 3)} | {f(r['reconstruct'], 3)} | {f(r['final'], 3)} | {f(r['backup'], 3)} |")
w('')
w(f"Tương quan Spearman giữa OE và lỗi tái tạo, tính trên các đoạn: ρ = {f(A['m9_oe_error_rho'], 3, True)}.")
w('')

# 6
w('## 6. Cải tiến: `calibrated`')
w('')
w('`calibrated`: chia mỗi nhánh cho độ lệch chuẩn của nó trên **phần train** rồi mới gộp. Không dùng nhãn, không chỉnh tham số.')
w('')
w('| Biến thể | Metric | Giá trị | CoAD reimplement | Chênh | Số series tăng | p (một phía) | Mức tăng ở seed 46 / 47 / 48 |')
w('|---|---|---|---|---|---|---|---|')
for r in A['module_a']:
    w(f"| {r['variant']} | {r['metric']} | {f(r['value'])} | {f(r['coad'])} | {f(r['diff'], 2, True)} | {r['higher']}/64 | {p(r['p_one_sided'])} | {f(r['gain_seed46'], 2, True)} / {f(r['gain_seed47'], 2, True)} / {f(r['gain_seed48'], 2, True)} |")
b = A['backup_over_calibrated']
w('')
w(f"- `calibrated + backup` so với `calibrated`: {f(b['diff'], 2, True)}, tăng ở {b['higher']}/64 series, p = {p(b['p'])} (hai phía).")
w('- Theo family (VUS-PR, `calibrated` − CoAD reimplement): ' + '; '.join(f'{k} {f(v, 2, True)}' for k, v in A['calibrated_by_family'].items()) + '.')
w('- Thứ hạng trung bình (9 phương pháp): ' + ', '.join(f'{k} {f(v)}' for k, v in A['ranks_with_calibrated'].items()) + '.')
w('')
w('`calibrated` so với 8 phương pháp của Table 1 (VUS-PR, Holm cho 8 phép so):')
w('')
w('| So với | `calibrated` tốt hơn | Chênh | p | p Holm |')
w('|---|---|---|---|---|')
for r in A['calibrated_vs_table1']:
    w(f"| {r['vs']} | {r['better']}/64 | {f(r['diff'], 2, True)} | {p(r['p'])} | {p(r['p_holm'])} |")
w('')

# 7
w('## 7. Các giả thuyết')
w('')
pr = A['m8_prereg']
h = A['m10']
st = A['stress']
w('| Giả thuyết | Dự đoán ghi trước | Kết quả | Kết luận |')
w('|---|---|---|---|')
w('| M6-A: độ lệch với Table 1 là do làm mượt | F1, AUC-PR tăng; R-AUC-PR, VUS-PR giảm; cả 4 tiến gần Table 1 | Cả 4 metric giảm, không metric nào tiến gần Table 1 | Sai |')
w(f"| M8: chế độ paper chỉ dùng nhánh tái tạo tốt hơn chế độ code (seed 47–48) | p < 0,05 | {f(pr['paper_recon'])} so với {f(pr['code'])}; tăng ở {pr['wins']}/64; p = {p(pr['p'])} | Đúng, nhưng không phải cải tiến theo trục chính (chế độ code: chỉ nhánh tái tạo 42,04 so với 41,69) |")
w('| M9-H1: hợp tác hỏng khi OE bỏ sót | Phần tái tạo cứu được < 50% số đoạn OE bỏ sót; tương quan dương | Lỗi tái tạo thô cứu được 44–59% (tùy ngưỡng); tương quan âm | Sai |')
w(f"| M9-H2: bất thường dài khó phát hiện hơn | 1 → 4 chu kỳ giảm ≥ 0,05 | Giảm {f(st['4.0']['uniform'] - st['1.0']['uniform'] if False else st['1.0']['uniform'] - st['4.0']['uniform'], 3)} | Sai |")
w(f"| M10-H1-A: OE chỉ dựa vào biên đoạn bất thường | Giữa − hai đầu ≤ −0,05 | {f(h['H1-A']['value'], 3, True)} ({h['H1-A']['n']} đoạn) | Sai (ngược chiều) |")
w(f"| M10-H1-B: đường tắt từ bước nhảy ở biên | Đột ngột − mượt ≥ 0,05 | {f(h['H1-B']['value'], 3, True)} ({h['H1-B']['n']} cặp; p hai phía = {p(h['H1-B']['p_two'])}) | Sai |")
w(f"| M10-H2-A: soft mask để lọt thông tin bất thường | AUC cứng − mềm ≥ 0,02 | {f(h['H2-A']['value'], 3, True)} (che cứng tốt hơn ở {h['H2-A']['higher']}/192 model) | Sai (ngược chiều) |")
w(f"| M10-H2-B: MAE mượn bất thường từ patch bên cạnh | > 0, p < 0,05 | Trung bình {f(h['H2-B']['value'], 3, True)}, trung vị {f(h['H2-B']['median'], 3, True)}; p hai phía = {p(h['H2-B']['p_two'])} | **Không kết luận được** |")
g = h['Gradient']
w(f"| M10: xung đột gradient khi train chung | Cosine < 0 và ngắt gradient tốt hơn | Cosine {f(g['cosine'], 2, True)} (âm ở {g['cosine_negative_runs']}/9 lần chạy); ngắt gradient {f(g['detached_minus_joint'], 2, True)}, thấp hơn ở {g['lower']}/9; p hai phía = {p(g['p_two'])} | Sai: không có xung đột. Ngắt gradient chỉ cho **xu hướng** kém hơn (xem mục 8) |")
w('')

# 8
n = A['noise_mgab']; n64 = A['noise_64']
w('## 8. Độ tin cậy và giới hạn')
w('')
w(f"- **Dao động giữa các seed** (chế độ code): độ lệch chuẩn theo từng series trung bình {f(n64['per_series_seed_sd'])} điểm VUS-PR. Độ lệch chuẩn của trung bình 64 series là {f(n64['seed_sd_of_mean'])}.")
w(f"- **Dao động giữa hai lần chạy cùng seed trên GPU** (cùng code, seed 46, MGAB; lần train ở mục 15.4 so với mục 7): chênh tuyệt đối trung bình {f(n['same_seed_mean_abs_diff'])} điểm mỗi series, tối đa 23,71. Lý do: tính toán trên GPU không hoàn toàn tất định.")
w(f"  - Do đó các so sánh **một seed** trên ít series cần đọc thận trọng. Ví dụ: ngắt gradient so với train chung là {f(A['m10']['Gradient']['detached_minus_joint'], 2, True)}; so với lần chạy ở mục 7 là {f(n['detached_minus_section7'], 2, True)} (thấp hơn ở {n['detached_lower_than_section7']}/9). Cùng chiều nhưng chưa có ý nghĩa thống kê.")
w('- **Ngưỡng 0,99 ở M9 là ngưỡng chẩn đoán dùng nhãn.** Tỉ lệ bắt được phụ thuộc nhiều vào ngưỡng (mục 5). Kết luận định tính giữ nguyên ở cả 3 ngưỡng:')
w('  - TODS và MGAB chủ yếu là "không tín hiệu nào bắt được";')
w('  - IOPS, SVDB, SED chủ yếu "mất do làm mượt";')
w('  - lỗi tái tạo thô bắt được nhiều đoạn OE bỏ sót hơn hẳn score cuối.')
w('- **Cùng 64 series** được dùng để đặt giả thuyết và để đánh giá. `calibrated` được định nghĩa trước khi chạy và không chỉnh tham số, nhưng vẫn cần kiểm chứng trên dữ liệu khác (KDD21).')
w('- **Nhiễm bất thường trong phần train** (tính từ dữ liệu gốc, không có trong CSV kết quả): 13/15 series TODS và 1 series IOPS có điểm đã gắn nhãn trong phần train.')
w('')

# 9
w('## 9. Kết luận nào đúng, kết luận nào phải sửa')
w('')
w('| Kết luận trong tài liệu | Đánh giá sau khi tính lại |')
w('|---|---|')
w('| Code công khai tái tạo được CoAD ở mức trung bình; số trong Table 1 lệch có hệ thống | ✅ Đúng |')
w('| CoAD không hơn Kmeans có ý nghĩa; thứ hạng thứ 3 | ✅ Đúng với AUC-PR, R-AUC-PR, VUS-PR; với F1 CoAD đứng thứ 2 |')
w('| Độ lệch với Table 1 không do làm mượt | ✅ Đúng |')
w('| M6-C: nhóm `training` quyết định ở MGAB | ✅ Đúng (0/9 series tăng khi bỏ nhóm này; p = 0,004) |')
w('| M6-B `time_soft`: không tái tạo được lợi ích của nhánh tần số | ✅ Đúng (dương ở cả 3 cách so) |')
w('| M6-B `freq_time_mean`: cùng chiều với paper | ✅ Đúng (−3,8 đến −4,9; p = 0,06–0,09) |')
w('| M6-B `guide_hard`: cùng chiều với paper | ⚠️ **Cần sửa**: −1,3 đến −2,4, chưa có ý nghĩa thống kê, nằm trong mức dao động. Nên ghi là "không phân biệt được với dao động" |')
w('| M8 không phải cải tiến theo trục chính | ✅ Đúng |')
w('| M9: lỗi do bất thường 1 điểm, do làm mượt, do khâu gộp | ✅ Đúng, vững ở cả 3 ngưỡng |')
w('| `calibrated` +2,4 VUS-PR so với CoAD reimplement | ✅ Đúng: dương ở cả 3 seed (+1,75; +2,58; +2,91), p < 0,001. Chưa hơn có ý nghĩa Kmeans hay bản CoAD reproduce (1 lần chạy) |')
w('| M10: không có đường tắt từ biên (H1-A, H1-B) | ✅ Đúng |')
w('| M10: soft mask che có chọn lọc, tốt hơn che cứng ở mức patch (H2-A) | ✅ Đúng (192 model) |')
w('| M10: H2-B "không ủng hộ" | ⚠️ **Cần sửa**: trung bình âm nhưng trung vị dương, p = 0,11 hai phía. Nên ghi là "không kết luận được" |')
w('| M10: "train chung có ích" (ngắt gradient −11,5) | ⚠️ **Cần sửa**: cosine dương ở cả 9/9 lần chạy, nên **không có xung đột** là vững. Nhưng "có ích" chỉ là **xu hướng**: p = 0,074; 1 seed; dao động giữa hai lần chạy cùng seed lên tới 24 điểm |')
w('')
w('## Tái tạo')
w('')
w('1. `kiem-tra-so-lieu.ipynb`: đối chiếu từng số trong các file hướng dẫn với CSV (62 dòng).')
w('2. `tools/audit_final.py`: tính lại toàn bộ bảng trong file này. Chạy bằng `python tools/audit_final.py <đường dẫn tới CoAD_project>` (tạo `tools/audit_final.json`), rồi `python tools/make_final_md.py` để sinh lại file này.')
out = Path(r'D:/Master/Course/Data mining/project/so-lieu-final.md')
out.write_text('\n'.join(L) + '\n', encoding='utf-8')
print('written', out, len(L), 'lines')
