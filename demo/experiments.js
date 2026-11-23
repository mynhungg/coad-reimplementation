/* Aggregate experiment results shown in the "Experiment results" screen, so every number on slides 22-25 and in the
   appendix can be opened during Q&A. Numbers come from docs/report.md (section given on each card). Bilingual: the
   text is picked from the current language, so the i18n layer has nothing to translate. Exported as window.EXP. */
(function () {
  'use strict';
  const en = () => !!(window.I18N && window.I18N.lang && window.I18N.lang() === 'en');
  const L = (vi, e) => (en() ? e : vi);

  // each card: id, slide, title [vi, en], claim [vi, en], tables: [{head:[[vi,en]...], rows:[[...]], hl: row index to bold}], note [vi, en], src
  const VIEWS = [
    { id: 'repro', label: ['Chạy lại Table 1', 'Table 1 rerun'], slides: '22 · A1 · A3', cards: [
      { slide: '22', title: ['So với Table 1 trên TSB-AD (64 series)', 'Against Table 1 on TSB-AD (64 series)'],
        claim: ['Bản bổ sung chi tiết từ code khớp code tác giả; chính code tác giả lệch paper ở F1 và AUC-PR.', 'The build with code details matches the authors\' code; the authors\' code itself is off the paper on F1 and AUC-PR.'],
        tables: [{ head: [['Metric', 'Metric'], ['Paper (Table 1)', 'Paper (Table 1)'], ['Bản chỉ theo paper (3 seed)', 'Paper-only build (3 seeds)'], ['Bản bổ sung chi tiết từ code (3 seed)', 'Build with code details (3 seeds)'], ['Code tác giả (1 lần)', 'Authors\' code (1 run)'], ['Code − paper', 'Code − paper'], ['p (bản bổ sung vs code)', 'p (code-details build vs code)']],
          rows: [['F1', '49.13', '45.99 ± 0.46', '45.62 ± 0.69', '45.99', '−3.14', '0.30'],
                 ['AUC-PR', '43.66', '39.81 ± 0.62', '39.87 ± 1.24', '40.43', '−3.23', '0.30'],
                 ['R-AUC-PR', '40.50', '39.63 ± 1.06', '42.80 ± 0.91', '42.73', '+2.23', '0.75'],
                 ['VUS-PR', '39.83', '38.72 ± 1.00', '41.69 ± 0.96', '41.70', '+1.87', '0.56']], hl: 3 }],
        note: ['p là Wilcoxon ghép cặp trên 64 series: không phát hiện khác biệt giữa bản bổ sung chi tiết và code tác giả.', 'p is a paired Wilcoxon test over 64 series: no difference detected between the code-details build and the authors\' code.'], src: '§4.1, §5.1' },
      { slide: '22', title: ['CoAD trên KDD21 (250 series, code tác giả)', 'CoAD on KDD21 (250 series, authors\' code)'],
        claim: ['Trên KDD21 cùng code đó khớp paper, chênh dưới 0.3 điểm: độ lệch chỉ nằm ở phần TSB-AD.', 'On KDD21 the same code matches the paper within 0.3 points: the gap is only in the TSB-AD part.'],
        tables: [{ head: [['Metric', 'Metric'], ['Table 1', 'Table 1'], ['Code tác giả', 'Authors\' code'], ['Chênh', 'Difference']],
          rows: [['F1', '52.82', '52.53', '−0.29'], ['AUC-PR', '48.10', '47.82', '−0.28'], ['R-AUC-PR', '46.35', '46.37', '+0.02'], ['VUS-PR', '45.67', '45.75', '+0.08']], hl: 3 }],
        note: ['Dữ liệu KDD21 dựng lại từ UCR Anomaly Archive 2021, khớp 250/250 series với all_length.csv của tác giả.', 'KDD21 data rebuilt from the UCR Anomaly Archive 2021; 250/250 series match the authors\' all_length.csv.'], src: '§6.5' },
      { slide: '22 · A1', title: ['Theo family (VUS-PR, trung bình 3 seed)', 'By family (VUS-PR, mean of 3 seeds)'],
        claim: ['Trung bình gần paper một phần do bù trừ: MGAB thấp hơn code 23 điểm, SVDB và TODS lại cao hơn.', 'The mean is close to the paper partly through offsetting: MGAB is 23 points below the code, SVDB and TODS above.'],
        tables: [{ head: [['Family', 'Family'], ['Bản chỉ theo paper', 'Paper-only build'], ['Bản bổ sung chi tiết từ code', 'Build with code details'], ['Code tác giả', 'Authors\' code']],
          rows: [['MGAB (9)', '18.44', '44.73', '41.71'], ['SED (3)', '88.30', '91.97', '93.19'], ['SVDB (20)', '58.97', '54.74', '56.09'], ['IOPS (17)', '28.62', '31.48', '32.50'], ['TODS (15)', '25.44', '23.99', '22.62']], hl: 0 }],
        note: ['', ''], src: '§4.3' },
      { slide: 'A3', title: ['Chạy lại các baseline của Table 1', 'Rerunning the Table 1 baselines'],
        claim: ['8/14 baseline deep và 5/7 baseline data mining khớp paper. Baseline deep khớp khi train 1 epoch như script của tác giả.', '8/14 deep and 5/7 data-mining baselines match the paper. Deep baselines match when trained 1 epoch as in the authors\' script.'],
        tables: [{ head: [['Phương pháp', 'Method'], ['TSB-AD: nhóm', 'TSB-AD: ours'], ['Table 1', 'Table 1'], ['KDD21: nhóm', 'KDD21: ours'], ['Table 1', 'Table 1'], ['Epoch', 'Epochs']],
          rows: [['DADA', '14.03', '14.53', '2.08', '2.05', 'zero-shot'], ['MOMENT', '24.55', '24.67', '9.40', '9.13', 'zero-shot'], ['TFMAE', '6.43', '5.77', '1.86', '1.75', '0'],
                 ['COUTA', '10.62', '11.01', '2.62', '3.82', '1'], ['M2N2', '9.12', '9.25', '2.77', '3.18', '1'], ['TFAD', '5.88', '5.99', '1.56', '1.58', '1'],
                 ['TranAD', '12.28', '12.05', '2.74', '7.89', '1'], ['DCdetector', '5.34', '5.69', '1.90', '1.85', '10'], ['FITS', '11.49', '10.37', '2.76', '3.45', '50'],
                 ['AnomalyTrans', '13.10', '6.13', '8.56', '1.80', '10'], ['CATCH', '14.14', '21.67', '2.38', '9.19', '1'], ['FCVAE', '13.97', '20.58', '2.62', '6.38', '1'],
                 ['MAUT', '10.34', '14.75', '2.92', '23.65', '1'], ['CutAddPaste', '11.52', '25.08', '4.14', '18.20', '1']] },
          { head: [['Data mining (VUS-PR)', 'Data mining (VUS-PR)'], ['TSB-AD: nhóm', 'TSB-AD: ours'], ['Table 1', 'Table 1'], ['KDD21: nhóm', 'KDD21: ours'], ['Table 1', 'Table 1']],
          rows: [['SAND', '31.06', '31.05', '35.99', '33.78'], ['Sub-PCA', '23.92', '23.85', '11.70', '13.17'], ['DAMP', '12.99', '12.60', '19.92', '24.18'], ['MP', '29.48', '28.94', '39.20', '24.13'],
                 ['Series2Graph', '31.44', '29.57', '23.12', '24.81'], ['KMeansAD', '41.01', '36.81', '33.97', '33.20'], ['KShapeAD', '35.86', '31.27', '40.10', '39.02']] }],
        note: ['Việc baseline deep khớp khi train 1 epoch là suy luận từ việc số trùng, chưa phải bằng chứng trực tiếp. CoAD vẫn đứng đầu, nhưng trên TSB-AD chỉ hơn KMeansAD 0.7 thay vì 3.0.', 'That deep baselines match at 1 epoch is an inference from the agreement, not direct evidence. CoAD stays first, but on TSB-AD leads KMeansAD by 0.7 instead of 3.0.'], src: '§6.5' }] },

    { id: 'checks', label: ['Phân tích thêm', 'Further analysis'], slides: '23 · A2 · A4', cards: [
      { slide: '23 · A2', title: ['Nhóm chi tiết từ code (MGAB, 9 series × 3 seed)', 'Code detail groups (MGAB, 9 series × 3 seeds)'],
        claim: ['Bỏ nhóm training thì cả 9 series đều giảm; nhóm architecture cũng cần.', 'Without the training details all 9 series drop; the architecture details are needed too.'],
        tables: [{ head: [['Cấu hình', 'Configuration'], ['VUS-PR', 'VUS-PR'], ['Thay đổi', 'Change'], ['p', 'p']],
          rows: [[L('Bản bổ sung chi tiết từ code', 'Build with code details'), '44.73', '—', '—'],
                 [L('− training (batch sạch + méo, cân bằng loss, trọng số lớp, lịch LR, fp16)', '− training (clean + distorted batches, loss balancing, class weight, LR schedule, fp16)'), '19.00', '−25.73 (0/9)', '0.004'],
                 [L('− architecture (GRU, dropout, MLP head, hidden size, SELU)', '− architecture (GRU, dropout, MLP head, hidden size, SELU)'), '30.68', '−14.04 (0/9)', '0.004'],
                 [L('− bất thường giả', '− synthetic-anomaly details'), '38.76', '−5.96', '0.164'],
                 [L('− scoring (suy luận 2 lượt, score tương đối)', '− scoring (two-pass inference, relative score)'), '43.59', '−1.13', '0.074'],
                 [L('Bản chỉ theo paper', 'Paper-only build'), '18.44', '—', '—'],
                 [L('Bản chỉ theo paper + training', 'Paper-only + training details'), '29.64', '+11.20 (8/9)', '0.012']], hl: 1 }],
        note: ['Thí nghiệm bỏ theo nhóm, chỉ trên MGAB: chưa tách được tác động của từng chi tiết.', 'Groups were removed together, on MGAB only: single details are not separated.'], src: '§5.3' },
      { slide: '23', title: ['Chỉ dùng score tái tạo (ghi dự đoán trước)', 'Reconstruction score only (pre-registered)'],
        claim: ['Dự đoán trước khi chạy: bản chỉ theo paper, dùng riêng score tái tạo, sẽ hơn bản bổ sung chi tiết từ code. Dự đoán đúng.', 'Predicted before running: the paper-only build scored on reconstruction only beats the build with code details. The prediction held.'],
        tables: [{ head: [['Cấu hình (seed 47–48, 64 series)', 'Configuration (seeds 47–48, 64 series)'], ['VUS-PR', 'VUS-PR']],
          rows: [[L('Bản chỉ theo paper, chỉ score tái tạo', 'Paper-only build, reconstruction only'), '44.76'], [L('Bản bổ sung chi tiết từ code, score đầy đủ', 'Build with code details, full score'), '42.23'],
                 [L('Số series tốt hơn / p', 'Series improved / p'), '41/64 · p = 0.019']], hl: 0 },
          { head: [['Seed 46', 'Seed 46'], ['Score đầy đủ', 'Full score'], ['Chỉ phân loại', 'Classifier only'], ['Chỉ tái tạo', 'Reconstruction only']],
          rows: [[L('Bản chỉ theo paper · 64 series', 'Paper-only · 64 series'), '37.60', '33.98', '47.61'], [L('Bản bổ sung chi tiết · 64 series', 'Code details · 64 series'), '40.61', '36.99', '40.88'],
                 [L('Bản chỉ theo paper · MGAB', 'Paper-only · MGAB'), '13.1', '12.2', '43.4'], [L('Bản bổ sung chi tiết · MGAB', 'Code details · MGAB'), '41.3', '41.2', '22.0']] }],
        note: ['Ở bản chỉ theo paper, bộ phân loại yếu kéo score gộp xuống, vì hai score được cộng mà không đưa về cùng thang.', 'In the paper-only build the weak classifier pulls the fused score down, because the two scores are added without a common scale.'], src: '§5.2' },
      { slide: '23', title: ['Bỏ bước làm mượt score (64 series × 3 seed, 192 model)', 'Removing score smoothing (64 series × 3 seeds, 192 models)'],
        claim: ['Bỏ làm mượt thì cả bốn metric đều giảm: làm mượt không giải thích được độ lệch với paper.', 'Without smoothing all four metrics drop: smoothing does not explain the gap to the paper.'],
        tables: [{ head: [['Metric', 'Metric'], ['Table 1', 'Table 1'], ['Có làm mượt', 'With smoothing'], ['Không làm mượt', 'Without'], ['p', 'p']],
          rows: [['F1', '49.13', '45.62', '43.80', '0.103'], ['AUC-PR', '43.66', '39.87', '38.31', '0.101'], ['R-AUC-PR', '40.50', '42.80', '37.73', '0.004'], ['VUS-PR', '39.83', '41.69', '36.85', '0.004']], hl: 3 }],
        note: ['Dự đoán ghi trước: bỏ làm mượt sẽ kéo cả bốn metric về gần Table 1. Kết quả bác bỏ dự đoán.', 'Pre-registered prediction: removing smoothing moves all four metrics towards Table 1. Rejected.'], src: '§6.1' },
      { slide: '23', title: ['Ablation của Table 2 (code tác giả, seed 46, 64 series)', 'Table 2 ablations (authors\' code, seed 46, 64 series)'],
        claim: ['Dùng mean thay max để gộp nhánh thời gian và tần số: AUC-PR giảm 4.27 (p = 0.006), ủng hộ lựa chọn max.', 'Mean instead of max to fuse time and frequency: AUC-PR drops 4.27 (p = 0.006), supporting max.'],
        tables: [{ head: [['Biến thể', 'Variant'], ['VUS-PR', 'VUS-PR'], ['So với bản đầy đủ', 'vs full model'], ['Paper', 'Paper'], ['Cùng hướng paper?', 'Same direction as paper?']],
          rows: [[L('Mean thay max (gộp thời gian/tần số)', 'Mean instead of max (time/frequency)'), '36.84', '−4.86 (p 0.061)', '−6.65', L('Có; AUC-PR −4.27, p 0.006', 'Yes; AUC-PR −4.27, p 0.006')],
                 [L('Mask cứng thay mask mềm', 'Hard mask instead of soft'), '39.30', '−2.40 (p 0.319)', '−5.07', L('Có, nhưng trong mức nhiễu seed', 'Yes, within seed noise')],
                 [L('Bỏ nhánh tần số', 'No frequency branch'), '43.60', '+1.91 (p 0.077)', '−2.91', L('Không, ngược hướng', 'No, opposite')]], hl: 0 }],
        note: ['', ''], src: '§6.2' },
      { slide: '23', title: ['Dữ liệu mới: 60 series của 12 family paper không dùng', 'New data: 60 series from 12 families the paper did not use'],
        claim: ['Nhân đôi trọng số nhánh tái tạo hơn bản chỉ cân bằng ổn định: +0.81, p = 0.0006, dương ở 9/12 family.', 'Doubling the reconstruction weight beats balance alone consistently: +0.81, p = 0.0006, positive in 9/12 families.'],
        tables: [{ head: [['So sánh', 'Comparison'], ['Tập xác nhận (34)', 'Confirmation (34)'], ['Dữ liệu mới (60)', 'New data (60)']],
          rows: [[L('Balance + 2× tái tạo − Balance', 'Balance + 2× reconstruction − Balance'), '+0.80 (Holm 0.004)', '+0.81 (p 0.0006, 9/12)'],
                 [L('Balance + 2× tái tạo − CoAD', 'Balance + 2× reconstruction − CoAD'), '+4.22 (p 0.001)', '+2.25 (p 0.17)'],
                 [L('Balance − CoAD', 'Balance − CoAD'), '+3.42 (Holm 0.078)', '+1.44 (p 0.44)']], hl: 0 }],
        note: ['12 family: YAHOO, UCR, WSD, SMD, Exathlon, OPPORTUNITY, NAB, Stock, SMAP, LTDB, MSL, NEK; 5 series mỗi family, default_rng(2027), bỏ file trùng.', '12 families: YAHOO, UCR, WSD, SMD, Exathlon, OPPORTUNITY, NAB, Stock, SMAP, LTDB, MSL, NEK; 5 series each, default_rng(2027), duplicate files removed.'], src: '§9' },
      { slide: 'A4', title: ['Early stopping (MGAB, 9 series × 3 seed)', 'Early stopping (MGAB, 9 series × 3 seeds)'],
        claim: ['Early stopping làm giảm kết quả, nên không giải thích được việc Table 1 cao hơn code ở F1 và AUC-PR.', 'Early stopping lowers the scores, so it does not explain why Table 1 is above the code on F1 and AUC-PR.'],
        tables: [{ head: [['Biến thể', 'Variant'], ['Train trên', 'Trained on'], ['Early stopping', 'Early stopping'], ['VUS-PR', 'VUS-PR'], ['F1', 'F1'], ['AUC-PR', 'AUC-PR']],
          rows: [[L('Tham chiếu', 'Reference'), '100%', L('Không', 'No'), '44.73', '—', '—'], ['cut80', '80%', L('Không', 'No'), '44.98', '48.13', '37.71'], ['es', '80%', L('Có (patience 10)', 'Yes (patience 10)'), '36.40', '44.60', '33.44']], hl: 2 }],
        note: ['es − cut80 = −8.57; p = 0.025 trên 27 cặp chạy, p = 0.055 trên trung bình 9 series (bằng chứng yếu hơn).', 'es − cut80 = −8.57; p = 0.025 over 27 run pairs, p = 0.055 over the 9 series means (weaker evidence).'], src: '§6.4' }] },

    { id: 'errors', label: ['Phân tích lỗi', 'Error analysis'], slides: '24 · 25 · A5', cards: [
      { slide: '24 · 25', title: ['Đoạn bất thường bị mất ở bước nào (8.262 đoạn, 192 model)', 'Where events are lost (8,262 events, 192 models)'],
        claim: ['Với 5.485 đoạn bộ phân loại bỏ sót: sai số tái tạo thô còn bắt được 53%, score cuối chỉ còn 7%.', 'Of the 5,485 events the classifier misses, the raw reconstruction error still detects 53%, the final score only 7%.'],
        tables: [{ head: [['Family', 'Family'], ['Phát hiện', 'Detected'], ['Mất ở làm mượt', 'Lost at smoothing'], ['Mất ở gộp', 'Lost at fusion'], ['Mất ở xử lý', 'Lost at processing'], ['Không nhánh nào thấy', 'No signal detects it']],
          rows: [['TODS', '0.021', '0.312', '0.006', '0.139', '0.522'], ['IOPS', '0.421', '0.452', '0.009', '0.050', '0.067'], ['MGAB', '0.713', '0.073', '0.011', '0.038', '0.165'], ['SVDB', '0.781', '0.158', '0.011', '0.043', '0.007'], ['SED', '0.818', '0.174', '0.000', '0.008', '0.000']] }],
        note: ['Shapley không phụ thuộc thứ tự bước: làm mượt chiếm 54–100% phần mất (IOPS, SVDB, SED), 70–98% ở TODS và MGAB; gộp chiếm 15–46% ở IOPS và SVDB.', 'Order-free Shapley shares: smoothing accounts for 54–100% of the loss (IOPS, SVDB, SED), 70–98% in TODS and MGAB; fusion 15–46% in IOPS and SVDB.'], src: '§7.1' },
      { slide: '25 · A5', title: ['Gai ngắn chèn vào chuỗi thật (gai 1 điểm, 3σ / 1σ)', 'Short spikes inserted into real series (1-point spike, 3σ / 1σ)'],
        claim: ['Sai số tái tạo bắt gần hết gai 1 điểm, nhưng score cuối mất phần lớn: nguyên nhân chính là trung bình trượt dài một chu kỳ.', 'The reconstruction error catches almost all 1-point spikes, the final score loses most: the main cause is the one-period moving average.'],
        tables: [{ head: [['Bộ phân loại', 'Classifier'], ['Sai số tái tạo thô', 'Raw reconstruction error'], ['Score chưa làm mượt', 'Unsmoothed score'], ['Score cuối', 'Final score']],
          rows: [['0.34 / 0.25', '0.95 / 0.73', '0.88 / 0.68', '0.71 / 0.37']] }],
        note: ['Gai 1σ đứng riêng: tỉ lệ phát hiện từ 0.66 xuống 0.03, do pha loãng (−0.36) và lan sang điểm bình thường (−0.29). Vị trí trong patch gần như không ảnh hưởng (±0.03).', 'Isolated 1σ spikes: detection falls from 0.66 to 0.03, by dilution (−0.36) and spill-over (−0.29). Position within the patch barely matters (±0.03).'], src: '§7.2' },
      { slide: '24', title: ['Năm giả thuyết về cơ chế hợp tác', 'Five hypotheses about the cooperation mechanism'],
        claim: ['Không thấy dấu hiệu lỗi ở cơ chế hợp tác; một phép thử chưa kết luận được.', 'No sign of a fault in the cooperation mechanism; one test is inconclusive.'],
        tables: [{ head: [['Giả thuyết', 'Hypothesis'], ['Kết quả', 'Result']],
          rows: [[L('Bộ phân loại dựa vào biên đoạn', 'Classifier relies on segment boundaries'), L('+0.065: bác bỏ (ngược dấu)', '+0.065: rejected (opposite sign)')],
                 [L('Học mẹo từ bước nhảy ở biên', 'Shortcut from the jump at the boundary'), L('+0.002 (480 cặp, p 0.13): bác bỏ', '+0.002 (480 pairs, p 0.13): rejected')],
                 [L('Soft mask làm lộ thông tin bất thường', 'Soft mask leaks anomaly information'), L('Mask cứng giảm AUC 0.097: mask mềm chọn đúng chỗ', 'Hard mask lowers AUC by 0.097: soft mask is selective')],
                 [L('MAE mượn thông tin từ patch lân cận', 'MAE borrows from neighbouring patches'), L('Trung bình −0.015, trung vị +0.049 (177 model, p 0.11): chưa kết luận', 'Mean −0.015, median +0.049 (177 models, p 0.11): inconclusive')],
                 [L('Hai mục tiêu train kéo ngược nhau', 'Gradient conflict between objectives'), L('Cosine +0.19, dương ở 9/9 lần chạy: không xung đột', 'Cosine +0.19, positive in 9/9 runs: no conflict')]] }],
        note: ['Hai nhánh gần như không tương quan theo đoạn (ρ = −0.089), tức bổ sung cho nhau như thiết kế.', 'The two branches are nearly uncorrelated over events (ρ = −0.089): complementary, as designed.'], src: '§7.3' }] },

    { id: 'base', label: ['So sánh baseline', 'Baseline comparison'], slides: '24 · 26', cards: [
      { slide: '24', title: ['64 series của paper (VUS-PR, Holm trên 7 phép so)', 'The paper\'s 64 series (VUS-PR, Holm over 7 comparisons)'],
        claim: ['CoAD có trung bình cao nhất nhưng không hơn KMeansAD một cách có ý nghĩa; xếp hạng theo series chỉ đứng thứ 3.', 'CoAD has the highest mean but is not significantly better than KMeansAD; on per-series ranks it is only 3rd.'],
        tables: [{ head: [['Baseline', 'Baseline'], ['Hạng TB', 'Mean rank'], ['CoAD thắng / thua', 'CoAD wins / losses'], ['Chênh', 'Diff.'], ['KTC 95%', '95% CI'], ['Holm p', 'Holm p']],
          rows: [['KMeansAD', '2.61', '25 / 39', '+0.69', '[−5.30, 7.19]', '0.463'], ['KShapeAD', '3.70', '33 / 31', '+5.84', '[−1.66, 13.45]', '0.463'], ['SAND', '4.50', '36 / 28', '+10.64', '[3.88, 17.71]', '0.060'],
                 ['Series2Graph', '4.66', '37 / 27', '+10.26', '[2.82, 18.02]', '0.118'], ['MP', '4.78', '40 / 24', '+12.21', '[5.85, 18.82]', '0.003'], ['PCA', '5.50', '48 / 16', '+17.77', '[11.30, 24.81]', '< 0.001'], ['DAMP', '6.44', '49 / 15', '+28.71', '[22.09, 35.61]', '< 0.001']], hl: 0 },
          { head: [['Family', 'Family'], ['CoAD', 'CoAD'], ['KMeansAD', 'KMeansAD'], ['KMeansAD thắng', 'KMeansAD wins']],
          rows: [['MGAB', '44.7', '1.0', '0/9'], ['SED', '92.0', '92.5', '3/3'], ['IOPS', '31.5', '32.5', '11/17'], ['SVDB', '54.7', '62.5', '14/20'], ['TODS', '24.0', '35.7', '14/15']] }],
        note: ['Lợi thế trung bình của CoAD gần như đến hết từ MGAB. CoAD dùng code tác giả, 1 lần chạy (41.70).', 'CoAD\'s mean advantage comes almost entirely from MGAB. CoAD = authors\' code, one run (41.70).'], src: '§8.1' },
      { slide: '24', title: ['58 series ngoài paper (12 family khác)', '58 series outside the paper (12 other families)'],
        claim: ['Trên dữ liệu mới, CoAD thua KMeansAD 6.61 điểm (p = 0.042).', 'On new data CoAD trails KMeansAD by 6.61 points (p = 0.042).'],
        tables: [{ head: [['Phương pháp', 'Method'], ['VUS-PR', 'VUS-PR']], rows: [['CoAD', '52.36'], ['KMeansAD', '58.96'], ['KShapeAD', '23.28']], hl: 1 }],
        note: ['58/60 series vì KShapeAD lỗi trên 2 series quá ngắn. Bỏ 4 family paper đã loại (NAB, WSD, YAHOO, Stock): CoAD thua 2.28, p = 0.73. KMeansAD được fit trên chính series kiểm tra, có lợi cho nó.', '58/60 series because KShapeAD fails on 2 very short series. Without the 4 families the paper excluded (NAB, WSD, YAHOO, Stock): CoAD trails by 2.28, p = 0.73. KMeansAD is fit on the test series itself, which favours it.'], src: '§8.2' }] },

    { id: 'fixes', label: ['Cải tiến', 'Improvements'], slides: '24 · 25 · A6', cards: [
      { slide: '25 · A6', title: ['Các hướng đã thử, không cần nhãn khi tính score', 'Directions tested, label-free scoring'],
        claim: ['Cải tiến khâu tính score đều tăng trên dữ liệu paper, nhưng trên 12 family mới không biến thể nào hơn CoAD có ý nghĩa.', 'The scoring changes all gain on the paper\'s data, but on the 12 new families none beats CoAD significantly.'],
        tables: [{ head: [['Hướng', 'Direction'], ['Tập xác nhận (34)', 'Confirmation (34)'], ['Dữ liệu mới (60)', 'New data (60)'], ['Kết luận', 'Verdict']],
          rows: [[L('Balance: chia mỗi nhánh cho độ lệch chuẩn trên phần train', 'Balance: divide each branch by its training SD'), '+3.42 (Holm 0.078)', '+1.44 (p 0.44)', L('Chưa tổng quát', 'Not general')],
                 [L('Balance + 2× trọng số tái tạo (demo: Our Improvement)', 'Balance + 2× reconstruction weight (demo: Our Improvement)'), '+4.22 (p 0.001)', '+2.25 (p 0.17)', L('Hơn Balance ổn định', 'Beats Balance consistently')],
                 [L('Mix: trung bình score thô và score đã làm mượt', 'Mix: mean of raw and smoothed score'), '+4.79 (Holm 0.001)', '+1.62 (p 0.16)', L('Chỉ giúp bất thường ngắn', 'Helps short anomalies only')],
                 [L('Gộp theo độ hiếm (4 biến thể)', 'Rarity-based fusion (4 variants)'), L('Đều kém Balance', 'All below Balance'), '—', L('Âm tính', 'Negative')],
                 [L('MAD thay độ lệch chuẩn', 'MAD instead of SD'), L('−1.37 so với Balance', '−1.37 vs Balance'), '—', L('Âm tính', 'Negative')],
                 [L('Nhánh residual độc lập (4 head)', 'Independent residual branch (4 heads)'), L('Không head nào hơn', 'No head better'), '—', L('Âm tính', 'Negative')],
                 [L('Làm sạch dữ liệu train (TODS)', 'Repairing training data (TODS)'), '+0.07', '—', L('Âm tính', 'Negative')]], hl: 1 }],
        note: ['Số là chênh VUS-PR so với CoAD gốc, chấm lại trên cùng model đã train. Chia 30 series thiết kế / 34 xác nhận bằng default_rng(2026), chốt cấu hình trên tập thiết kế. Bỏ WSD thì Balance trên dữ liệu mới còn −0.02.', 'Values are VUS-PR differences vs original CoAD, rescoring the same trained models. 30 design / 34 confirmation series split with default_rng(2026), choices locked on the design set. Without WSD, Balance on new data is −0.02.'], src: '§9' }] }
  ];

  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pick = x => (Array.isArray(x) ? L(x[0], x[1]) : x);

  function table(t) {
    return `<div class="tablewrap"><table class="xtab"><thead><tr>${t.head.map((h, i) => `<th class="${i ? 'r' : ''}">${esc(pick(h))}</th>`).join('')}</tr></thead><tbody>
      ${t.rows.map((r, j) => `<tr class="${j === t.hl ? 'hl' : ''}">${r.map((c, i) => `<td class="${i ? 'r' : ''}">${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function card(c) {
    const note = pick(c.note);
    return `<div class="panel xcard"><div class="ph"><h2>${esc(pick(c.title))}</h2></div>
      <div class="pb"><p class="xclaim">${esc(pick(c.claim))}</p>${c.tables.map(table).join('')}
      <div class="note">${note ? esc(note) + ' · ' : ''}${L('Nguồn: báo cáo', 'Source: report')} ${esc(c.src)}</div></div></div>`;
  }
  window.EXP = {
    views: () => VIEWS.map(v => ({ id: v.id, label: pick(v.label), slides: v.slides })),
    html: id => { const v = VIEWS.find(x => x.id === id); return v ? v.cards.map(card).join('') : ''; }
  };
})();
