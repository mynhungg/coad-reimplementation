"""Writes train_classify_mask_t12.py: the authors' train_classify_mask.py plus the 5 rows of Table 2 that the
published code does not implement. Definitions follow Appendix G of the paper.

Rows already in the authors' code (run unchanged, model names of runners/run_ablation_CoAD.sh):
  freq_hard (Cooperative, Random)      freq_gating (Cooperative, Grating)   guide_hard (Guide w/ Hard Mask)
  time_soft (Cooperative, Time + Soft) freq_time_point (Step Level)         freq_time_window (Window Level)
  freq_time_add (Feature_Add)          freq_time_gate (Feature_Gate)        freq_time_mean (Decision_Mean)

Rows added here (everything else - data, windows, distortions, optimiser, epochs, scoring code - is the authors'):
  oe_time         OE alone, Time            Appendix G: "takes only the time domain features for classification"
                  classifier = DetectionNetTime, loss = BCE only, score = classifier
  oe_freq_time    OE alone, Time+Frequency  Appendix G: "maximum fusion ... of time and frequency domains"
                  classifier = DetectionNetTimeFreq (max of the two classifiers), loss = BCE only, score = classifier
  mae_random      MAE alone, Random mask    Appendix G: "applies either the random or grating masking strategy"
                  reconstruction = HardReconstructionNet (random mask), loss = omega * MSE only, score = reconstruction
  mae_gating      MAE alone, Grating mask   reconstruction = GatingReconstructionNet (odd/even patches),
                  loss = omega * MSE only, score = reconstruction
  guide_wo_score  Guide w/o Score           Appendix G: "the same guided soft masking strategy as CoAD, but relies
                  solely on MAE's reconstruction error for final anomaly scoring"
                  = CoAD (freq_tf_v2, same construction and loss), score = reconstruction only
Interpretation left to us: in the 'alone' rows the unused module is still built (its loss is not optimised and its
output not scored); the inputs are the authors' (half clean, half distorted windows).

usage: python t12_ablation_patch.py <repo root>
"""
import sys
from pathlib import Path

EXTRA = '''
# --- t12: rows of Table 2 missing from the published code (see t12_ablation_patch.py) ---
# name: (detect_mode, recon_mode, num_layers, loss: both/classify/recon, score: both/classify/recon)
T12_EXTRA = {'oe_time': ('time', 'soft', 3, 'classify', 'classify'),
             'oe_freq_time': ('freq_time', 'soft', 3, 'classify', 'classify'),
             'mae_random': ('freq_time_v2', 'hard', 3, 'recon', 'recon'),
             'mae_gating': ('freq_time_v2', 'gating', 3, 'recon', 'recon'),
             'guide_wo_score': ('freq_time_v2', 'soft', 2, 'both', 'recon')}
T12_LOSS = T12_SCORE = None
'''


def patch(src: str) -> str:
    def rep(old, new, count=1):
        nonlocal src
        assert src.count(old) == count, (old[:70], src.count(old))
        src = src.replace(old, new)

    rep("import random\n", "import random\n" + EXTRA)
    rep("    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        "    if args.model_name in T12_EXTRA:\n"
        "        T12_LOSS, T12_SCORE = T12_EXTRA[args.model_name][3:]\n")
    rep('        if args.model_name == "mix_soft":\n',
        '        if args.model_name in T12_EXTRA:\n'
        '            _d, _r, _l = T12_EXTRA[args.model_name][:3]\n'
        '            model = PatchFrequencyMask(patch_size=args.patch_size, expansion_ratio=3, num_layers=_l, n_fft=4,\n'
        '                                       patch_num=None, detect_mode=_d, recon_mode=_r, omega=args.omega)\n'
        '        elif args.model_name == "mix_soft":\n')
    rep("                    loss, classify_loss, reconstruction_loss = model(data, subsequence_length=subsequence_length)\n",
        "                    loss, classify_loss, reconstruction_loss = model(data, subsequence_length=subsequence_length)\n"
        "                    if T12_LOSS == 'classify':\n"
        "                        loss = classify_loss\n"
        "                    elif T12_LOSS == 'recon':\n"
        "                        loss = model.omega * reconstruction_loss\n")
    rep("    test_anomaly_score = (test_classify_anomaly_score + test_recon_anomaly_score) / 2\n",
        "    test_anomaly_score = (test_classify_anomaly_score + test_recon_anomaly_score) / 2\n"
        "    if T12_SCORE == 'classify':\n"
        "        test_anomaly_score = test_classify_anomaly_score\n"
        "    elif T12_SCORE == 'recon':\n"
        "        test_anomaly_score = test_recon_anomaly_score\n")
    return src


if __name__ == '__main__':
    repo = Path(sys.argv[1])
    out = repo / 'train_classify_mask_t12.py'
    out.write_text(patch((repo / 'train_classify_mask.py').read_text(encoding='utf-8')), encoding='utf-8')
    print('written', out)
