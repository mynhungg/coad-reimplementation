"""CoAD reimplementation (code mode), extracted verbatim from CoAD_reimplementation.ipynb cells
6, 10, 14, 18, 24, 26, 32 (config, distortions, STFT/patches, model, windows/loss, training, scoring).
Do not edit here: change the notebook and re-extract (see demo/README.md)."""
import random
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_dataset = evaluate = None          # injected by precompute.py (authors' loader and evaluator)

# ---- notebook cell 6
FAITHFUL_TO_CODE = True    # False: follow the paper (main version); True: follow the authors' code (D1-D21)
SEEDS = [46, 47, 48]
PATCH = 8                  # P
WINDOW_MULTIPLE = 4        # T = 4 x dominant period
N_FFT = 4                  # K, STFT size
LAMBDA = 10                # weight of the reconstruction loss
LR = 0.002
EPOCHS = 300
BATCH_SIZE = 128

def use_code_details(faithful=None):
    """Per-call override of FAITHFUL_TO_CODE (None = use the global switch)."""
    return FAITHFUL_TO_CODE if faithful is None else faithful

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_series(group):
    """One TSB-AD series (min-max scaled on train) plus the window settings of the authors' code."""
    train, test, label, period = [values[0] for values in load_dataset('TSB-AD', [str(group)])]
    multiple = WINDOW_MULTIPLE if len(train) // period > WINDOW_MULTIPLE else len(train) // period
    window = (period * multiple) // PATCH * PATCH
    period = min(period, window)
    n = len(train)
    train_stride = 2 if n < 1000 else 6 if n < 10000 else 10 if n < 60000 else PATCH if n < 100000 else period
    return dict(group=group, train=train, test=test, label=label,
                period=period, window=window, train_stride=train_stride)



# ---- notebook cell 10
DISTORTIONS = ['uniform', 'scale', 'jitter', 'mirror']

def distort(x, period, kind, faithful=None):
    """x: clean windows (B, T, 1). Returns (distorted copy, point labels (B, T))."""
    faithful = use_code_details(faithful)
    x = x.clone()
    B, T, _ = x.shape
    n = int(period * np.random.uniform(0.1, 1))           # segment length, one draw per batch
    if n == 0:
        n = 2 if kind == 'uniform' else 1
    if n >= T:
        n = T - 2
    start = torch.randint(0, T - n, (B, 1))               # segment start, one per window
    if faithful and kind == 'mirror':
        torch.randint(0, T - n, (B, 1))                   # unused draw of the authors' mirror_flip: keeps the RNG stream aligned
    rows = torch.arange(B)[:, None]

    if kind == 'scale':
        factor = np.random.randint(2, 5)
        if faithful or np.random.rand() < 0.5:            # lengthen: first n / factor points, each repeated
            n = n // factor * factor
            source = start + torch.arange(n) // factor
        else:                                             # shorten: every factor-th point of a longer span
            source = (start + torch.arange(n) * factor).clamp(max=T - 1)
    cols = start + torch.arange(n)
    segment = x[rows, cols]                               # (B, n, 1)

    if kind == 'uniform':
        if faithful:
            x[rows, cols] = x[rows, start]
        else:
            low, high = segment.min(1, keepdim=True).values, segment.max(1, keepdim=True).values
            x[rows, cols] = low + (high - low) * torch.rand(B, 1, 1, device=x.device)
    elif kind == 'scale':
        x[rows, cols] = x[rows, source]
    elif kind == 'jitter':
        if faithful:
            std = 0.05 * (x.max(1, keepdim=True).values - x.min(1, keepdim=True).values)
        else:
            std = 0.1 ** 0.5
        x[rows, cols] = segment + torch.randn(B, n, 1, device=x.device) * std
    elif kind == 'mirror':
        if faithful or np.random.rand() < 0.5:            # across the y-axis: reverse time
            x[rows, cols] = torch.flip(segment, [1])
        else:                                             # across the x-axis: mirror values inside the segment range
            x[rows, cols] = segment.min(1, keepdim=True).values + segment.max(1, keepdim=True).values - segment
    else:
        raise ValueError(f'unknown distortion {kind}')

    labels = torch.zeros(B, T, device=x.device)
    labels[rows, cols] = 1
    return x, labels

# ---- notebook cell 14
def stft_features(x):
    """x: (B, T) -> (B, T, 2 * bins): real and imaginary part of every bin, one frame per time point."""
    spectrum = torch.stft(x, N_FFT, hop_length=N_FFT // 4, window=torch.hann_window(N_FFT, device=x.device),
                          normalized=True, onesided=True, return_complex=True)
    spectrum = torch.view_as_real(spectrum)            # (B, bins, frames, 2)
    spectrum = spectrum.permute(0, 2, 1, 3).flatten(2)  # (B, frames, bins * 2)
    return spectrum[:, :x.shape[1]]                     # frames = T + 1 (centred STFT); keep T

def make_patches(x, x_freq):
    """x: (B, T), x_freq: (B, T, F) -> time patches (B, N, P) and frequency patches (B, N, P * F)."""
    B, T = x.shape
    return x.reshape(B, T // PATCH, PATCH), x_freq.reshape(B, T // PATCH, -1)

# ---- notebook cell 18
def head(in_size, hidden, faithful):
    """Patch classifier: linear in the paper (Eq. 2, 3, 8), two-layer MLP in the code (D4)."""
    if faithful:
        return nn.Sequential(nn.Linear(in_size, hidden), nn.SELU(), nn.Linear(hidden, 1, bias=False))
    return nn.Sequential(nn.Linear(in_size, 1))

def gru(size, faithful):
    """Bi-GRU encoder (Section 3.2): 2 layers and dropout 0.1 in the code, 3 layers in the paper (D1, D2)."""
    return nn.GRU(size, size, num_layers=2 if faithful else 3, batch_first=True, bidirectional=True,
                  dropout=0.1 if faithful else 0.0)

def init_linear(module):
    """Xavier-uniform weights and zero biases for every linear layer, as in the authors' code."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class TimeFreqClassifier(nn.Module):
    """Module 3 (Eq. 1-4) and the residual classifier of module 5 (Eq. 7-8)."""

    def __init__(self, faithful):
        super().__init__()
        self.faithful = faithful
        freq_size = PATCH * 2 * (N_FFT // 2 + 1)                  # 8 points x 3 bins x (real, imaginary) = 48
        hidden = 3 * freq_size if faithful else 24                # D5
        activation = [nn.SELU()] if faithful else []              # D18
        self.proj_freq = nn.Sequential(nn.Linear(freq_size, hidden), *activation)   # W_f^p
        self.proj_time = nn.Sequential(nn.Linear(PATCH, hidden), *activation)       # W_t^p
        self.encoder = gru(hidden, faithful)                      # shared by both branches (D3)
        self.classifier_time = head(2 * hidden, hidden, faithful)       # W_t (created first, as in the authors' code,
        self.classifier_freq = head(2 * hidden, hidden, faithful)       # W_f  so a seed gives the same initial weights)
        self.classifier_residual = head(2 * hidden, hidden, faithful)   # W_r, shared by both residual branches
        init_linear(self)

    def features(self, x):
        """x: (B, T) -> encoder outputs of the frequency and of the time patches, each (B, N, 2H)."""
        time_patches, freq_patches = make_patches(x, stft_features(x))
        projected = torch.cat([self.proj_freq(freq_patches), self.proj_time(time_patches)])
        encoded, _ = self.encoder(projected)
        return encoded.chunk(2)

    def forward(self, x):
        """Eq. 1-4: logit of A = max(A_f, A_t), (B, N, 1). Max of logits = max of probabilities (sigmoid is increasing)."""
        h_freq, h_time = self.features(x)
        return torch.maximum(self.classifier_freq(h_freq), self.classifier_time(h_time))

    def residual_logit(self, x, x_rec):
        """Eq. 7-8: logit of A^r from the feature difference between the input and its reconstruction."""
        h_freq, h_time = self.features(torch.cat([x, x_rec]))
        (freq_in, freq_rec), (time_in, time_rec) = h_freq.chunk(2), h_time.chunk(2)
        return torch.maximum(self.classifier_residual(freq_in - freq_rec),
                             self.classifier_residual(time_in - time_rec))


class SoftMaskMAE(nn.Module):
    """Module 4 (Eq. 5-6): patch embeddings softly replaced by a learned mask embedding, then reconstructed."""

    def __init__(self, faithful):
        super().__init__()
        hidden = 3 * PATCH                                        # 24 in both modes
        self.proj = nn.Linear(PATCH, hidden)                      # W_m
        self.act = nn.SELU() if faithful else nn.Identity()       # D18
        self.encoder = gru(hidden, faithful)
        self.head = nn.Linear(2 * hidden, PATCH)                  # W_o
        self.mask_embedding = nn.Parameter(torch.randn(hidden))   # E_mask, one vector repeated over patches (D19)
        init_linear(self)

    def forward(self, x, prob):
        """x: (B, T), prob: patch anomaly probability (B, N, 1) -> reconstruction X_r (B, T)."""
        B, T = x.shape
        embedding = self.act(self.proj(x.reshape(B, T // PATCH, PATCH)))
        mixed = prob * self.mask_embedding + (1 - prob) * embedding    # Eq. 5
        encoded, _ = self.encoder(mixed)
        return self.head(encoded).reshape(B, T)                         # Eq. 6


class CoADReimpl(nn.Module):
    def __init__(self, faithful=None):
        super().__init__()
        self.faithful = use_code_details(faithful)
        self.classifier = TimeFreqClassifier(self.faithful)
        self.mae = SoftMaskMAE(self.faithful)

    def forward(self, x):
        """Training pass. x: (B, T) -> logits of A and A^r (B, N, 1), reconstruction X_r (B, T)."""
        logit = self.classifier(x)
        x_rec = self.mae(x, torch.sigmoid(logit))
        return logit, self.classifier.residual_logit(x, x_rec), x_rec

    @torch.no_grad()
    def predict(self, x):
        """Inference (call .eval() first). x: (B, T) -> A_c (B, N, 1) (Eq. 9) and the reconstruction (B, T)."""
        prob = torch.sigmoid(self.classifier(x))
        x_rec = self.mae(x, prob)
        prob = (prob + torch.sigmoid(self.classifier.residual_logit(x, x_rec))) / 2
        if self.faithful:                   # D10: reconstruct again, masked with A_c
            x_rec = self.mae(x, prob)
        return prob, x_rec

    def load_official(self, official):
        """Copy the weights of an authors' PatchFrequencyMask (code mode; test T4)."""
        renamed = {k.replace('detection_net.', 'classifier.').replace('reconstruction_net.', 'mae.'): v
                   for k, v in official.state_dict().items()
                   if k.startswith(('detection_net.', 'reconstruction_net.'))}
        self.load_state_dict(renamed)

# ---- notebook cell 24
def make_windows(values, window, stride):
    """values (L,) -> windows (M, window) float32 and the front padding length."""
    L = len(values)
    pad = 0 if stride == 1 or (L - window) % stride == 0 else stride - (L - window) % stride
    padded = np.concatenate([np.zeros(pad), values])
    starts = np.arange(0, len(padded) - window + 1, stride)
    return np.stack([padded[s:s + window] for s in starts]).astype(np.float32), pad

def windows_to_series(window_values, length, stride, pad):
    """window_values (M, window) -> (length,): mean over the overlapping windows, padding removed."""
    total, count = np.zeros(pad + length), np.zeros(pad + length)
    for i, values in enumerate(window_values):
        total[i * stride:i * stride + len(values)] += values
        count[i * stride:i * stride + len(values)] += 1
    return total[pad:] / count[pad:]

def patch_labels(labels):
    """Point labels (B, T) -> patch labels (B, N, 1)."""
    B, T = labels.shape
    return labels.reshape(B, T // PATCH, PATCH).amax(-1, keepdim=True)

def training_batch(clean, period, faithful, code_distortion=None):
    """Clean windows (B, T) -> model input, point labels, reconstruction target.
    code_distortion (M6-C only): distortion details D12-D15 chosen apart from the batch layout D9."""
    code_distortion = faithful if code_distortion is None else code_distortion
    distorted, labels = distort(clean.unsqueeze(-1), period, random.choice(DISTORTIONS), code_distortion)
    distorted = distorted.squeeze(-1)
    if faithful:                                    # D9: clean copy + distorted copy
        return torch.cat([clean, distorted]), torch.cat([torch.zeros_like(labels), labels]), torch.cat([clean, clean])
    return distorted, labels, clean

def coad_loss(model, x, labels, target, code_loss=None):
    """Eq. 10. Returns (total, classification term, reconstruction term).
    code_loss (M6-C only): loss details D6-D8, D20 chosen apart from the model."""
    logit, residual_logit, x_rec = model(x)
    y = patch_labels(labels)
    if (model.faithful if code_loss is None else code_loss):
        pos_weight = torch.tensor([2.0], device=x.device)                                    # D6
        classify = (F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight) +
                    F.binary_cross_entropy_with_logits(residual_logit, y, pos_weight=pos_weight)) / 2   # D7
        error = (x_rec - target).pow(2).reshape(y.shape[0], y.shape[1], PATCH)
        anomalous = (error * y).sum() / (y.sum() * PATCH + 1e-7)
        normal = (error * (1 - y)).sum() / ((1 - y).sum() * PATCH + 1e-7)
        recon = 0.5 * anomalous + 0.5 * normal                                               # D8
    else:
        prob = (torch.sigmoid(logit) + torch.sigmoid(residual_logit)) / 2                    # A_c, Eq. 9
        classify = F.binary_cross_entropy(prob, y)
        recon = PATCH * F.mse_loss(x_rec, target)      # Eq. 10: squared norm summed over a patch, averaged over patches (D20)
    return classify + LAMBDA * recon, classify, recon

# ---- notebook cell 26
def train_model(series, faithful=None, seed=46, epochs=None, log_every=50, groups=None):
    """Train CoADReimpl on the training part of one series. Returns the model and the mean losses per epoch.
    groups (M6-C only): set of code-only detail groups (section 12) applied on top of paper mode."""
    epochs = EPOCHS if epochs is None else epochs
    set_seed(seed)
    if groups is None:
        faithful = use_code_details(faithful)
        code_distortion = faithful
        model = CoADReimpl(faithful).to(DEVICE)
    else:
        faithful, code_distortion = 'training' in groups, 'distortion' in groups
        model = HybridCoAD(groups).to(DEVICE)
    windows, _ = make_windows(series['train'], series['window'], series['train_stride'])
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99) if faithful else None   # D16
    use_amp = faithful and DEVICE.type == 'cuda'                                                            # D16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = []
    for epoch in range(epochs):
        model.train()
        sums = np.zeros(3)
        for (clean,) in loader:
            x, labels, target = training_batch(clean.to(DEVICE), series['period'], faithful, code_distortion)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16) if use_amp else nullcontext():
                losses = coad_loss(model, x, labels, target, faithful)
            scaler.scale(losses[0]).backward()
            scaler.step(optimizer)
            scaler.update()
            sums += [value.item() for value in losses]
        if scheduler:
            scheduler.step()
        history.append(dict(zip(['loss', 'classify', 'recon'], sums / len(loader))))
        if log_every and (epoch + 1) % log_every == 0:
            print(f'  epoch {epoch + 1}/{epochs}: ' + ', '.join(f'{k} {v:.4f}' for k, v in history[-1].items()),
                  flush=True)
    return model, pd.DataFrame(history)

# ---- notebook cell 32
@torch.no_grad()
def predict_series(model, values, window, stride):
    """Whole series -> per-point A_c and reconstruction, averaged over overlapping windows."""
    windows, pad = make_windows(values, window, stride)
    model.eval()
    probs, recs = [], []
    for batch in torch.from_numpy(windows).split(2 * BATCH_SIZE):
        prob, rec = model.predict(batch.to(DEVICE))
        probs.append(prob.repeat_interleave(PATCH, dim=1).squeeze(-1).cpu())
        recs.append(rec.cpu())
    return (windows_to_series(torch.cat(probs).numpy(), len(values), stride, pad),
            windows_to_series(torch.cat(recs).numpy(), len(values), stride, pad))

def moving_average(score, width):
    """Centred moving average keeping the length (np.convolve 'same', as in the authors' code)."""
    return np.convolve(score, np.ones(width) / width, mode='same')

def anomaly_score(model, series):
    """Per-point anomaly score of the test series."""
    period, window = series['period'], series['window']
    test_prob, test_rec = predict_series(model, series['test'], window, period)
    if not model.faithful:
        return moving_average(test_prob + np.abs(series['test'] - test_rec), period)    # Eq. 11
    train_prob, train_rec = predict_series(model, series['train'], window, period)      # D11
    recon = np.abs(np.abs(series['test'] - test_rec) - np.abs(series['train'] - train_rec).mean())
    classify = np.abs(test_prob - train_prob.mean())
    recon, classify = moving_average(recon, period), moving_average(classify, period)
    recon[:period] = classify[:period] = 0
    return (recon + classify) / 2

METRIC_KEYS = ['f1_score', 'auc_prc', 'auc_roc', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR']

def score_metrics(score, series):
    """Checks of test T7, then the repository evaluator (values in 0-1)."""
    assert len(score) == len(series['label']), 'score and label lengths differ'
    assert np.isfinite(score).all(), 'score contains NaN or Inf'
    result = evaluate(ground_truth=series['label'], anomaly_scores=score, subsequence=series['period'], mode='test')
    row = {k: float(result.point_wise[k]) for k in METRIC_KEYS[:3]}
    row.update({k: float(result.vus[k]) for k in METRIC_KEYS[3:]})
    return row

