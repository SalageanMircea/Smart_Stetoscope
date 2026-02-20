import os, random, warnings
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

warnings.filterwarnings('ignore')

class CFG:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = 6 if torch.cuda.is_available() else 4

    sr, seg_sec, max_sec = 4000, 5.0, 12.0 # 4000 esantioane/sec, 5 sec/seg, 12 sec dintr-un wav
    bp_low, bp_high = 20, 500 
    n_fft, hop, n_mels, fmin, fmax = 512, 64, 80, 20, 500 
    K_train, K_eval = 12, 20  # pt train se iau 12 seg si pt validare 20
    epochs, patience = 250, 40
    lr_max, weight_decay, grad_clip = 5e-5, 0.01, 1.0 
    spec_time_mask, spec_freq_mask = 8, 5
    wave_noise, wave_gain, wave_shift = 0.001, (0.95, 1.05), 0.05     
    label_smooth = 0.05  
    focal_gamma = 1.5  #focus pe exemple grele
    focal_alpha = [1.2, 1.8, 1.5, 1.0]  # penalizari in fct de clasa

CLASSES = ["normal", "murmur", "artifact", "extra"]
NUM_CLASSES = 4

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# inlocuiește valorile invalide (NaN/Inf)
# scalează semnalul cu radacina mediei patratelor
def normalize(x):
    
    x = np.nan_to_num(x.astype(np.float64))
    rms = np.sqrt(np.mean(x**2))

    if rms > 1e-9: 
        x = x / rms

    p99 = np.percentile(np.abs(x), 99.5)

    if p99 > 1e-9: 
        x = np.clip(x, -p99, p99)

    return x

# bandpass Butterworth pentru a pastra zona utila a sunetelor cardiace
def bandpass(x, sr, lo, hi):

    nyq = sr / 2
    hi = min(hi, nyq - 1)

    if lo >= hi: return x

    sos = butter(6, [lo/nyq, hi/nyq], btype="band", output="sos")

    return sosfiltfilt(sos, x)



def resample(x, sr_old, sr_new):

    if sr_old == sr_new: 
        return x

    return np.interp(np.linspace(0,1,int(len(x)*sr_new/sr_old)), np.linspace(0,1,len(x)), x)

def get_segments(x, seg_len, K):

    n = len(x)

    if n <= seg_len: 
        return [0] * K
    
    starts = []
    env = np.abs(hilbert(x)) if len(x) > 100 else np.abs(x)
    thresh = np.percentile(env, 65)

    #vf locale peste prag
    peaks = [i for i in range(1, len(env)-1) 
             if env[i] > thresh and env[i] > env[i-1] and env[i] > env[i+1]]

    for p in peaks[::max(1, len(peaks)//(K//2+1))][:K//2]:
        starts.append(max(0, min(p - seg_len//2, n - seg_len)))

    for pos in [0, (n-seg_len)//2, n-seg_len]:
        if len(starts) < K: starts.append(pos)

    while len(starts) < K:
        starts.append(random.randint(0, n - seg_len))

    return starts[:K]



def augment_wave(x):

    n = len(x)

    x = np.roll(x, random.randint(-int(CFG.wave_shift*n), int(CFG.wave_shift*n)+1))
    x = x * random.uniform(*CFG.wave_gain) + np.random.randn(n) * CFG.wave_noise

    if random.random() < 0.3: 
        x = -x

    return x


def create_mel_fb(sr, n_fft, n_mels, fmin, fmax):

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr/2, n_freqs)

    hz2mel = lambda h: 2595 * np.log10(1 + h/700)
    mel2hz = lambda m: 700 * (10**(m/2595) - 1)

    mel_pts = np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr/2-1)), n_mels+2)
    hz_pts = mel2hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        l, c, r = hz_pts[i], hz_pts[i+1], hz_pts[i+2]

        for j in range(n_freqs):
            if l <= freqs[j] <= c: 
                fb[i,j] = (freqs[j]-l)/(c-l+1e-12)

            elif c < freqs[j] <= r: 
                fb[i,j] = (r-freqs[j])/(r-c+1e-12)

    return fb / (fb.sum(1, keepdims=True) + 1e-12)



def compute_logmel(wave, mel_fb):

 # STFT + proiectare pe Mel + log + standardizare

    win = torch.hann_window(CFG.n_fft, device=wave.device)
    stft = torch.stft(wave, CFG.n_fft, CFG.hop, CFG.n_fft, win, center=False, return_complex=True)

    mel = (mel_fb @ stft.abs()**2).clamp(1e-12) #
    logmel = torch.log(mel)


# normalizeaza pe fiecare segment (ajuta stabilitatea)
    return ((logmel - logmel.mean()) / (logmel.std() + 1e-6)).unsqueeze(0)


def spec_augment(spec):   # mascare pe timp si frecventa (anti-overfit)

    d, F, T = spec.shape

    for d in range(random.randint(1,2)):
        if T > 10: 
            t = random.randint(1, min(CFG.spec_time_mask, T//3))
            spec[:, :, random.randint(0,T-t):random.randint(0,T-t)+t] = 0

        if F > 10:
            f = random.randint(1, min(CFG.spec_freq_mask, F//3))
            spec[:, random.randint(0,F-f):random.randint(0,F-f)+f, :] = 0

    return spec



class HeartDataset(Dataset):

    def __init__(self, items, mel_fb, train=False):
        self.items, self.mel_fb, self.train = items, mel_fb, train
        self.seg_len = int(CFG.sr * CFG.seg_sec)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x, sr = sf.read(path, dtype='float32')

        if x.ndim > 1: 
            x = x.mean(1)

        # resample + taiere la max_sec + bandpass + normalize
        x = normalize(resample(x, sr, CFG.sr))[:int(CFG.sr * CFG.max_sec)]
        x = normalize(bandpass(x, CFG.sr, CFG.bp_low, CFG.bp_high))

        if self.train: 
            x = augment_wave(x)

        K = CFG.K_train if self.train else CFG.K_eval
        specs = []


 # construim K spectrograme pentru acelasi fisier
        for st in get_segments(x, self.seg_len, K):
            seg = x[st:st+self.seg_len]

            if len(seg) < self.seg_len: 
                seg = np.pad(seg, (0, self.seg_len - len(seg)))
            wave = torch.from_numpy(seg.astype(np.float32))
            spec = compute_logmel(wave, self.mel_fb)
            
            if self.train: 
                spec = spec_augment(spec)
            specs.append(spec)

        return torch.stack(specs), torch.tensor(label, dtype=torch.long)



class AttentionPooling(nn.Module):

    def __init__(self, dim):
        super().__init__()   # Multi-Layer Perceptron da scoruri de atentie pt fiecare segment
        self.attention = nn.Sequential(nn.Linear(dim, dim//2), nn.Tanh(), nn.Linear(dim//2, 1))

    def forward(self, x):
        return (x * torch.softmax(self.attention(x), dim=1)).sum(dim=1)

class Model(nn.Module):
    def __init__(self, num_classes=4):

        super().__init__()

        #  backbone standard (ResNet34), pt 1 canal
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_feat = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_feat, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes))
        
        self.attention_pool = AttentionPooling(num_classes)
    def forward(self, x):
        x = F.interpolate(x, (160, 160), mode='bilinear', align_corners=False)

        return self.backbone(x)

def focal_loss(logits, targets, gamma=1.5, alpha=None):

    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=CFG.label_smooth)
    pt = torch.exp(-ce)
    fl = ((1 - pt) ** gamma) * ce

    if alpha:
        fl = fl * torch.tensor(alpha, device=logits.device)[targets]

    return fl.mean()

def load_dataset(root):
    items = []
    label_map = {"normal":0, "murmur":1, "artifact":2, "extra":3, "extrastole":3, "extrasystole":3}
    wav_files = {}

    for dp, _, fns in os.walk(os.path.join(root, "Datasets")):
        for f in fns:
            if f.lower().endswith('.wav'): wav_files[f.lower()] = os.path.join(dp, f)

    csv_path = os.path.join(root, "Datasets_csv", "set_a.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            fname = os.path.basename(str(row.get('fname', ''))).lower()
            label = str(row.get('label', '')).lower()

            if fname in wav_files and label in label_map:
                items.append((wav_files[fname], label_map[label]))

    for fname, fpath in wav_files.items():
        if 'set_b' not in fpath.lower(): 
            continue
        
        for prefix, lbl in [('normal',0), ('murmur',1), ('artifact',2), ('extra',3)]:
            if fname.startswith(prefix): 
                items.append((fpath, lbl)); break
            
    print(f"Loaded {len(items)} samples: {Counter(y for _,y in items)}")
    
    return items

def split_data(items, test_size=0.2):
    labels = np.array([y for _,y in items])

    for seed in range(42, 242):
        sss = StratifiedShuffleSplit(1, test_size=test_size, random_state=seed)
        tr_idx, va_idx = next(sss.split(np.zeros(len(labels)), labels))

        if all(Counter(labels[va_idx])[i] > 0 for i in range(NUM_CLASSES)):
            return [items[i] for i in tr_idx], [items[i] for i in va_idx]
        
    raise RuntimeError("Could not split data")


@torch.no_grad()
def evaluate(model, loader, device):


    # evaluare: rulam backbone pe toate segmentele, apoi facem pooling
    model.eval()
    preds, labels = [], []

    for specs, lbl in loader:
        specs = specs.to(device)
        B, K = specs.shape[:2]

        # forward pe toate segmentele
        logits = model(specs.view(B*K, *specs.shape[2:])).view(B, K, -1)
        pooled = model.attention_pool(logits)
        preds.extend(pooled.argmax(1).cpu().numpy())
        labels.extend(lbl.numpy())

    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()

    return acc, f1_score(labels, preds, average='macro', zero_division=0), labels, preds



def main():
    set_seed(CFG.seed)
    device = CFG.device
    print(f"Device: {device.upper()}")


    root = os.path.dirname(os.path.abspath(__file__))
    items = load_dataset(root)

    if not items: 
        return print("ERROR: No data found!")
    
    train_items, val_items = split_data(items)
    print(f"Train: {len(train_items)}, Val: {len(val_items)}")
    
    mel_fb = torch.from_numpy(create_mel_fb(CFG.sr, CFG.n_fft, CFG.n_mels, CFG.fmin, CFG.fmax))
    counts = Counter(y for _,y in train_items)
    weights = [1/counts[y] * [0.8, 2.0, 1.8, 1.5][y] for _,y in train_items]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(HeartDataset(train_items, mel_fb, True), CFG.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(HeartDataset(val_items, mel_fb, False), CFG.batch, num_workers=0)
    all_loader = DataLoader(HeartDataset(items, mel_fb, False), CFG.batch, num_workers=0)
    model = Model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr_max, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    best_f1, patience_cnt = 0, 0


    for epoch in range(1, CFG.epochs + 1):
        model.train()
        train_loss = 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            B, K = specs.shape[:2]
            logits = model(specs.view(B*K, *specs.shape[2:])).view(B, K, -1)
            pooled = model.attention_pool(logits)
            loss = focal_loss(pooled, labels, CFG.focal_gamma, CFG.focal_alpha)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            train_loss += loss.item()


        scheduler.step()
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"E{epoch:03d} | Loss={train_loss/len(train_loader):.4f} | ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}")
     
     
        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
            torch.save({'model': model.state_dict(), 'best_f1': best_f1, 'mel_fb': mel_fb.cpu()}, 'best_model_FINAL.pt')
            print(f"  >> BEST! F1={best_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.patience: 
                print(f"\nEarly stop at epoch {epoch}"); break
            
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

# incarc best value si evaluez setu (sanity check)
    ckpt = torch.load('best_model_FINAL.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])


    all_acc, all_f1, all_true, all_pred = evaluate(model, all_loader, device)


    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8,7))
    plt.imshow(cm, cmap='Blues'); plt.colorbar()
    
    
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
            

    plt.xticks(range(4), CLASSES, rotation=45); plt.yticks(range(4), CLASSES)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc={all_acc:.3f}, F1={all_f1:.3f})')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=200); plt.close()

    
    print(f"\n{'='*60}\nFINAL: Acc={all_acc:.4f}, F1={all_f1:.4f}\n{'='*60}")
    print(classification_report(all_true, all_pred, target_names=CLASSES, zero_division=0))

if __name__ == "__main__":
    main()
