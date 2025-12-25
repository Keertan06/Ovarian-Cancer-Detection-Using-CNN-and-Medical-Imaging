import os, re, json, math, random, time, argparse
from glob import glob
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Config via CLI
# ---------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", default="clinical.tsv", help="Clinical TSV path")
    ap.add_argument("--tcga_root", default="TCGA-OV", help="Root folder with TCGA-OV DICOMs")
    ap.add_argument("--out_dir", default="processed_data/stage_images", help="Where to save extracted PNGs")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--slices_per_series", type=int, default=3, help="How many slices to sample per series")
    ap.add_argument("--min_slices", type=int, default=3, help="A folder counts as a series if it has >= this many DICOMs")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ---------------------------
# Utilities
# ---------------------------
ROMAN = {"I":"Stage I","II":"Stage II","III":"Stage III","IV":"Stage IV"}
ID_RE = re.compile(r"(TCGA-\w{2}-\w{4})", re.IGNORECASE)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def normalize_figo(txt):
    if not isinstance(txt, str):
        return None
    t = txt.upper().replace("FIGO","").replace("STAGE","").strip()
    m = re.search(r"(I{1,3}V?)", t)  # I, II, III, IV
    if not m: return None
    key = m.group(1)
    return ROMAN.get(key, None)

def safe_read_dcm(path):
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        # robust windowing to 0–255
        arr = np.nan_to_num(arr)
        arr -= arr.min()
        if arr.max() > 0: arr = arr / arr.max()
        arr = (arr * 255.0).clip(0,255).astype(np.uint8)
        return arr
    except Exception:
        return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# Slice extraction
# ---------------------------
def collect_series_dirs(tcga_root, min_slices=3):
    series = []
    for r, _, files in os.walk(tcga_root):
        dcm_count = len([f for f in files if f.lower().endswith(".dcm")])
        if dcm_count >= min_slices:
            series.append(r)
    return series

def submitter_from_path(path):
    m = ID_RE.search(path)
    return m.group(1) if m else None

def pick_evenly_spaced(files, k):
    if k >= len(files):
        return files
    idxs = np.linspace(0, len(files)-1, k).round().astype(int)
    return [files[i] for i in idxs]

def extract_pngs(clinical_map, tcga_root, out_dir, img_size=224, slices_per_series=3, min_slices=3):
    ensure_dir(out_dir)
    series_dirs = collect_series_dirs(tcga_root, min_slices=min_slices)
    saved = []

    for sdir in series_dirs:
        sid = submitter_from_path(sdir)
        if sid is None: 
            continue
        if sid not in clinical_map: 
            continue  # skip patients without FIGO stage

        stage_text, stage_idx = clinical_map[sid]
        dcm_files = sorted(glob(os.path.join(sdir, "*.dcm")))
        if len(dcm_files) < min_slices: 
            continue

        # pick K slices through the volume
        chosen = pick_evenly_spaced(dcm_files, slices_per_series)
        for dcm_path in chosen:
            arr = safe_read_dcm(dcm_path)
            if arr is None: 
                continue
            # to RGB PIL
            pil = Image.fromarray(arr).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            base = f"{sid}__{stage_idx}__{os.path.basename(dcm_path).replace('.dcm','')}.png"
            out_path = os.path.join(out_dir, base)
            pil.save(out_path)
            saved.append((out_path, stage_idx))
    return saved

# ---------------------------
# Dataset
# ---------------------------
class StageImageDataset(Dataset):
    def __init__(self, items, img_size=224):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(y, dtype=torch.long)

# ---------------------------
# Model
# ---------------------------
def make_model(num_classes=4):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters(): 
        p.requires_grad = False
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return m

# ---------------------------
# Main training
# ---------------------------
def main():
    args = get_args()
    set_seed(args.seed)

    print("✅ Loading clinical.tsv...")
    df = pd.read_csv(args.clinical, sep="\t")
    sub_col = [c for c in df.columns if c.lower().endswith("submitter_id")][0]
    figo_col = [c for c in df.columns if c.lower().endswith("figo_stage")][0]

    clinical = (df[[sub_col, figo_col]]
                .rename(columns={sub_col:"submitter_id", figo_col:"figo_raw"}))
    clinical["stage_text"] = clinical["figo_raw"].apply(normalize_figo)
    clinical = clinical[clinical["stage_text"].isin(["Stage I","Stage II","Stage III","Stage IV"])]
    clinical = clinical.drop_duplicates(subset=["submitter_id"])

    stage_to_idx = {"Stage I":0,"Stage II":1,"Stage III":2,"Stage IV":3}
    clinical["stage_idx"] = clinical["stage_text"].map(stage_to_idx)
    clinical_map = {r["submitter_id"]: (r["stage_text"], int(r["stage_idx"])) 
                    for _, r in clinical.iterrows()}

    print(f"   Valid staged patients: {len(clinical_map)}")

    print("🔎 Extracting PNGs from DICOMs (this may take a while on first run)...")
    saved = extract_pngs(
        clinical_map=clinical_map,
        tcga_root=args.tcga_root,
        out_dir=args.out_dir,
        img_size=args.img_size,
        slices_per_series=args.slices_per_series,
        min_slices=args.min_slices
    )
    if len(saved) == 0:
        print("❌ No images saved. Check TCGA-OV path and clinical mapping.")
        return

    print(f"   Saved {len(saved)} PNG slices to: {args.out_dir}")

    # Train/val split stratified by label
    all_paths, all_labels = zip(*saved)
    all_paths, all_labels = list(all_paths), list(all_labels)

    # group by submitter so slices from same patient don’t leak across splits
    patient_ids = [os.path.basename(p).split("__")[0] for p in all_paths]
    patient_df = pd.DataFrame({"pid":patient_ids, "y":all_labels})
    unique_p = patient_df.drop_duplicates("pid")

    p_train, p_val = train_test_split(
        unique_p["pid"], test_size=args.val_split, stratify=unique_p["y"], random_state=args.seed
    )
    train_items, val_items = [], []
    for p, y, path in zip(patient_ids, all_labels, all_paths):
        (train_items if p in set(p_train) else 
         val_items if p in set(p_val) else train_items).append((path,y))

    print(f"   Train slices: {len(train_items)} | Val slices: {len(val_items)}")
    if len(train_items) == 0 or len(val_items) == 0:
        print("❌ Empty split. Reduce val_split or check data.")
        return

    # Datasets/Loaders
    train_ds = StageImageDataset(train_items, img_size=args.img_size)
    val_ds   = StageImageDataset(val_items,   img_size=args.img_size)

    # Class weights for imbalance
    y_train = [y for _,y in train_items]
    counts = Counter(y_train)
    print("   Train label counts:", dict(counts))
    weights = torch.tensor(
        [1.0/(counts[i] if counts[i]>0 else 1) for i in range(4)],
        dtype=torch.float32
    )
    sampler_weights = [1.0/(counts[y] if counts[y]>0 else 1) for y in y_train]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)

    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥  Device:", DEVICE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = make_model(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Training loop with early stop
    best_val = 0.0
    patience, bad = 6, 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct/total if total else 0
        train_loss = run_loss/max(total,1)

        # validate
        model.eval()
        v_correct, v_total = 0, 0
        v_probs, v_truth = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
                v_probs.extend(out.softmax(1).cpu().numpy())
                v_truth.extend(y.cpu().numpy())
        val_acc = v_correct/max(v_total,1)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        # save best
        if val_acc > best_val:
            best_val = val_acc; bad = 0
            torch.save(model.state_dict(), "models/ov_stage_resnet50_best.pth")
        else:
            bad += 1
            if bad >= patience:
                print("⏹ Early stopping.")
                break

    # Save final + label map
    torch.save(model.state_dict(), "models/ov_stage_resnet50_final.pth")
    with open("models/ov_stage_label_map.json","w") as f:
        json.dump({"0":"Stage I","1":"Stage II","2":"Stage III","3":"Stage IV"}, f)

    # Final evaluation
    model.load_state_dict(torch.load("models/ov_stage_resnet50_best.pth", map_location="cpu"))
    model = model.to(DEVICE).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print("\nClassification report (VAL):")
    print(classification_report(y_true, y_pred, target_names=["Stage I","Stage II","Stage III","Stage IV"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n✅ Saved:")
    print("  models/ov_stage_resnet50_best.pth")
    print("  models/ov_stage_resnet50_final.pth")
    print("  models/ov_stage_label_map.json")

if __name__ == "__main__":
    main()
