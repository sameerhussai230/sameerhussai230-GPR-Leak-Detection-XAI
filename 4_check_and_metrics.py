
import os
import re
import importlib
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')
CLASSES = ['cavity_leak', 'intact_soil']
SPLITS = ['train', 'val', 'test']
AUG_PATTERN = re.compile(r'_aug_\d+')


"""

1) Split-integrity check: do augmented copies of the same source scan
   (file names like 'cavity_leak_41_aug_8.png') appear in more than one split?
2) Held-out test metrics in deterministic eval mode (no MC Dropout):
   confusion matrix, accuracy, precision / recall / F1 for 'cavity_leak',
   and Expected Calibration Error (ECE).
"""
def source_key(cls, filename):
    """Collapse augmented copies of one scan into a single source key."""
    stem = os.path.splitext(filename)[0]
    return f"{cls}/{AUG_PATTERN.split(stem)[0]}"


def leakage_check(cfg):
    base = cfg.paths.processed_data
    keys = {s: defaultdict(int) for s in SPLITS}
    total = matched = 0

    for split in SPLITS:
        for cls in CLASSES:
            d = os.path.join(base, split, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.lower().endswith(IMG_EXT):
                    continue
                total += 1
                matched += bool(AUG_PATTERN.search(f))
                keys[split][source_key(cls, f)] += 1

    print("\n=== 1) SPLIT-INTEGRITY CHECK ===")
    print(f"Images found: {total} | with '_aug_<n>' in name: {matched}")
    if matched == 0:
        print("[NOTE] No '_aug_' pattern found, so this check cannot detect shared sources. "
              "Inspect file names manually.")
    for s in SPLITS:
        print(f"{s:5s}: {sum(keys[s].values())} images from {len(keys[s])} source scans")
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        shared = set(keys[a]) & set(keys[b])
        print(f"Source scans present in BOTH {a} and {b}: {len(shared)}")

    n_test = sum(keys['test'].values())
    n_leaky = sum(c for k, c in keys['test'].items() if k in keys['train'])
    pct = 100 * n_leaky / max(n_test, 1)
    print(f"Test images whose source scan also appears in train: {n_leaky}/{n_test} ({pct:.1f}%)")


def expected_calibration_error(confidences, correct, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return ece


def test_metrics(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_module = importlib.import_module("2_train")

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, 'best_gpr_model.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = train_module.GPRClassifier(cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['state_dict'])
    model.to(device).eval()  # deterministic: dropout OFF

    dm = train_module.GPRDataModule(cfg.paths.processed_data, cfg.data.batch_size, cfg.data.img_size)
    dm.setup()
    class_to_idx = dm.test_ds.class_to_idx
    cav = class_to_idx['cavity_leak']

    probs, ys = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    preds = probs.argmax(axis=1)

    y_pos = (ys == cav)
    p_pos = (preds == cav)
    prec, rec, f1, _ = precision_recall_fscore_support(y_pos, p_pos, average='binary', zero_division=0)
    acc = accuracy_score(ys, preds)
    cm = confusion_matrix(y_pos, p_pos, labels=[True, False])
    ece = expected_calibration_error(probs.max(axis=1), (preds == ys).astype(float))

    print("\n=== 2) HELD-OUT TEST METRICS (positive class = cavity_leak) ===")
    print(f"class_to_idx: {class_to_idx}")
    print(f"Test images: {len(ys)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ECE (10 bins): {ece:.4f}")
    print("Confusion matrix (rows = true, cols = predicted; order: cavity_leak, intact_soil)")
    print(cm)
    print(f"Mean max-softmax confidence: {probs.max(axis=1).mean():.4f} "
          f"(label smoothing 0.15 pushes training targets toward ~0.925 for 2 classes)")


def main():
    cfg = OmegaConf.load("config.yaml")
    leakage_check(cfg)
    test_metrics(cfg)


if __name__ == "__main__":
    main()