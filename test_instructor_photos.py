# Tests every saved model on the instructor's photos, with no training, and keeps the answer for every photo.
# Usage: python test_instructor_photos.py <models folder> <photos parent folder> <output json>
# Folders named "rock test photos", "paper test photos" and "scissor test photos" hold the labels.
import hashlib
import json
import os
import sys

import numpy as np
import pillow_heif
import tensorflow as tf
from PIL import Image, ImageOps
from scipy.stats import binomtest

pillow_heif.register_heif_opener()
models_dir, photos_dir, out_path = sys.argv[1:4]
CLASS_NAMES = ['rock', 'paper', 'scissors']
FOLDERS = ['rock test photos', 'paper test photos', 'scissor test photos']
MODEL_FILES = ['final_rps_model.keras', 'small_set_model.keras', 'small_shadow_model.keras',
               'final_rps_model_own.keras', 'small_set_model_own.keras', 'small_shadow_model_own.keras']
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.heic')

images, labels, seen, skipped = [], [], set(), 0
for label, folder in enumerate(FOLDERS):
    for root, _, names in os.walk(os.path.join(photos_dir, folder)):
        for name in sorted(names):
            if not name.lower().endswith(IMAGE_EXT):
                continue
            data = open(os.path.join(root, name), 'rb').read()
            digest = hashlib.md5(data).hexdigest()
            if digest in seen:                               # the same photo uploaded twice counts once
                skipped += 1
                continue
            seen.add(digest)
            img = ImageOps.exif_transpose(Image.open(os.path.join(root, name))).convert('RGB')
            images.append(np.array(img.resize((300, 300))))  # same preparation as the Colab cell
            labels.append(label)
x, y = np.stack(images), np.array(labels)
print(f"{len(y)} photos {dict(zip(CLASS_NAMES, np.bincount(y, minlength=3).tolist()))}, duplicates skipped: {skipped}")

results, predictions = {}, {}
print(f"{'model':<32}{'correct':>16}{'rock':>9}{'paper':>9}{'scissors':>10}   predicted as")
for name in MODEL_FILES:
    model = tf.keras.models.load_model(os.path.join(models_dir, name))
    probs = model.predict(x, batch_size=16, verbose=0)
    pred = probs.argmax(1)
    per_class = [int(((pred == c) & (y == c)).sum()) for c in range(3)]
    counts = np.bincount(pred, minlength=3).tolist()
    correct = int((pred == y).sum())
    results[name] = {'correct': correct, 'total': len(y), 'accuracy': correct / len(y),
                     'per_class': per_class, 'predicted': counts,
                     'mean_confidence': float(probs.max(1).mean())}
    predictions[name] = pred
    print(f"{name:<32}{f'{correct}/{len(y)} ({correct / len(y):.1%})':>16}{per_class[0]:>9}{per_class[1]:>9}{per_class[2]:>10}   {counts}")

pairs = [('final_rps_model.keras', 'small_shadow_model.keras'), ('final_rps_model.keras', 'final_rps_model_own.keras'),
         ('small_set_model.keras', 'small_set_model_own.keras'), ('small_shadow_model.keras', 'small_shadow_model_own.keras'),
         ('final_rps_model.keras', 'small_shadow_model_own.keras')]
paired = {}
print("\nPaired comparison (photos only one of the two models gets right; sign test)")
for a, b in pairs:
    ra, rb = predictions[a] == y, predictions[b] == y
    k, m = int((ra & ~rb).sum()), int((rb & ~ra).sum())
    p = float(binomtest(k, k + m).pvalue) if k + m else 1.0
    paired[f'{a} vs {b}'] = {'only_first': k, 'only_second': m, 'p': p}
    print(f"  {a} {k} vs {b} {m} | p = {p:.3f}")

json.dump({'photos': len(y), 'per_class': np.bincount(y, minlength=3).tolist(), 'duplicates_skipped': skipped,
           'results': results, 'paired': paired}, open(out_path, 'w'), indent=1)
print('saved', out_path)
