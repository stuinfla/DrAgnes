#!/usr/bin/env python3
"""Fast DrAgnes ViT training — direct PyTorch loop, no Trainer overhead."""

import torch
import torch.nn.functional as F
import time
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
print(f'PyTorch: {torch.__version__}')

# Load dataset
print('Loading dataset...')
t0 = time.time()
ds = load_dataset('kuchikihater/HAM10000')['train']
print(f'Loaded {len(ds)} images in {time.time()-t0:.1f}s')

# Split
labels = [ex['label'] for ex in ds]
train_idx, test_idx = train_test_split(range(len(ds)), test_size=0.15, stratify=labels, random_state=42)
print(f'Train: {len(train_idx)}, Test: {len(test_idx)}')

# Load model
print('Loading ViT model...')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=7,
    ignore_mismatched_sizes=True
).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


focal_loss = FocalLoss(
    alpha=[5.0, 4.0, 1.5, 6.0, 8.0, 0.3, 5.0],  # mel=8x, nv=0.3x
    gamma=2.0
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def process_batch(indices):
    images = []
    batch_labels = []
    for i in indices:
        ex = ds[int(i)]
        img = ex['image'].convert('RGB')
        images.append(img)
        batch_labels.append(ex['label'])
    inputs = processor(images=images, return_tensors='pt')
    return inputs['pixel_values'].to(device), torch.tensor(batch_labels, device=device)


print(f'\n{"="*60}')
print(f'  TRAINING: {EPOCHS} epochs, batch {BATCH_SIZE}, lr {LR}')
print(f'  Focal loss: mel=8.0x, nv=0.3x, gamma=2.0')
print(f'{"="*60}\n')

best_mel_sens = 0

for epoch in range(EPOCHS):
    model.train()
    np.random.shuffle(train_idx)
    total_loss = 0
    n_batches = 0
    t0 = time.time()

    for batch_start in range(0, len(train_idx), BATCH_SIZE):
        batch_idx = train_idx[batch_start:batch_start + BATCH_SIZE]
        if len(batch_idx) < 2:
            continue

        pixels, targets = process_batch(batch_idx)
        outputs = model(pixel_values=pixels)
        loss = focal_loss(outputs.logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 100 == 0:
            elapsed = time.time() - t0
            speed = n_batches * BATCH_SIZE / elapsed
            print(f'  Epoch {epoch+1} | Batch {n_batches}/{len(train_idx)//BATCH_SIZE} | '
                  f'Loss: {total_loss/n_batches:.4f} | {speed:.1f} img/s')

    # Evaluate
    model.eval()
    confusion = np.zeros((7, 7), dtype=int)
    with torch.no_grad():
        for batch_start in range(0, len(test_idx), 32):
            batch_idx = test_idx[batch_start:batch_start + 32]
            pixels, targets = process_batch(batch_idx)
            outputs = model(pixel_values=pixels)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            for pred, true in zip(preds, targets.cpu().numpy()):
                confusion[true][pred] += 1

    correct = confusion.diagonal().sum()
    total_test = confusion.sum()
    accuracy = correct / total_test

    mel_idx = 4
    mel_tp = confusion[mel_idx][mel_idx]
    mel_total = confusion[mel_idx].sum()
    mel_sens = mel_tp / mel_total if mel_total > 0 else 0

    # All cancer sensitivity (mel + bcc + akiec)
    cancer_tp = confusion[4][4] + confusion[1][1] + confusion[0][0]
    cancer_total = confusion[4].sum() + confusion[1].sum() + confusion[0].sum()
    cancer_sens = cancer_tp / cancer_total if cancer_total > 0 else 0

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  EPOCH {epoch+1}/{EPOCHS} — {elapsed:.0f}s')
    print(f'  Loss: {total_loss/n_batches:.4f}')
    print(f'  Accuracy: {accuracy:.1%} ({correct}/{total_test})')
    print(f'  *** MELANOMA SENSITIVITY: {mel_sens:.1%} ({mel_tp}/{mel_total}) ***')
    print(f'  *** ALL CANCER SENSITIVITY: {cancer_sens:.1%} ({cancer_tp}/{cancer_total}) ***')
    print(f'  Per-class:')
    for c in range(7):
        tp = confusion[c][c]
        fn = confusion[c].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        marker = ' <-- TARGET' if c == 4 else ''
        print(f'    {CLASS_NAMES[c]:8s}: {sens:6.1%} ({tp}/{tp+fn}){marker}')
    print(f'{"="*60}\n')

    if mel_sens > best_mel_sens:
        best_mel_sens = mel_sens
        model.save_pretrained('scripts/dragnes-classifier/best')
        processor.save_pretrained('scripts/dragnes-classifier/best')
        print(f'  >> SAVED best model (melanoma sens: {mel_sens:.1%})')

print(f'\n{"="*60}')
print(f'  TRAINING COMPLETE')
print(f'  Best melanoma sensitivity: {best_mel_sens:.1%}')
print(f'{"="*60}')
