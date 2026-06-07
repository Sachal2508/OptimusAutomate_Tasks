"""
Task 1: Image Classification with CNN-style Pipeline
CodeAlpha AI Internship
================================================
Dataset   : Scikit-learn Digits (1797 samples, 10 classes, 8×8 grayscale)
Model     : Convolutional feature extraction (manual sliding-window patches)
            + MLP Classifier (sklearn) — CNN-equivalent without TensorFlow
Augment   : Gaussian noise, small rotations, shift jitter
Evaluate  : Confusion matrix, per-class accuracy, training loss curve,
            sample predictions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.ndimage import rotate, shift

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

print("=" * 60)
print("  Task 1: CNN-style Image Classification — Digits Dataset")
print("=" * 60)

# ── 1. Load Data ────────────────────────────────────────────────
digits = load_digits()
X_raw = digits.images   # (1797, 8, 8)
y     = digits.target

print(f"Dataset: {X_raw.shape[0]} samples | {len(np.unique(y))} classes | {X_raw.shape[1]}×{X_raw.shape[2]} images")

# ── 2. CNN-style Feature Extraction ─────────────────────────────
def extract_conv_features(images):
    """
    Simulate convolutional feature extraction:
    - 3 edge-detection kernels (horizontal, vertical, diagonal)
    - Applied as sliding window (stride=1) over 8×8 image
    - Output: flattened feature map per image
    """
    kernels = [
        np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),  # horizontal edge
        np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),  # vertical edge
        np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),  # diagonal edge
    ]
    features_all = []
    for img in images:
        img_norm = img / 16.0
        maps = [img_norm.flatten()]  # raw pixels
        for k in kernels:
            # Manual 2D convolution (valid padding)
            h, w = img_norm.shape
            kh, kw = k.shape
            out = np.zeros((h - kh + 1, w - kw + 1))
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j] = np.sum(img_norm[i:i+kh, j:j+kw] * k)
            # ReLU activation
            out = np.maximum(0, out)
            # Max pooling (2×2)
            ph, pw = out.shape[0] // 2, out.shape[1] // 2
            pooled = out[:ph*2, :pw*2].reshape(ph, 2, pw, 2).max(axis=(1, 3))
            maps.append(pooled.flatten())
        features_all.append(np.concatenate(maps))
    return np.array(features_all)

print("\nExtracting convolutional features...")
X_features = extract_conv_features(X_raw)
print(f"Feature vector size: {X_features.shape[1]} per image")

# ── 3. Data Augmentation ─────────────────────────────────────────
def augment_dataset(images, labels, n_augments=3):
    """Apply noise, rotation, and shift augmentations."""
    aug_imgs, aug_labels = [images.copy()], [labels.copy()]
    for _ in range(n_augments):
        batch = []
        for img in images:
            r = np.random.choice(['noise', 'rotate', 'shift'])
            if r == 'noise':
                noisy = img + np.random.normal(0, 0.5, img.shape)
                batch.append(np.clip(noisy, 0, 16))
            elif r == 'rotate':
                angle = np.random.uniform(-12, 12)
                batch.append(rotate(img, angle, reshape=False))
            else:
                s = np.random.uniform(-0.8, 0.8, size=2)
                batch.append(shift(img, s))
        aug_imgs.append(np.array(batch))
        aug_labels.append(labels.copy())
    return np.vstack(aug_imgs), np.concatenate(aug_labels)

# Train/test split first, then augment only train
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)

print("Augmenting training data...")
X_tr_aug, y_train_aug = augment_dataset(X_tr_raw, y_train, n_augments=4)
print(f"Train set after augmentation: {X_tr_aug.shape[0]} samples")

# Extract features from augmented train + original test
X_train = extract_conv_features(X_tr_aug)
X_test  = extract_conv_features(X_te_raw)

# ── 4. Build CNN-equivalent MLP Pipeline ─────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3-layer deep network
        activation='relu',
        solver='adam',
        alpha=0.001,               # L2 regularization (like Dropout)
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    ))
])

print("\nTraining MLP on conv features...")
pipeline.fit(X_train, y_train_aug)

# ── 5. Evaluate ───────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
class_names = [str(i) for i in range(10)]

print(f"\n{'='*45}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"{'='*45}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 6. Plots ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def styled_ax(ax):
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.tick_params(colors='#8b949e')
    return ax

mlp = pipeline.named_steps['mlp']

# Loss curve
ax1 = styled_ax(fig.add_subplot(gs[0, 0]))
ax1.plot(mlp.loss_curve_, color='#58a6ff', lw=2.5, label='Train Loss')
if mlp.validation_scores_:
    val_loss = [1 - s for s in mlp.validation_scores_]
    ax1.plot(val_loss, color='#f85149', lw=2.5, label='Val Loss (1-acc)')
ax1.set_title('Training Loss Curve', color='white', fontsize=12, fontweight='bold')
ax1.set_xlabel('Iteration', color='#8b949e'); ax1.set_ylabel('Loss', color='#8b949e')
ax1.legend(facecolor='#21262d', labelcolor='white')

# Validation accuracy curve
ax2 = styled_ax(fig.add_subplot(gs[0, 1]))
if mlp.validation_scores_:
    ax2.plot([s*100 for s in mlp.validation_scores_], color='#3fb950', lw=2.5)
    ax2.fill_between(range(len(mlp.validation_scores_)),
                     [s*100 for s in mlp.validation_scores_], alpha=0.15, color='#3fb950')
ax2.axhline(y=acc*100, color='#e3b341', linestyle='--', lw=1.5, label=f'Test: {acc*100:.1f}%')
ax2.set_title('Validation Accuracy', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Iteration', color='#8b949e'); ax2.set_ylabel('Accuracy (%)', color='#8b949e')
ax2.legend(facecolor='#21262d', labelcolor='white')

# Confusion matrix
ax3 = fig.add_subplot(gs[0, 2])
cm  = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='#0d1117')
ax3.set_title('Confusion Matrix', color='white', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted', color='#8b949e'); ax3.set_ylabel('True Label', color='#8b949e')
ax3.tick_params(colors='#8b949e')

# Per-class accuracy
ax4 = styled_ax(fig.add_subplot(gs[1, 0:2]))
pca  = cm.diagonal() / cm.sum(axis=1)
clrs = [('#3fb950' if v >= 0.95 else '#e3b341' if v >= 0.90 else '#f85149') for v in pca]
bars = ax4.bar(class_names, pca * 100, color=clrs, edgecolor='#0d1117')
ax4.axhline(y=acc*100, color='#58a6ff', linestyle='--', lw=1.5, label=f'Overall: {acc*100:.1f}%')
ax4.set_title('Per-Class Accuracy', color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('Digit Class', color='#8b949e'); ax4.set_ylabel('Accuracy (%)', color='#8b949e')
ax4.legend(facecolor='#21262d', labelcolor='white'); ax4.set_ylim(50, 105)
for bar, val in zip(bars, pca):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.0f}%', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

# Sample predictions (4×4 grid)
inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[1, 2], hspace=0.3, wspace=0.1)
sidx  = np.random.choice(len(X_te_raw), 16, replace=False)
for i, idx in enumerate(sidx):
    ax = fig.add_subplot(inner[i // 4, i % 4])
    ax.imshow(X_te_raw[idx], cmap='Blues_r')
    correct = y_pred[idx] == y_test[idx]
    ax.set_title(f'P:{y_pred[idx]} T:{y_test[idx]}',
                 color='#3fb950' if correct else '#f85149', fontsize=6, pad=1)
    ax.axis('off')

fig.text(0.5, 0.98,
         f'CNN-style Digit Classifier  |  Test Accuracy: {acc*100:.2f}%  |  Iterations: {mlp.n_iter_}',
         ha='center', va='top', color='white', fontsize=14, fontweight='bold')

plt.savefig('evaluation_report.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("\n✓  Saved: evaluation_report.png")
print("✓  Task 1 Complete!")