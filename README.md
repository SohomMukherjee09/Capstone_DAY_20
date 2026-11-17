# Day 20 Capstone — CIFAR-10 Transfer Learning

## Project

This capstone combines the complete Computer Vision workflow in one offline notebook. It trains a small CNN from scratch on a balanced CIFAR-10 subset, then compares it with MobileNetV2 transfer learning using frozen feature extraction and fine-tuning.

## Deliverables

```text
day20_capstone.ipynb
README_day20_capstone.md
```

After execution, the notebook also produces:

```text
day20_best.keras
day20_final_finetuned.keras
```

## Learning objectives

- Load CIFAR-10 from the local Keras cache without downloading data.
- Visualize images and confirm balanced class counts.
- Build a CNN baseline from scratch.
- Apply data augmentation.
- Use MobileNetV2 as a frozen feature extractor.
- Fine-tune the final 30 backbone layers with a low learning rate.
- Apply EarlyStopping and ModelCheckpoint.
- Evaluate with accuracy, a confusion matrix, and per-class precision/recall/F1.
- Interpret correct predictions using Grad-CAM.
- Write a numbers-backed baseline-versus-transfer reflection.

## Environment assumptions

The hosted JupyterLab environment must already contain:

- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn
- CIFAR-10 in `~/.keras/datasets`
- MobileNetV2 ImageNet weights in the local Keras model cache

The notebook performs no URL download. If CIFAR-10 or MobileNetV2 weights are missing and Keras attempts network access, stop and notify the instructor.

## Dataset

The notebook creates balanced subsets from CIFAR-10:

| Split | Images per class | Total images |
|---|---:|---:|
| Training | 500 | 5,000 |
| Test | 100 | 1,000 |

The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Models

### Baseline CNN

The baseline contains three convolutional blocks with BatchNormalization, pooling, global average pooling, Dropout, and a 10-class softmax output. It trains for at most 25 epochs with early stopping.

### Transfer learning: feature extraction

MobileNetV2 is loaded with ImageNet weights, `include_top=False`, and a 96×96 input. The backbone is frozen and only a new global-pooling, Dropout, and 10-class head is trained.

### Transfer learning: fine-tuning

The final 30 backbone layers are made trainable, while BatchNormalization layers remain frozen for stability. The learning rate is reduced from `1e-3` to `1e-5` before six additional epochs.

## Preprocessing

CIFAR-10 images are stored as float32 values in `[0,1]`. Inside the transfer model they are:

1. augmented during training;
2. resized from 32×32 to 96×96;
3. scaled to `[0,255]`; and
4. transformed by `mobilenet_v2.preprocess_input` to MobileNetV2's expected range.

Because preprocessing is inside the model, training and inference use the same transformation.

## Evaluation

The notebook reports:

- baseline test accuracy;
- feature-extraction test accuracy;
- fine-tuned test accuracy;
- absolute fine-tuned versus baseline lift;
- a 10×10 confusion matrix;
- per-class precision, recall, and F1;
- the largest directed class confusion; and
- Grad-CAM overlays for three correctly classified images.

## Honest interpretation

Expected ranges in the assignment are guidance, not guaranteed scores. CPU behaviour, TensorFlow version, random operations, and early stopping can change results. If fine-tuning performs worse than frozen feature extraction, report it honestly and discuss likely causes such as overfitting, excessive unfreezing, or unstable learning.

Grad-CAM is qualitative evidence. A heatmap focused on the object supports the prediction, while attention on sky, grass, or other background regions may indicate reliance on spurious cues.

## Step 8 reflection

Before submission, replace the reflection TODO with 4–6 original sentences that cite:

1. baseline and fine-tuned accuracy;
2. the exact absolute lift;
3. feature-extraction versus fine-tuning performance;
4. the worst-confused class pair;
5. what Grad-CAM highlighted; and
6. one concrete next experiment.

Do not leave the generated template unchanged.

## Run instructions

1. Open `day20_capstone.ipynb` in the hosted JupyterLab environment.
2. Select the Python 3 kernel.
3. Run all cells from top to bottom.
4. Confirm that all plots render.
5. Complete the Step 8 reflection using the measured results.
6. Save the notebook with outputs visible.
7. Submit `day20_capstone.ipynb`.

## Final checklist

- [ ] CIFAR-10 loads without network access.
- [ ] Training and test shapes are `(5000, 32, 32, 3)` and `(1000, 32, 32, 3)`.
- [ ] Every class has 500 training examples.
- [ ] Baseline CNN training completes.
- [ ] Augmentation preview renders.
- [ ] MobileNetV2 frozen feature extraction completes.
- [ ] Fine-tuning completes with `1e-5` learning rate.
- [ ] Confusion matrix and classification report render.
- [ ] Grad-CAM overlays render.
- [ ] Reflection contains real results and no TODO.
- [ ] Notebook is saved with all outputs visible.

## Git commands

```bash
git add day20_capstone.ipynb README_day20_capstone.md
git commit -m "Complete Day 20 transfer learning capstone"
git push
```

Model files are usually too large for normal Git. Add `*.keras` to `.gitignore` unless the instructor explicitly requires model artifacts or Git LFS is configured.
