# 🐾 Pet Breed Classifier with an Honest Evaluation

Fine-tuned image classifier for the 37 cat and dog breeds of the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), evaluated on the official test set and on 54 web images collected separately, to check how well it generalises to photos from a different source.

| Metric | Result |
|---|---|
| Test accuracy (3,669 images, 37 breeds) | 94.4% |
| Test top-5 accuracy | 99.7% |
| Accuracy on my own web images (54 images, 5 breeds) | 88.9% (95% CI 78–95%) |
| Test accuracy on the same 5 breeds | 87.6% |

---

## How to run

The whole project is one notebook: `pet_classifier.ipynb`.

1. Open the notebook in [Google Colab](https://colab.research.google.com) (File → Upload notebook, or open it from GitHub).
2. Select a GPU: Runtime → Change runtime type → T4 GPU.
3. Run Cells 1–15 in order. The dataset (~800 MB) downloads automatically through `torchvision`. Training takes about 10–15 minutes.
4. For the generalisation check: zip the `my_images` folder from this repository, upload `my_images.zip` in Colab's Files panel, then run Cells 16–20.

All charts and metrics are saved to the `results/` folder. The trained weights (`best_model.pth`, ~110 MB) are not included because of GitHub's file-size limit; running the notebook recreates them.

To run locally instead: `pip install -r requirements.txt` and open the notebook in Jupyter or VS Code (a GPU is strongly recommended).

---

## Data split

| Split | Images | Source | Used for |
|---|---|---|---|
| Train | 3,128 | 85% of the official `trainval` split | Learning the weights |
| Validation | 552 | 15% of the official `trainval` split | Choosing the best epoch |
| Test | 3,669 | The official `test` split | Final evaluation only, used once |

- The test set is never used for any decision. It is evaluated exactly once, after training, so the reported score isn't biased by model selection.
- Stratified split: every breed keeps the same share in train and validation, so no breed is missing or under-represented in validation.
- Reproducible: the random seed is fixed (42) and the exact indices are saved in `results/split_indices.json`.

---

## Model and training choices

| Choice | What and why |
|---|---|
| Model | ConvNeXt-Tiny pretrained on ImageNet (torchvision). Strong on fine-grained images and small enough to fine-tune in minutes on a free Colab GPU. |
| New output layer | The 1,000-class ImageNet layer is replaced with a 37-class layer. |
| Stage 1 (2 epochs) | Backbone frozen, only the new layer trained (lr 1e-3). The new layer starts random, so this avoids damaging the pretrained features with large early gradients. |
| Stage 2 (6 epochs) | Whole network fine-tuned with AdamW: backbone lr 1e-4, head lr 1e-3, weight decay 0.05, cosine learning-rate schedule. |
| Augmentation | Random resized crop (scale 0.7–1.0), horizontal flip, mild colour jitter: teaches the breed rather than one exact photo. |
| Label smoothing (0.1) | Reduces over-confidence, which helps on unfamiliar images. |
| Mixed precision | About 2x faster training on GPU with no accuracy loss. |
| Checkpointing | The model with the best validation accuracy is kept, not the last epoch. |

Training curves (`results/training_curves.png`): the head-only stage already reached 94.9% validation accuracy, showing how strong the pretrained features are; full fine-tuning raised it to 96.9%. Training accuracy reached ~100% while validation levelled off around 97%, a small overfitting gap, but validation loss stayed flat rather than rising.

---

## Results on the test set

- Accuracy: 94.4%, top-5 accuracy: 99.7%.
- Test accuracy is 2.5 points below the best validation accuracy (96.9%). That's expected: validation was used to choose the checkpoint, which makes it slightly optimistic, and the validation set is small (552 images).
- Per-class precision and recall: `results/per_class_metrics.csv`. Confusion matrix: `results/confusion_matrix.png`.

### The three breeds the model confuses most

| Breed | Wrong (of 100) | Recall | Mostly mistaken for |
|---|---|---|---|
| American Pit Bull Terrier | 39 | 0.61 | Staffordshire Bull Terrier (27 images) |
| Ragdoll | 22 | 0.78 | Birman (14 images) |
| British Shorthair | 18 | 0.82 | Bombay |

Other frequent confusions: Egyptian Mau → Bengal (8), American Pit Bull Terrier → American Bulldog (8), Staffordshire Bull Terrier → American Pit Bull Terrier (7).

These are mostly genuinely similar-looking breeds: the "bully" dog breeds share head shape, muscular build, and coat colours; Ragdolls and Birmans are both long-haired, colour-pointed cats with blue eyes; Egyptian Maus and Bengals both have spotted coats. Even people often need context (pedigree, size, or a closer look at specific features) to tell them apart.

---

## Generalisation check: my own images

I collected 54 web images of 5 breeds (Abyssinian, American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle), separate from the dataset. They are in `my_images/`, one folder per breed. I included the hard pair **American Bulldog / American Pit Bull Terrier** on purpose, alongside easier breeds.

| | Accuracy |
|---|---|
| Test set, all 37 breeds | 94.4% |
| Test set, the same 5 breeds (fair comparison) | 87.6% |
| My web images | 88.9% (95% CI 78–95%) |
| My web images, top-5 | 98.1% |

| Breed | Correct on my images |
|---|---|
| Abyssinian | 8 / 8 |
| American Pit Bull Terrier | 11 / 11 |
| Basset Hound | 12 / 12 |
| Beagle | 11 / 11 |
| American Bulldog | 6 / 12 (every error predicted as American Pit Bull Terrier) |

What this shows:
- Comparing against the right baseline matters. The headline test accuracy (94.4%) is the wrong comparison, because my photos include the hardest breed. Against the test accuracy on the same five breeds (87.6%), accuracy on new images (88.9%) shows no clear drop.
- The sample is small. With 54 images, the 95% confidence range is 78–95%, so a drop of several points could be hidden. The result supports "generalises reasonably well", not "generalises perfectly".
- The model is less sure on new images: average confidence fell from 0.84 on the test set to 0.77 on my photos, a sign that web photos differ from the dataset (lighting, framing, image quality).
- All 6 errors are one confusion: American Bulldog → American Pit Bull Terrier. The easy breeds were 100% correct, so the weakness is the known look-alike pair, not new images in general.

Full predictions: `results/own_images_predictions.csv` and `results/own_images_grid.png`.

---

## Failure case

![Failure case](results/failure_case.png)

What happened: a brown American Bulldog with a white chest was predicted as an American Pit Bull Terrier with 90% confidence.

What I think caused it:
1. Appearance overlap. This dog has a brown coat, white chest, lean athletic build, and folded ears, all very typical of pit bulls. In the dataset, many American Bulldogs are mostly white with a heavier build, so this dog sits closer to the pit-bull examples the model learned from.
2. The model relies on the face. Grad-CAM shows that the evidence for both breeds comes from the same muzzle and face region. The two breeds' heads are very similar, so the deciding features are subtle, and the model doesn't use body shape or size, which separate them more clearly.
3. Possible label noise in web images. "American Bulldog" photos online are sometimes mixed-breed dogs or pit-bull types. Some of the 6 "errors" may be partly a labelling problem, which is itself a lesson about evaluating with web data.

**What I would try to fix it:
- More varied training data for these breeds, especially American Bulldogs in brown, brindle, and mixed coats, plus hard-example mining that over-samples confusable pairs.
- **Use more of the body:** train with a less aggressive crop, or a higher input resolution, so build and proportions are visible.
- Flag uncertainty instead of guessing: when the top two classes are close or both belong to a known look-alike group, return "American Bulldog or American Pit Bull Terrier" rather than one confident wrong answer. Top-2 accuracy here is far higher than top-1.
- Calibrate confidence (for example temperature scaling on the validation set), so 90% confidence really means about 90% correct.
- Verify web labels with a second source before using them for evaluation.

---

## What I would improve next

1. A bigger, more diverse generalisation set: all 37 breeds, more images per breed, and phone photos rather than only web images, so the confidence interval becomes much narrower.
2. Targeted fixes for look-alike groups: hard-negative mining and more varied training images for the bully breeds, Ragdoll/Birman, and Egyptian Mau/Bengal.
3. Test-time augmentation: (averaging predictions over flipped and differently cropped versions of each image).
4. Confidence calibration and an "uncertain" output for low-confidence or look-alike predictions.
5. Compare architectures (for example ConvNeXt-Small, EfficientNetV2, or a ViT) at higher resolution.
6. Use the dataset's head bounding boxes to test whether cropping to the animal improves accuracy on cluttered web photos.

---

## Repository structure

```text
pet-classifier/
├── pet_classifier.ipynb      # full pipeline: data, training, evaluation, generalisation, failure analysis
├── my_images/                # 54 web images used for the generalisation check, one folder per breed
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.csv
│   ├── test_mistakes.png
│   ├── own_images_grid.png
│   ├── own_images_predictions.csv
│   ├── failure_case.png
│   ├── split_indices.json
│   ├── history.json
│   └── summary.json
├── requirements.txt
└── README.md
```
