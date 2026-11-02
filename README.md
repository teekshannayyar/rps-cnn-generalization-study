# Rock-Paper-Scissors with CNNs

**Personal project.** A case study in generalization, distribution shift and honest evaluation, using TensorFlow/Keras and MobileNetV2. It began as a course challenge, and I extended it with real-world tests, robustness experiments and a browser game.

The first models reached 100% on validation and 80–95% on the public test set, then recognised only 1–2 of 6 photos of a real hand. This project documents how that was diagnosed and fixed, including the experiments that failed.

**[Open the notebook](RPS_CNN_Generalization_Study.ipynb)**. It covers every step from first to last. Section 21 collects the timeline, the difficulties and how they were solved, how the training data changed, and every test set.

## Results

| Model | Public test (372) | Real photos, new people (99) | HaGRID, 900 people | Hard angles (3,871) | Instructor's test (262) |
|---|---:|---:|---:|---:|---:|
| 4,955 images | 100.0% | 97.0% | 95.0% | 91.7% | 89.7% |
| 2,400 images | 100.0% | 96.0% | 94.1% | 92.8% | 86.6% |
| 2,400 images + shadow training | 100.0% | 96.0% | 95.9% | 93.1% | 88.9% |
| **Final: 2,400 images + shadow training + 77 own photos** | **100.0%** | **99.0%** | **97.8%** | **94.8%** | **92.4%** |

These scores come from the saved model files. The notebook's own rerun gives slightly different numbers, shown and explained there. The final model had no same-session control, so part of its gain may be run-to-run variation.

## What made the difference

1. **Testing on real photos early.** The public test set looks like the training set, so it could not show that the models failed on real hands.
2. **Real data.** Adding real photos of many people raised accuracy on new people from about 68% to 97–99%. Augmenting computer-generated images could not do that.
3. **Measuring one change at a time.** This found the largest remaining weakness, shadows (about 15 points), which led to a targeted fix.
4. **Paired tests and control runs.** The same code trained twice can differ by several points, so single scores were not trusted.
5. **Thinking about the user of the model.** An input guard stops accuracy falling to 33% when someone scales the pixels first.

## Run it

Open the notebook in Google Colab, choose a T4 GPU, then Runtime → Run all. A full run takes about an hour and downloads about 1.5 GB of public data.

The final model was trained on private photos, which are not included. Without them the notebook rebuilds the shadow-trained model. Sections 8, 16 and 17 use those photos when `own_photos.zip` is uploaded, and otherwise report the recorded results.

## Files

| File | Purpose |
|---|---|
| `RPS_CNN_Generalization_Study.ipynb` | The full case study, with outputs |
| `test_instructor_photos.py` | Tests saved models on a folder of labelled photos, with no training |
| `instructor_results.json` | Results of that test for all six models |

## Tools

TensorFlow/Keras, MobileNetV2 transfer learning, `tf.data`, data augmentation, background compositing, scikit-learn, SciPy sign tests. A side project ran the model in the browser with ONNX Runtime Web and MediaPipe.

## Datasets

* [TensorFlow Datasets `rock_paper_scissors`](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
* [Giusti et al., rock-paper-scissors](https://github.com/alessandro-giusti/rock-paper-scissors)
* [Rock-Paper-Scissors Images (Kaggle, drgfreeman)](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
* [HaGRID](https://github.com/hukenovs/hagrid), through the [30k sample](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)
* [nicokofi/Rock-Paper-Scissors-Hand-Data](https://huggingface.co/datasets/nicokofi/Rock-Paper-Scissors-Hand-Data)
* [Imagenette](https://www.tensorflow.org/datasets/catalog/imagenette), used only for backgrounds

The instructor's test photos and my own photos are private and are not part of this repository.
