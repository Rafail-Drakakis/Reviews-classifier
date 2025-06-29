# Reviews-classifier

This repository contains a small project for sentiment classification of IMDb movie reviews.  It includes
experiments with classical machine learning algorithms and simple deep learning
models built with PyTorch.  A short project report is available in the
`report/` directory.

## Contents

- `main.py` &ndash; downloads the IMDb dataset, preprocesses the text and trains
  several models (logistic regression, Naive Bayes, linear SVM, LSTM and CNN).
- `predict.py` &ndash; loads a pre-trained CNN model and predicts the sentiment of
  example sentences.
- `model.pt` and `vocab.pkl` &ndash; weights and vocabulary for the CNN model
  produced by `main.py` (included for convenience).
- `report/` &ndash; LaTeX sources and the final PDF report describing the
  methodology and results.

## Requirements

The code was tested with Python 3 and requires the following packages:

```
nltk
scikit-learn
torch
matplotlib
seaborn
gdown
```

Install them with `pip`:

```
pip install nltk scikit-learn torch matplotlib seaborn gdown
```

The first run will also download the required NLTK data sets (`punkt` and
`stopwords`).

## Training

To train the models from scratch, simply run:

```
python main.py
```

The script automatically downloads and extracts the IMDb Large Movie Review
Dataset (around 80&nbsp;MB) from Google Drive.  Training produces a number of
confusion-matrix figures in a new `figures/` directory and saves the best CNN
model to `model.pt` together with the vocabulary file `vocab.pkl`.

## Prediction

After training (or using the included model weights) you can predict the
sentiment of a custom review via `predict.py`:

```
python predict.py
```

This script prints the sentiment label (positive/negative) and confidence for
some example sentences.  You can modify `predict.py` to pass your own text or
call the `predict_sentiment` function from your own code.

## Report

A full write-up of the project is provided in `report/report.pdf`.  The LaTeX
sources are also included for reference.
