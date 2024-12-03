# Satellite Image Classification Framework

This framework provides a modular, extensible system for evaluating machine learning models on satellite image classification tasks. While initially designed for the [SPARK 2021](https://cvi2.uni.lu/spark-2021-dataset/) space object detection dataset, its architecture allows it to be adapted for any computer vision classification task.

Please see [RESULTS.md](./RESULTS.md) for a breakdown of our experimental results.

## Overview

The framework implements a complete machine learning pipeline that handles everything from data loading to model evaluation. It was designed with several key principles in mind:

1. Modularity: Each component (feature extraction, model training, evaluation) is independent and can be easily replaced or extended.
2. Reproducibility: All random operations are seeded, and all parameters are explicitly tracked.
3. Comprehensive Evaluation: Models are evaluated using multiple metrics including accuracy, precision, recall, F1 score, and ROC curves.
4. Easy Experimentation: Command-line interface makes it simple to try different models and parameters.

```mermaid
flowchart TB
    subgraph Input
        D[Dataset Directory] --> DL[DatasetLoader]
        DL --> PP[Preprocessing]
    end

    subgraph Feature Extraction
        PP --> FE[Feature Extractors]
        FE --> |HOG Features| FM[Feature Matrix]
        FE -.-> |Future: HSV, LBP, etc.| FM
    end

    subgraph Model Training
        FM --> |Training Data| MT[Model Training]
        L[Labels] --> MT
        MT --> |SVM| M[Models]
        MT --> |Logistic Regression| M
        MT -.-> |Future: Neural Networks, etc.| M
    end

    subgraph Evaluation
        M --> E[Evaluation]
        E --> |Metrics| R[Results]
        E --> |Visualizations| V[Visualizations]
    end

    classDef future fill:#f9f,stroke:#333,stroke-dasharray: 5 5
    classDef current fill:#9f9,stroke:#333
    class D,DL,PP,FM,M,E,R,V current
    class HSV,LBP future
```

## Getting Started

### Installation

The project uses Poetry for dependency management. To get started:

```bash
# Clone the repository
git clone git@github.com:seansica/satellite-image-classifier.git
cd satellite-image-classifier

# Install dependencies
poetry install
```

### Basic Usage

The simplest way to run the classifier is:

```bash
poetry run python -m app.cli --data-path path/to/dataset
```

This will run the classification pipeline with default settings. For more control, you can specify additional parameters:

```bash
poetry run python -m app.cli \
    --data-path path/to/dataset \
    --output-path results \
    --feature-extractor hog \
    --models svm logistic \
    --samples-per-class 1000 \
    --image-size 224 224 \
    --log-level DEBUG
```

### Output

The framework generates comprehensive evaluation results in the specified output directory:

```
output_path/
├── plots/
│   ├── SVM_confusion_matrix.png
│   ├── SVM_roc_curves.png
│   ├── LogisticRegression_confusion_matrix.png
│   └── LogisticRegression_roc_curves.png
└── metrics/
    ├── SVM_metrics.json
    ├── SVM_feature_importance.csv
    ├── SVM_training_summary.json
    ├── LogisticRegression_metrics.json
    ├── LogisticRegression_feature_importance.csv
    ├── LogisticRegression_training_summary.json
    └── overall_comparison.csv
```

## Dataset Structure

The framework expects datasets to be organized in a directory structure where each subdirectory represents a class:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

While initially designed for the SPARK 2021 space object detection dataset, which includes classes like AcrimSat, Aquarius, Aura, etc., the framework can work with any image dataset organized in this manner.

## Features

- Multiple feature extraction methods (currently supports HOG)
- Multiple classification models (currently supports SVM and Logistic Regression)
- Automatic data splitting with stratification
- Comprehensive evaluation metrics
- Visualization of results including confusion matrices and ROC curves
- Detailed logging and error handling
- Easy extension points for new models and feature extractors

## Performance Considerations

When working with large datasets, consider:

1. The `samples-per-class` parameter to limit memory usage
2. Image resolution settings via `image-size`
3. Feature extraction method choice, as some methods may be more computationally intensive

## Limitations

- Currently supports only image classification tasks
- Images must be in a format readable by OpenCV (typically .jpg, .png)
- All images in a dataset must be accessible as files on disk

See [CONTRIBUTING.md](./CONTRIBUTING.md) for information on extending the framework with new models or feature extractors.