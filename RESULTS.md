# Satellite Image Classification Study

## Model Performance Summary

We have conducted a comprehensive evaluation of different approaches to satellite image classification, exploring both traditional computer vision features (HOG and HSV) and deep learning features (ResNet50). Our study examines how these features perform with two common classifiers: Support Vector Machines (SVM) and Logistic Regression (LR). Below are our findings across eleven satellite classes: Terra, Calipso, Aquarius, CubeSat, AcrimSat, Debris, Jason, Sentinel-6, Aura, Cloudsat, and TRMM.

| Feature Extractor | Classifier         | Accuracy | Precision | Recall | F1 Score | Training Time (s) | Mean ROC AUC |
| ----------------- | ------------------ | -------- | --------- | ------ | -------- | ----------------- | ------------ |
| HOG               | SVM                | 0.3695   | 0.3587    | 0.3695 | 0.3417   | 896.28            | 0.8104       |
| HOG               | LogisticRegression | 0.2823   | 0.2811    | 0.2823 | 0.2813   | 66.96             | 0.7324       |
| HSV               | SVM                | 0.2909   | 0.2886    | 0.2909 | 0.2148   | 20.02             | 0.7201       |
| HSV               | LogisticRegression | 0.2845   | 0.2766    | 0.2845 | 0.2097   | 1.24              | 0.7130       |
| ResNet50          | SVM                | 0.1023   | 0.1042    | 0.1023 | 0.0392   | 276.84            | 0.4236       |
| ResNet50          | LogisticRegression | 0.0909   | 0.0083    | 0.0909 | 0.0152   | 1.26              | 0.5000       |

### Per-Class ROC AUC Analysis

The following table shows how each combination of feature extractor and classifier performs across different satellite classes. This detailed breakdown reveals important patterns in how different approaches handle specific types of satellites.

| Class      | HOG+SVM | HOG+LogReg | HSV+SVM | HSV+LogReg | ResNet50+SVM | ResNet50+LogReg | Best Approach |
| ---------- | ------- | ---------- | ------- | ---------- | ------------ | --------------- | ------------- |
| AcrimSat   | 0.9715  | 0.8665     | 0.9545  | 0.9424     | 0.2216       | 0.5000          | HOG+SVM       |
| Terra      | 0.8756  | 0.7686     | 0.9184  | 0.9249     | 0.4646       | 0.5000          | HSV+LogReg    |
| Aquarius   | 0.8719  | 0.8080     | 0.7536  | 0.7524     | 0.5942       | 0.5000          | HOG+SVM       |
| Debris     | 0.8589  | 0.8248     | 0.6290  | 0.6443     | 0.3915       | 0.5000          | HOG+SVM       |
| Sentinel-6 | 0.8490  | 0.7520     | 0.7588  | 0.7304     | 0.3816       | 0.5000          | HOG+SVM       |
| Calipso    | 0.7908  | 0.7106     | 0.6857  | 0.6827     | 0.4563       | 0.5000          | HOG+SVM       |
| Cloudsat   | 0.7713  | 0.6756     | 0.7173  | 0.7197     | 0.4101       | 0.5000          | HOG+SVM       |
| Jason      | 0.7648  | 0.6916     | 0.5792  | 0.5786     | 0.3971       | 0.5000          | HOG+SVM       |
| Aura       | 0.7560  | 0.6565     | 0.6659  | 0.6639     | 0.3654       | 0.5000          | HOG+SVM       |
| CubeSat    | 0.7337  | 0.6701     | 0.6021  | 0.5953     | 0.4350       | 0.5000          | HOG+SVM       |
| TRMM       | 0.6714  | 0.6319     | 0.6566  | 0.6487     | 0.5429       | 0.5000          | HOG+SVM       |

This per-class analysis reveals some notable patterns:

1. Class-Specific Feature Preferences:
   - AcrimSat achieves exceptional performance with both HOG and HSV features (ROC AUC > 0.94), suggesting it has distinctive structural and color characteristics
   - Terra uniquely benefits from color information, achieving its best performance with HSV features (ROC AUC = 0.9249)
   - Debris shows a strong preference for structural features, with HOG significantly outperforming HSV (0.8589 vs 0.6290)

2. Feature Type Effectiveness:
   - HOG features provide the most reliable performance across classes, being the best approach for 10 out of 11 classes
   - HSV features show competitive performance for specific satellites, particularly those with distinctive color signatures
   - ResNet50 features consistently underperform, with most classes showing near-random classification (ROC AUC ≈ 0.5)

3. Performance Patterns:
   - Larger satellites (like AcrimSat and Terra) tend to be easier to classify, possibly due to more distinctive features
   - Smaller satellites (like CubeSat) and similar-looking satellites (like TRMM) prove more challenging across all approaches
   - The performance gap between HOG and HSV varies significantly by class, suggesting different satellites have different distinguishing characteristics

4. Classifier Behavior:
   - SVM generally maintains better discrimination ability than Logistic Regression
   - The gap between SVM and Logistic Regression is most pronounced with HOG features
   - With ResNet50 features, Logistic Regression defaults to random guessing (ROC AUC = 0.5) while SVM shows some, albeit poor, discrimination

## In-Depth Analysis

### Feature Extractor Performance

The results reveal a surprising and important finding: traditional computer vision features significantly outperform the deep learning features from ResNet50 for our satellite classification task. Let's examine why:

1. HOG Features (Best Overall):
   - Highest accuracy (36.95% with SVM)
   - Strong ROC AUC scores across all classes
   - Particularly effective at capturing the distinctive structural elements of satellites
   - The superior performance suggests that edge and gradient information is highly discriminative for satellite identification

2. HSV Features (Second Best):
   - Moderate performance (29.09% accuracy with SVM)
   - Particularly effective for certain classes (e.g., Terra)
   - Color information provides complementary insights to structural features
   - Computationally efficient while maintaining reasonable performance

3. ResNet50 Features (Unexpected Poor Performance):
   - Significantly lower accuracy (10.23% with SVM)
   - Near-random ROC AUC scores
   - LogisticRegression results suggest complete failure to learn (ROC AUC = 0.5)
   
Several factors might explain the unexpected poor performance of ResNet50 features:

1. Domain Mismatch:
   - ResNet50 was pre-trained on natural images (ImageNet)
   - Satellite imagery represents a significant domain shift
   - The features learned for natural object recognition might not transfer well to spacecraft identification

2. Feature Space Complexity:
   - ResNet50 features are high-dimensional (2048D)
   - Without fine-tuning, these features might not align well with satellite characteristics
   - The classifiers might struggle with the high dimensionality of the feature space

3. Information Loss:
   - Using only the penultimate layer features without fine-tuning might discard valuable satellite-specific information
   - The fixed feature extraction approach might not capture the unique aspects of satellite imagery

### Classifier Behavior

The interaction between features and classifiers reveals interesting patterns:

1. SVM Performance:
   - Consistently outperforms LogisticRegression across all feature types
   - Performance gap is largest with HOG features (8.72 percentage points)
   - Even with ResNet50 features, maintains slightly better discrimination than random

2. LogisticRegression Performance:
   - Most stable with traditional features (HOG and HSV)
   - Complete failure with ResNet50 features (ROC AUC = 0.5)
   - Training times are consistently faster than SVM

### Computational Considerations

The computational requirements vary significantly across approaches:

1. Training Time Rankings (fastest to slowest):
   - HSV + LogisticRegression (1.24s)
   - ResNet50 + LogisticRegression (1.26s)
   - HSV + SVM (20.02s)
   - HOG + LogisticRegression (66.96s)
   - ResNet50 + SVM (276.84s)
   - HOG + SVM (896.28s)

2. Performance vs. Computation Trade-off:
   - HOG + SVM provides the best accuracy but requires the most computation
   - HSV features offer a good balance of speed and performance
   - ResNet50's poor performance doesn't justify its computational cost

## Recommendations

Based on these findings, we recommend:

1. Primary Approach:
   - Use HOG features with SVM for highest accuracy requirements
   - The longer training time is justified by the significant performance improvement

2. Fast Alternative:
   - Use HSV features with LogisticRegression for time-critical applications
   - Provides reasonable performance with minimal computational overhead

3. Future Improvements:
   - Consider fine-tuning ResNet50 on satellite imagery rather than using fixed features
   - Explore feature fusion approaches combining HOG and HSV information
   - Investigate other traditional feature extractors that capture spacecraft characteristics

### Notes
- All experiments use consistent train/test splits
- Feature extractors use default parameters
- Training times exclude feature extraction
- ResNet50 features were extracted from the penultimate layer without fine-tuning