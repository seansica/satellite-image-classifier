# SVM + HSV
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/ \
#   --output-path .test_results/2024_12_07_hsv_svm_224_224_ratio_01 \
#   --feature-extractor hsv \
#   --models svm \
#   --train-ratio 0.05 \
#   --test-ratio 0.1 \
#   --image-size 224 224 \
#   --log-level DEBUG

# SVM + LBP
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/ \
#   --output-path .test_results/2024_12_07_lbp_svm_224_224_ratio_01 \
#   --feature-extractor lbp \
#   --models svm \
#   --train-ratio 0.05 \
#   --test-ratio 0.1 \
#   --image-size 224 224 \
#   --log-level DEBUG

# RANDOM FOREST + HOG
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/ \
#   --output-path .test_results/2024_12_07_hog_rf_224_224_ratio_01 \
#   --feature-extractor hog \
#   --models rf \
#   --train-ratio 0.05 \
#   --test-ratio 0.1 \
#   --image-size 224 224 \
#   --log-level DEBUG \
#   --rf-n-estimators 5 \
#   --rf-max-depth 2 \
#   --rf-hidden-dim 32 \

# GRADIENT BOOSTED MODEL + HOG
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/ \
#   --output-path .test_results/2024_12_07_hog_gbm_224_224_ratio_01 \
#   --feature-extractor hog \
#   --models gbm \
#   --train-ratio 0.05 \
#   --test-ratio 0.1 \
#   --image-size 224 224 \
#   --log-level DEBUG \
#   --rf-n-estimators 5 \
#   --rf-max-depth 2 \
#   --rf-hidden-dim 32 \

# SVM + RESNET50
python -m app.cli \
  --device mps \
  --data-path ../ICIP-2021/ \
  --output-path .test_results/2024_12_07_resnet50_svm_224_224_ratio_01 \
  --feature-extractor resnet50 \
  --models svm \
  --train-ratio 0.05 \
  --test-ratio 0.1 \
  --image-size 224 224 \
  --log-level DEBUG
