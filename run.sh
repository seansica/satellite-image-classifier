# LOGISTIC REGRESSION
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/ \
#   --output-path .test_results/2024_12_07_hog_logistic_224_224_ratio_01 \
#   --feature-extractor hog \
#   --models logistic \
#   --train-ratio 0.05 \
#   --test-ratio 0.1 \
#   --image-size 224 224 \
#   --log-level DEBUG

# RANDOM FOREST
python -m app.cli \
  --device mps \
  --data-path ../ICIP-2021/ \
  --output-path .test_results/2024_12_07_hog_rf_224_224_ratio_01 \
  --feature-extractor hog \
  --models rf \
  --train-ratio 0.05 \
  --test-ratio 0.1 \
  --image-size 224 224 \
  --log-level DEBUG \
  --rf-n-estimators 5 \
  --rf-max-depth 2 \
  --rf-hidden-dim 32 \
