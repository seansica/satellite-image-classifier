# FULL RUN:
# python -m app.cli \
#   --device mps \
#   --data-path ../ICIP-2021/train_rgb \
#   --output-path .test_results/hog_224_224_7500 \
#   --feature-extractor hog \
#   --models svm logistic rf gbm \
#   --samples-per-class 7500 \
#   --image-size 224 224 \
#   --log-level DEBUG

# TEST RUN:
python -m app.cli \
  --device mps \
  --data-path ../ICIP-2021/ \
  --output-path .test_results/2024_12_06_hog_224_224_1000 \
  --feature-extractor hog \
  --models svm \
  --samples-per-class 100 \
  --image-size 224 224 \
  --log-level DEBUG
