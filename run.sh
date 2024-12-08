python -m app.cli \
  --device mps \
  --data-path ../ICIP-2021/ \
  --feature-extractor resnet50 \
  --models svm \
  --train-ratio 0.05 \
  --test-ratio 0.1 \
  --image-size 224 224 \
  --log-level DEBUG
