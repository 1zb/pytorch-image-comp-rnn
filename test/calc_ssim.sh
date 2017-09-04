#!/bin/bash

echo -n "" > ssim.csv
for i in {01..24..1}; do
  echo Processing test/decoded/kodim$i
  for j in {00..15..1}; do
    echo -n `python metric.py -m ssim -o test/images/kodim$i.png -c test/decoded/kodim$i/$j.png`', ' >> ssim.csv
  done
  echo "" >> ssim.csv
done
