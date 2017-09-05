#!/bin/bash

LSTM=test/lstm_ssim.csv
JPEG=test/jpeg_ssim.csv

echo -n "" > $LSTM
for i in {01..24..1}; do
  echo Processing test/decoded/kodim$i
  for j in {00..15..1}; do
    echo -n `python metric.py -m ssim -o test/images/kodim$i.png -c test/decoded/kodim$i/$j.png`', ' >> $LSTM
  done
  echo "" >> $LSTM
done

echo -n "" > $JPEG
for i in {01..24..1}; do
  echo Processing test/jpeg/kodim$i
  for j in {01..20..1}; do
    echo -n `python metric.py -m ssim -o test/images/kodim$i.png -c test/jpeg/kodim$i/$j.jpg`', ' >> $JPEG
  done
  echo "" >> $JPEG
done
