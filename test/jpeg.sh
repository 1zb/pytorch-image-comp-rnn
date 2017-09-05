#!/bin/bash

for i in {01..24..1}; do
  echo JPEG Encoding test/images/kodim$i.png
  mkdir -p test/jpeg/kodim$i
  for j in {1..20..1}; do
    convert test/images/kodim$i.png -quality $(($j*5)) -sampling-factor 4:2:0 test/jpeg/kodim$i/`printf "%02d" $j`.jpg
  done
done
