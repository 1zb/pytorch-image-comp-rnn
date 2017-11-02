#!/bin/bash

for i in {01..24..1}; do
  echo Encoding test/images/kodim$i.png
  mkdir -p test/codes
  python encoder.py --model checkpoint/encoder_epoch_00000066.pth --input test/images/kodim$i.png --cuda --output test/codes/kodim$i --iterations 16

  echo Decoding test/codes/kodim$i.npz
  mkdir -p test/decoded/kodim$i
  python decoder.py --model checkpoint/decoder_epoch_00000066.pth --input test/codes/kodim$i.npz --cuda --output test/decoded/kodim$i
done
