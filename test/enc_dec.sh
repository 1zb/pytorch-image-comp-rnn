#!/bin/bash
files="test/images/*"
for fname in $files
do
	echo Encoding $fname
	mkdir -p test/codes
	npz_out=$(echo $fname | sed 's/\/images\//\/codes\//g' | sed 's/\.[a-z]\+/\.npz/g')
	python encoder.py --model checkpoint/encoder_epoch_00000066.pth --input $fname --cuda --output $npz_out --iterations 16
        
	echo Decoding $npz_out
	decoded_out=$(echo $fname | sed 's/\/images\//\/decoded\//g' | sed 's/\.[a-z]\+//g')
	mkdir -p $decoded_out
	python decoder.py --model checkpoint/decoder_epoch_00000066.pth --input $npz_out --cuda --output $decoded_out
done
