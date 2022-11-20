#!/bin/bash

test_imgs="test/images/*"
for fname in $test_imgs
do
	echo JPEG Encoding $fname
	if [[ $fname == *".gif"* ]]
	then
		echo Only the first frame will be converted to jpg
		is_gif="[0]"
	else
		is_gif=""
	fi
	jpeg_folder=$(echo $fname | sed 's/\/images\//\/jpeg\//g' | sed 's/\.[a-z]\+//g')
	mkdir -p $jpeg_folder
	for j in {1..20..1}
	do
		qlty=$(($j*5))
		jpeg_fname="$jpeg_folder/`printf %03d $qlty`.jpg"
		convert "${fname}${is_gif}" -quality $qlty -sampling-factor 4:2:0 $jpeg_fname
		echo Converted $fname to $jpeg_fname
	done
done
