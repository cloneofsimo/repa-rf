#!/bin/bash

# Install img2dataset if not already installed
#pip install img2dataset

# Run img2dataset
img2dataset \
    --url_list ../dataset/pd12m.csv \
    --input_format "csv" \
    --url_col "url" \
    --caption_col "caption" \
    --output_format "webdataset" \
    --output_folder "../dataset/raw_downloaded" \
    --thread_count 16 \
    --processes_count 32 \
    --thread_count 256 \
    --number_sample_per_shard 10000 \
    --min_image_size 512 \
    --resize_only_if_bigger=True --resize_mode="center_crop" --skip_reencode=True \
    --image_size 512 \
    --skip_reencode True \
    --encode_format png \
    --encode_quality 9 