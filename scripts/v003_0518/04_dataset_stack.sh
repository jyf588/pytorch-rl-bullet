ROOT_DIR=/home/mguo
SRC_SET=stacking_v003_2K_20K
DST_SET=stacking_v003_2K_20K_0518
PNG_DIR=$ROOT_DIR/data/$DST_SET/png

time python ns_vqa_dart/bullet/gen_dataset.py \
    --format flat \
    --states_dir $ROOT_DIR/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/$SRC_SET/unity_output/images \
    --cam_dir $ROOT_DIR/data/$SRC_SET/unity_output/json \
    --dst_dir $ROOT_DIR/data/$DST_SET/data \
    --disable_pngs \
    --coordinate_frame unity_camera \
    --start_sid 0 \
    --end_sid 20000 \
    --objects_to_include -1
