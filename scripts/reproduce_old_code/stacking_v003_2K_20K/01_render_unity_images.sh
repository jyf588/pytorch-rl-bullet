SET=stacking_v003_2K_20K

time python system/run_unity_from_states.py \
    --port 8001 \
    --unity_dir /home/mguo/unity/builds/Linux8001_0515 \
    --states_dir /home/mguo/states/full/$SET \
    --start_id 0 \
    --end_id 20000 \
    --camera_control position \
    --cam_dir /home/mguo/data/$SET/json \
    --update_cam_target_every_frame
