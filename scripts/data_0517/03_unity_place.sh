SET=place_2K_20K_0517

time python system/run_unity_from_states.py \
    --port 8001 \
    --unity_dir /home/mguo/unity/builds/Linux8001_0515 \
    --states_dir /home/mguo/states/scenes/$SET \
    --cam_dir /home/mguo/data/$SET/json \
    --start_trial 1 \
    --end_trial 2000 \
    --stage place
