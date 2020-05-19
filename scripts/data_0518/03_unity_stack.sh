SET=stack_2K_20K_0518

time python system/run_unity_from_states.py \
    --port 8001 \
    --seed 1 \
    --stage place \
    --unity_dir /home/mguo/unity/builds/Linux8001_0515 \
    --states_dir /home/mguo/states/scenes/$SET \
    --cam_dir /home/mguo/data/$SET/json \
    --start_trial 1 \
    --end_trial 2000 \
    --cam_version v2
