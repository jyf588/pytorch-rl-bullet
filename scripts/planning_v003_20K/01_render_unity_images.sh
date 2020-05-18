SET=planning_v003_20K

time python system/run_unity_from_states.py \
    --port 8001 \
    --unity_dir /home/mguo/unity/builds/Linux8001_0515 \
    --states_dir /home/mguo/states/full/$SET \
    --cam_dir /home/mguo/data/$SET/json \
    --stage plan \
    --start_id 0 \
    --end_id 20000 \
    --missing_trial_info
