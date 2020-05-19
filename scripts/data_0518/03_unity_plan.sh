SET=plan_20K_0518

time python system/run_unity_from_states.py \
    --port 8002 \
    --seed 1 \
    --stage plan \
    --unity_dir /home/mguo/unity/builds/Linux8002_0515 \
    --states_dir /home/mguo/states/scenes/$SET \
    --cam_dir /home/mguo/data/$SET/json \
    --cam_version v2
