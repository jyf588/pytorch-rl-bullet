#!/usr/bin/env bash

python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 1
python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 0
python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 0 --flag_0426 1

python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 0 --use_height 1
python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 0 --use_height 0
python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 0 --use_height 0 --flag_0426 1

python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 1 --use_height 1
python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 1 --use_height 0
python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 1 --use_height 0 --flag_0426 1

python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 0 --use_height 1
python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 0 --use_height 0
python main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 0 --use_height 0 --flag_0426 1

#*******total: 0.593)*******total w/o OR: 0.754)
#*******total: 0.640)*******total w/o OR: 0.824)
#*******total: 0.607)*******total w/o OR: 0.788)
#
#*******total: 0.640)*******total w/o OR: 0.771)
#*******total: 0.697)*******total w/o OR: 0.867)
#*******total: 0.660)*******total w/o OR: 0.811)
#
#*******total: 0.620)*******total w/o OR: 0.873)
#*******total: 0.563)*******total w/o OR: 0.790)
#*******total: 0.663)*******total w/o OR: 0.861)
#
#*******total: 0.813)*******total w/o OR: 0.917)
#*******total: 0.747)*******total w/o OR: 0.852)
#*******total: 0.763)*******total w/o OR: 0.902)


