#!/usr/bin/env bash

# test_placing: 0 test stacking; 1 test placing on table
# long_move: 0 normal testing; 1 only test harder cases when
# source and destination are further away from each other
python3 main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 1 --add_place_stack_bit 1
python3 main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 0 --use_height 1 --add_place_stack_bit 1
python3 main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 1 --use_height 1 --add_place_stack_bit 1
python3 main_sim_clean_test.py --seed 1099 --test_placing 1 --long_move 0 --use_height 1 --add_place_stack_bit 1