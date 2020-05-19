import os
import cv2


from ns_vqa_dart.bullet import util

# run_dir = "/home/mguo/outputs/system/t1/0404/2020_05_17_15_17_16"
run_dir = "/home/mguo/outputs/system/t1/0404/2020_05_17_21_48_18"
task = "stack"
sid = "31"
ts = "000646"
pickle_path = os.path.join(run_dir, "pickle", f"{task}_{sid}_{ts}.p")
p = util.load_pickle(pickle_path)
debug = 0

# Model inputs.
input_rgb = p["vision_inputs"]
input_bgr = input_rgb[:, :, ::-1]
input_bgr = cv2.resize(input_bgr, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite("image.png", input_bgr)

# print out the predicted values.
# update visualization
