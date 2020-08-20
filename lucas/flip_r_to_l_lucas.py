import re


open_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/lucas/right_side.cs", "r")
new_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/lucas/left_side.cs", "w+")

for line in open_file:
    x = line.replace("Right", "Left")

    new_file.write(x)
