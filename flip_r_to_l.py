import re
# scale = """
#         stuffscale=\"0.001 0.001 0.001\"stuff
#         stuffscale=\"0.001 0.001 0.001\"stuff
#         """
# x = re.sub("scale=\"0.001 0.001 0.001\"", "scale=\"0.001 -0.001 0.001\"", scale)
# print(x)

# origin = "stuff <origin rpy=\"10 20 30\" xyz=\"-0.054 -0.009 -0.2621\"/> stuff"

"""
TODO: Generalize replace negatives function
FIXME: Escape "." with "\"
"""

# open_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf", "r")
# new_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/left_experimental.urdf", "w+")
open_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_right_hand.urdf", "r")
new_file = open("/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_left_hand.urdf", "w+")

def conditional_replace_origin(match):
    groups = match.groups()
    new_groups = (groups[0], groups[2], groups[3], groups[5], groups[7], groups[8])
    return_str = r"{}"
    if not match.group(2):
        return_str += r"-"
    return_str += r"{}{}"
    if not match.group(5):
        return_str += r"-"
    return_str += r"{}"
    if not match.group(7):
        return_str += r"-"
    return_str += r"{}{}"
    
    return return_str.format(*new_groups) 
        
def conditional_replace_scale(match):
    groups= match.groups()
    groups = (groups[0], groups[2])
    if match.group(2):
        return r"{}{}".format(*groups)
    return r"{}-{}".format(*groups)

def condtional_replace_axis(match):
    groups = match.groups()
    if match.group(4):
        new_groups = (groups[0], groups[3], groups[5], groups[7], groups[10])
        if match.group(2):
            return r"{}{}{}{}{}".format(*new_groups)
        else:
            return r"{}-{}{}{}{}".format(*new_groups)
    elif match.group(9):
        new_groups = (groups[0], groups[2], groups[5], groups[8], groups[10])
        if match.group(7):
            return r"{}{}{}{}{}".format(*new_groups)
        else:
            return r"{}{}{}-{}{}".format(*new_groups)
    else:
        new_groups = (groups[0], groups[2], groups[5], groups[7], groups[10])
        return r"{}{}{}{}{}".format(*new_groups)

def contional_replace_inertia(match):
    groups = match.groups()
    groups = (groups[0], groups[2], groups[4])
    return_str = r"{}"
    if not match.group(2):
        return_str += "-"
    return_str += r"{}"
    if not match.group(4):
        return_str += "-"
    return_str += r"{}"
    return return_str.format(*groups)

for line in open_file:
    x = line
    x = x.replace("rh", "lh")
    x = x.replace("\"r_", "\"l_")
    if re.search("origin", line) is not None:
        # 1-9 groups: negative @2, 5, 7
        x = re.sub("(<origin rpy=\")(-?)(\d*.?\d+)( -?\d*.?\d+ )(-?)(\d*.?\d+\" xyz=\"-?\d*.?\d+ )(-?)(\d*.?\d+)( -?\d*.?\d+\"/>)", conditional_replace_origin, line)
        if x == line:
            print("Origin not replaced")
            print(x)
    elif re.search("scale", line) is not None:
        x = re.sub("(scale=\"-?\d*.?\d+ )(-?)(\d*.?\d+ -?\d*.?\d+\")", conditional_replace_scale, line)
        if x == line:
            print("Scale not replaced")
            print(x)
    elif re.search("axis", line) is not None:
        # 1: (<axis xyz=\")
        # 2: (-?)
        # 3: (0?)
        # 4: (1?)
        # 5: (\.0)?
        # 6: ( -?[0-1].?0? )
        # 7: (-?)
        # 8: (0?)
        # 9: (1?)
        # 10: (\.0)?
        # 11: (\"\/>)
        x = re.sub("(<axis xyz=\")(-?)(0?)(1?)(\.0)?( -?[0-1].?0? )(-?)(0?)(1?)(\.0)?(\"\/>)", condtional_replace_axis, line)
        if x == line:
            print("Axis not replaced")
            print(x)
    elif re.search("<inertia ixx", line) is not None:
        x = re.sub("(<inertia ixx=\"-?\d*.?\d+\" ixy=\")(-?)(\d*.?\d+\" ixz=\"-?\d*.?\d+\" iyy=\"-?\d*.?\d+\" iyz=\")(-?)(\d*.?\d+\" izz=\"-?\d*.?\d+\"/>)",
           contional_replace_inertia, line)
        if x == line:
            print("Inertia not replaced")
            print(x)

    new_file.write(x)
