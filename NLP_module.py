#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:59:40 2020

@author: yannis
"""

# TODO:
# multiple names for single shape x
# multiple names for verb x
# test on, on top of, right, to the right side of, etc. (no "on" for now) (on top of is tricky) x
# remove size x
#  need another fix on the code, does not handle ambiguity for between

# define scope:
# one verb, one target object, one destination object (2 or if between B and C)
# Since multiple "blue ball"s may exist, both target and desti obj should be able to have one modifier obj
# Since multiple "blue ball"s may exist, both target and desti obj should handle "leftmost" & "rightmost"
# -- "Put the red box right to the blue ball that is on the left" (?)
# will not include shape (smaller/ larger) since there are boxes that are taller but thinner than another

# Place A on B
# Place A between B and C
# Place A that is behind C on B (dependency parsing "fails", thinking C should be on B)
#    (One possible remedy: For the A that is behind C, place it on B)
#    need a fix still on current code
# Place A on B that is behind C
# Place A between B and C that is on top of D
#     need another fix on the code, does not handle ambiguity for between
# Place A between B that is on top of D and C (dependency parsing "fails", thinking B should be on C&D)
#     plan to just give up this case and ask user to do the case above.

# TODO: to the right side of something wont work, right and something are both children of side


# infer place floor or not
#   infer btm obj idx
# infer target obj idx & target xyz
#
# idx here means
# infer the shape of target obj (may not be necessary if mixed shape pi)

# Input 1, a list of obj dicts without any order (gt or vision, should have fields "shape", "color", (init)"position", "height")
# Input 2, the sentence
# Output 1, idx (wrt Input 1) of the obj to be manipulated
# Output 2, destination x and y
# Output 3, idx (wrt Input 1) of the obj to be stacked on (None if placing on floor)


# Note:
# Build structured OBJECTS list in which first entry is the target object
# looks like this code assumes 1st obj in sentence to be pick_obj, 2nd (and 3rd if "between") to be desti_obj
# modifier objs are later
# do not seem to handle right to & behind of

import spacy
from nltk import Tree
import collections
import numpy as np
from pdb import set_trace as bp

SHAPE_NAME_LIST = ["square", "box", "block", "cylinder", "ball", "sphere"]
SHAPE_NAME_MAP = {"box": "box", "block": "box", "square": "box",
                  "cylinder": "cylinder",
                  "sphere": "sphere", "ball": "sphere"}
COLOR_NAME_LIST = ["red", "green", "blue", "yellow", "grey"]
RELATION_NAME_LIST = ["right", "left", "behind", "front", "top", "on", "between"]
RELATION_NAME_MAP = {"right": "right",
                     "left": "left",
                     "behind": "behind", "back": "behind",
                     "front": "front",
                     "top": "top", "on": "top", "above": "top",
                     "between": "between"}
# DISAMBIGUATE relation: leftmost, rightmost


def PlotTree(doc):
    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(
                node.orth_, [to_nltk_tree(child) for child in node.children]
            )
        else:
            return node.orth_

    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


def traverse_and_print_from_token(token):
    # breadth first traverse of subtree with token as root
    queue = collections.deque([token])
    # print(token)
    while queue:
        vertex = queue.popleft()
        print(vertex.text)
        for neighbour in vertex.children:
            queue.append(neighbour)

    print()


# breadth-first search for shallowest match
def search_dep_tree_of_token(token, Attribute_list):
    # visited = set()
    queue = collections.deque([token])
    while queue:
        vertex = queue.popleft()
        if vertex.text in Attribute_list:
            # print('Attribute found,', vertex.text)
            return vertex.text

        for neighbour in vertex.children:
            #        if neighbour not in visited:
            #            visited.add(neighbour)
            queue.append(neighbour)


# complete breadth-first search
def search_dep_tree_of_token_multi(token, Attribute_list):
    attribute = []
    # visited = set()
    queue = collections.deque([token])
    while queue:
        vertex = queue.popleft()
        if vertex.text in Attribute_list:
            # print('Attribute found,', vertex.text)
            attribute.append(vertex.text)

        for neighbour in vertex.children:
            #        if neighbour not in visited:
            #            visited.add(neighbour)
            queue.append(neighbour)
    return attribute


def NLPmod(sentence, vision_output):
    """
    Args:
        sentence: The sentence to parse.
        vision_output: a list of obj dicts without any order (gt or vision)
             should have fields "shape", "color", (init)"position", "height" (not used in this module)
    
    Returns:
        pick_idx: idx (wrt Input 1) of the obj to be manipulated
        dest_xy: destination [x, y] in world frame
        stack_idx: idx (wrt Input 1) of the obj to be stacked on, None if placing on floor at (x, y)
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    print(doc)

    target_object = {}
    reference_objects = []
    relations = []

    PlotTree(doc)

    root_word = None
    for token in doc:
        if token.pos_ == "VERB":
            root_word = token
            break

    queue = collections.deque([root_word])
    while queue:
        token = queue.popleft()

        # if token.pos_ == "VERB":
        if token == root_word:      # asuume it is a verb
            print('Found the verb!', token)
            target_object["shape"] = search_dep_tree_of_token(
                token, SHAPE_NAME_LIST
            )
            target_object["shape"] = SHAPE_NAME_MAP[target_object["shape"]]
            target_object["color"] = search_dep_tree_of_token(
                token, COLOR_NAME_LIST
            )

        if token.text in RELATION_NAME_LIST:

            # TODO:tmp HACK, avoid double counting for "on top"/"on the top of"
            # "top" is always immediate child of "on" in the parsing tree
            ignore_on = False

            if token.text == "on":
                for child in token.children:
                    if child.text in RELATION_NAME_LIST:
                        ignore_on = True
                        break

            if not ignore_on:
                # print('Found the relations! i = ', i)

                relations.append(RELATION_NAME_MAP[token.text])

                if token.text == "between":
                    reference_object1 = {}
                    reference_object2 = {}
                    shapes = search_dep_tree_of_token_multi(token, SHAPE_NAME_LIST)
                    colors = search_dep_tree_of_token_multi(token, COLOR_NAME_LIST)
                    # sizes = search_dep_tree_of_token_multi(token, Size_list)

                    reference_object1["shape"] = SHAPE_NAME_MAP[shapes[0]]
                    reference_object2["shape"] = SHAPE_NAME_MAP[shapes[1]]
                    reference_object1["color"] = colors[0]
                    reference_object2["color"] = colors[1]

                    reference_objects.append(
                        [reference_object1, reference_object2]
                    )
                else:
                    reference_object = {}

                    reference_object["shape"] = search_dep_tree_of_token(
                        token, SHAPE_NAME_LIST
                    )
                    reference_object["shape"] = SHAPE_NAME_MAP[reference_object["shape"]]
                    reference_object["color"] = search_dep_tree_of_token(
                        token, COLOR_NAME_LIST
                    )
                    reference_objects.append(reference_object)

        for child in token.children:
            queue.append(child)

    print("--------")
    print("Target object: ", target_object)
    print("Relations: ", relations)
    print("Reference object(s) ", reference_objects)
    print("--------")

    def Parse_objects(querry, Object_list):
        object_index = []
        for i in range(len(vision_output)):
            if (
                querry["shape"] == vision_output[i]["shape"]
                and querry["color"] == vision_output[i]["color"]
            ):
                object_index = object_index + [i]
        if not object_index:
            print("No object of matching description found.")
            exit()
        return object_index

    # First, get the target object ID
    target_ID = Parse_objects(target_object, vision_output)
    print("Target ID:", target_ID)
    # Then, get the reference object ID. There might be more than one fitting the description!
    reference_ID = []
    for i in range(len(relations)):
        if relations[i] == "between":
            ID = []
            for j in range(len(reference_objects[i])):
                ID = ID + Parse_objects(reference_objects[i][j], vision_output)
        else:
            ID = Parse_objects(reference_objects[i], vision_output)
        reference_ID.append(ID)
    print("Reference IDs: ", reference_ID)

    # remove ambiguity
    # TODO: secondary relations could modify between, or modify the target object (rather than desti object)
    mult = len(reference_ID[0])
    if mult > 1 and relations[0] != "between":
        flag = [0, 0, 0, 0]         # TODO: hardcoded, at most 4 candidates
        if relations[1] == "right":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][1]
                    < vision_output[reference_ID[1][0]]["position"][1]
                ):
                    flag[i] = 1
        if relations[1] == "left":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][1]
                    > vision_output[reference_ID[1][0]]["position"][1]
                ):
                    flag[i] = 1
        if relations[1] == "front":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][0]
                    < vision_output[reference_ID[1][0]]["position"][0]
                ):
                    flag[i] = 1
        if relations[1] == "behind":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][0]
                    > vision_output[reference_ID[1][0]]["position"][0]
                ):
                    flag[i] = 1
        if relations[1] == "between":
            max_x = max(
                vision_output[reference_ID[1][0]]["position"][0],
                vision_output[reference_ID[1][1]]["position"][0],
            )
            min_x = min(
                vision_output[reference_ID[1][0]]["position"][0],
                vision_output[reference_ID[1][1]]["position"][0],
            )
            max_y = max(
                vision_output[reference_ID[1][0]]["position"][1],
                vision_output[reference_ID[1][1]]["position"][1],
            )
            min_y = min(
                vision_output[reference_ID[1][0]]["position"][1],
                vision_output[reference_ID[1][1]]["position"][1],
            )
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][0] > min_x
                    and vision_output[reference_ID[0][i]]["position"][0]
                    < max_x
                    and vision_output[reference_ID[0][i]]["position"][1]
                    > min_y
                    and vision_output[reference_ID[0][i]]["position"][1]
                    < max_y
                ):
                    flag[i] = 1
        print("Flag", flag)
        assert sum(flag) == 1  # make sure only one remains
        true_id = np.argmax(flag)
        reference_ID[0] = [reference_ID[0][true_id]]
        print("New Reference IDs", reference_ID)

    # finally, get coordinates

    def obtain_target_loc_coordinates(
        reference_ID, Vision_output, relations
    ) -> np.ndarray:
        """
        Args:
            reference_ID: The ID(s) of the objects.
            Vision_output: The object dictionaries.
            relations: The positional relations w.r.t. the target object.
        
        Returns:
            target_xyz: The xyz position of the target location.
        """
        offset = 0.2
        # for i in range(len(relations)):
        if relations[0] == "right":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, -offset, 0, 0])
        if relations[0] == "left":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, offset, 0, 0])
        if relations[0] == "front":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([-offset, 0, 0, 0])
        if relations[0] == "behind":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([offset, 0, 0, 0])
        if relations[0] == "top":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, 0, offset, 0])
        if relations[0] == "between":
            target_xyz = 0.5 * (
                np.asarray(Vision_output[reference_ID[0][0]]["position"])
                + np.asarray(Vision_output[reference_ID[0][1]]["position"])
            )
        return target_xyz

    target_xyz = obtain_target_loc_coordinates(
        reference_ID, vision_output, relations
    )
    print("--------")

    # print(target_xyz)
    # # Build structured OBJECTS list in which first entry is the target object
    # OBJECT_coord = np.array([vision_output[target_ID[0]]["position"]])
    # for i in range(len(vision_output)):
    #     if i != target_ID[0]:
    #         OBJECT_coord = np.concatenate(
    #             (OBJECT_coord, np.array([vision_output[i]["position"]]))
    #         )
    # return OBJECT_coord, target_xyz

    if relations[0] == "top":
        stack_idx = reference_ID[0][0]
    else:
        stack_idx = None

    return target_ID[0], target_xyz[:2], stack_idx


if __name__ == "__main__":
    # sentence = "Put the red box between the blue ball and yellow box"
    # obj1 = {
    #     "shape": "box",
    #     "color": "yellow",
    #     "position": np.array([1.0, 0.5, 0, 0]),
    # }  # ref 1
    # obj2 = {
    #     "shape": "box",
    #     "color": "red",
    #     "position": np.array([0.0, 0.0, 0, 0]),
    # }  # target
    # obj3 = {
    #     "shape": "sphere",
    #     "color": "blue",
    #     "position": np.array([0.0, 1, 0, 0]),
    # }  # ref 2
    # obj4 = {
    #     "shape": "sphere",
    #     "color": "yellow",
    #     "position": np.array([0.8, 0.5, 0, 0]),
    # }  # irrelevant
    # Vision_output = [obj1, obj2, obj3, obj4]
    # [OBJECTS, dest] = NLPmod(sentence=sentence, vision_output=Vision_output)

    # "Put the red box right to the blue ball" two matching - error.
    # sentence = "Put the red box right to the yellow box that is on top of the green ball"

    # sentence = "Put the red box on the right of the yellow box that is on top of the green ball"

    sentence = "For the red box in front of the blue ball, put it behind the yellow box"

    # sentence = "Pick up the red block that is to the right of the blue ball, and put it behind the yellow box"
    #
    # sentence = "Pick up the red sphere that is left to the green box, and place it in front of the blue cylinder."
    # sentence = "Pick up the red sphere that is to the left of the green box, and place it in front of the blue cylinder."

    # sentence = "Put the red box on the right on top of the yellow box"
    #
    # sentence = "Pick the red box on the right, and put it on top of the yellow box"

    # # sentence = "Put the red box right to the blue ball that is behind the yellow box"   # same as above
    # # sentence = "Put the red box that is right to the blue ball behind the yellow box"       # wrong(?) behavior
    # # for the above, the parsing is not expected already
    # # if the tree is "correct",

    # sentence = "Place the leftmost red box behind the yellow box"

    # sentence = "Place the red box between the blue ball and the yellow box that is in front of the green ball"

    obj1 = {
        "shape": "sphere",
        "color": "green",
        "position": np.array([1.0, 0.5, 0, 0]),
    }
    obj2 = {
        "shape": "box",
        "color": "red",
        "position": np.array([0.0, 0.0, 0, 0]),
    }
    obj3 = {
        "shape": "sphere",
        "color": "blue",
        "position": np.array([0.0, 1, 0, 0]),
    }
    obj4 = {
        "shape": "box",
        "color": "yellow",
        "position": np.array([1.2, 0.5, 0, 0]),
    }
    obj5 = {
        "shape": "box",
        "color": "yellow",
        "position": np.array([0.8, 0.4, 0, 0]),
    }
    Vision_output = [obj1, obj2, obj3, obj4, obj5]
    pick_idx, dest_xy, stack_idx = NLPmod(sentence=sentence, vision_output=Vision_output)


