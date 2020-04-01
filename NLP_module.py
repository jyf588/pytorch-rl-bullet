#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:59:40 2020

@author: yannis
"""

# TODO:
# multiple names for single shape
# multiple names for verb
# test on, on top of, right, to the right side of, etc.
# remove size

# define scope:
# one verb, one target object, one destination object (2 or if between B and C)
# Since multiple "blue ball"s may exist, both target and desti obj should be able to have one modifier obj
# Since multiple "blue ball"s may exist, both target and desti obj should handle "leftmost" & "rightmost"
# -- "Put the red box right to the blue ball that is on the left" (?)
# will not include shape (smaller/ larger) since there are boxes that are taller but thinner than another


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


def PlotTree(doc):
    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(
                node.orth_, [to_nltk_tree(child) for child in node.children]
            )
        else:
            return node.orth_

    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


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
    Color_list = ["red", "green", "blue", "yellow", "grey"]
    Shape_list = ["square", "box", "block", "cylinder", "ball"]
    # Size_list = ["small", "smaller", "big", "bigger", "large", "larger"]
    Relation_list = ["right", "left", "behind", "front", "top", "between"]

    target_object = {}
    reference_objects = []
    relation = []

    PlotTree(doc)

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

    i = 0
    print(doc)
    for token in doc:
        # TODO: this is saying searching from left to right

        if token.pos_ == "VERB":
            # print('Found the verb! i = ', i)
            target_object["shape"] = search_dep_tree_of_token(
                token, Shape_list
            )
            target_object["color"] = search_dep_tree_of_token(
                token, Color_list
            )
            # target_object["size"] = search_dep_tree_of_token(token, Size_list)

        if token.text in Relation_list:
            # print('Found the relation! i = ', i)
            if token.text == "between":
                relation.append("between")
                reference_object1 = {}
                reference_object2 = {}
                shapes = search_dep_tree_of_token_multi(token, Shape_list)
                colors = search_dep_tree_of_token_multi(token, Color_list)
                # sizes = search_dep_tree_of_token_multi(token, Size_list)

                reference_object1["shape"] = shapes[0]
                reference_object2["shape"] = shapes[1]
                reference_object1["color"] = colors[0]
                reference_object2["color"] = colors[1]

                reference_objects.append(
                    [reference_object1, reference_object2]
                )
                # if sizes:
                #     reference_object1["size"] = sizes[0]
                #     reference_object2["size"] = sizes[1]
                # else:
                #     reference_object1["size"] = None
                #     reference_object2["size"] = None
                #     reference_objects.append(
                #         [reference_object1, reference_object2]
                #     )

            else:
                reference_object = {}
                relation.append(token.text)
                reference_object["shape"] = search_dep_tree_of_token(
                    token, Shape_list
                )
                reference_object["color"] = search_dep_tree_of_token(
                    token, Color_list
                )
                # reference_object["size"] = search_dep_tree_of_token(
                #     token, Size_list
                # )
                reference_objects.append(reference_object)

        i = i + 1

    print("--------")
    print("Target object: ", target_object)
    print("Relations: ", relation)
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
    for i in range(len(relation)):
        if relation[i] == "between":
            ID = []
            for j in range(len(reference_objects[i])):
                ID = ID + Parse_objects(reference_objects[i][j], vision_output)
        else:
            ID = Parse_objects(reference_objects[i], vision_output)
        reference_ID.append(ID)
    print("Reference IDs: ", reference_ID)

    # remove ambiguity
    mult = len(reference_ID[0])
    if mult > 1 and relation[0] != "between":
        flag = [0, 0, 0, 0]
        if relation[1] == "right":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][1]
                    < vision_output[reference_ID[1][0]]["position"][1]
                ):
                    flag[i] = 1
        if relation[1] == "left":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][1]
                    > vision_output[reference_ID[1][0]]["position"][1]
                ):
                    flag[i] = 1
        if relation[1] == "front":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][0]
                    < vision_output[reference_ID[1][0]]["position"][0]
                ):
                    flag[i] = 1
        if relation[1] == "behind":
            for i in range(mult):
                if (
                    vision_output[reference_ID[0][i]]["position"][0]
                    > vision_output[reference_ID[1][0]]["position"][0]
                ):
                    flag[i] = 1
        if relation[1] == "between":
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
        reference_ID, Vision_output, relation
    ) -> np.ndarray:
        """
        Args:
            reference_ID: The ID(s) of the objects.
            Vision_output: The object dictionaries.
            relation: The positional relation w.r.t. the target object.
        
        Returns:
            target_xyz: The xyz position of the target location.
        """
        offset = 0.2
        # for i in range(len(relation)):
        if relation[0] == "right":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, -offset, 0, 0])
        if relation[0] == "left":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, offset, 0, 0])
        if relation[0] == "front":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([-offset, 0, 0, 0])
        if relation[0] == "behind":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([offset, 0, 0, 0])
        if relation[0] == "top":
            target_xyz = np.asarray(
                Vision_output[reference_ID[0][0]]["position"]
            ) + np.array([0, 0, offset, 0])
        if relation[0] == "between":
            target_xyz = 0.5 * (
                np.asarray(Vision_output[reference_ID[0][0]]["position"])
                + np.asarray(Vision_output[reference_ID[0][1]]["position"])
            )
        return target_xyz

    target_xyz = obtain_target_loc_coordinates(
        reference_ID, vision_output, relation
    )
    print("--------")
    print(target_xyz)
    # Build structured OBJECTS list in which first entry is the target object
    OBJECT_coord = np.array([vision_output[target_ID[0]]["position"]])
    for i in range(len(vision_output)):
        if i != target_ID[0]:
            OBJECT_coord = np.concatenate(
                (OBJECT_coord, np.array([vision_output[i]["position"]]))
            )
    return OBJECT_coord, target_xyz


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
    #     "shape": "ball",
    #     "color": "blue",
    #     "position": np.array([0.0, 1, 0, 0]),
    # }  # ref 2
    # obj4 = {
    #     "shape": "ball",
    #     "color": "yellow",
    #     "position": np.array([0.8, 0.5, 0, 0]),
    # }  # irrelevant
    # Vision_output = [obj1, obj2, obj3, obj4]
    # [OBJECTS, dest] = NLPmod(sentence=sentence, vision_output=Vision_output)

    # "Put the red box right to the blue ball" two matching - error.
    sentence = "Put the red box right to the blue ball behind the yellow box"
    # # sentence = "Put the red box right to the blue ball that is behind the yellow box"   # same as above
    # # sentence = "Put the red box that is right to the blue ball behind the yellow box"       # wrong(?) behavior
    # # for the above, the parsing is not expected already
    # # if the tree is "correct",

    # sentence = "Place the leftmost red box behind the yellow box"

    obj1 = {
        "shape": "ball",
        "color": "blue",
        "position": np.array([1.0, 0.5, 0, 0]),
    }  # ref 1
    obj2 = {
        "shape": "box",
        "color": "red",
        "position": np.array([0.0, 0.0, 0, 0]),
    }  # target
    obj3 = {
        "shape": "ball",
        "color": "blue",
        "position": np.array([0.0, 1, 0, 0]),
    }  # ref 2
    obj4 = {
        "shape": "box",
        "color": "yellow",
        "position": np.array([0.8, 0.5, 0, 0]),
    }  # irrelevant
    Vision_output = [obj1, obj2, obj3, obj4]
    [OBJECTS, dest] = NLPmod(sentence=sentence, vision_output=Vision_output)


