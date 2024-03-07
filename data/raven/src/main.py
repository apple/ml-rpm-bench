# -*- coding: utf-8 -*-
"""
Developed base on the code from https://github.com/WellyZhang/RAVEN
"""

import argparse
import copy
import os
import random
import json

import numpy as np
from tqdm import trange
import pandas as pd

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from const import IMAGE_SIZE
from rendering import (imsave, render_panel, merge_matrix_answer, split_matrix_answer)
from sampling import sample_attr, sample_attr_avail, sample_rules
from serialize import serialize_aot, serialize_rules
from solver import solve


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def fuse(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # header: index	image_path	question	answer	type	A	B	C	D	E	F	G	H
    fuse_output_tsv = pd.DataFrame(columns=["index", "image_path", "question", "answer", "type", "A", "B", "C", "D", "E", "F", "G", "H"])
    segment_output_tsv = pd.DataFrame(columns=["index", "image_path", "question", "answer", "type", "A", "B", "C", "D", "E", "F", "G", "H"])
    for k in trange(args.num_samples * len(all_configs)):
        tree_name = random.choice(list(all_configs.keys()))
        root = all_configs[tree_name]
        while True:
            rule_groups = sample_rules()
            new_root = root.prune(rule_groups)    
            if new_root is not None:
                break
        
        start_node = new_root.sample()

        row_1_1 = copy.deepcopy(start_node)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_1_2 = rule_num_pos.apply_rule(row_1_1)
            row_1_3 = rule_num_pos.apply_rule(row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_2 = rule.apply_rule(row_1_1, row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_3 = rule.apply_rule(row_1_2, row_1_3)
            if l == 0:
                to_merge = [row_1_1, row_1_2, row_1_3]
            else:
                merge_component(to_merge[1], row_1_2, l)
                merge_component(to_merge[2], row_1_3, l)
        row_1_1, row_1_2, row_1_3 = to_merge

        row_2_1 = copy.deepcopy(start_node)
        row_2_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_2_2 = rule_num_pos.apply_rule(row_2_1)
            row_2_3 = rule_num_pos.apply_rule(row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_2 = rule.apply_rule(row_2_1, row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_3 = rule.apply_rule(row_2_2, row_2_3)
            if l == 0:
                to_merge = [row_2_1, row_2_2, row_2_3]
            else:
                merge_component(to_merge[1], row_2_2, l)
                merge_component(to_merge[2], row_2_3, l)
        row_2_1, row_2_2, row_2_3 = to_merge

        row_3_1 = copy.deepcopy(start_node)
        row_3_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_3_2 = rule_num_pos.apply_rule(row_3_1)
            row_3_3 = rule_num_pos.apply_rule(row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_2 = rule.apply_rule(row_3_1, row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_3 = rule.apply_rule(row_3_2, row_3_3)
            if l == 0:
                to_merge = [row_3_1, row_3_2, row_3_3]
            else:
                merge_component(to_merge[1], row_3_2, l)
                merge_component(to_merge[2], row_3_3, l)
        row_3_1, row_3_2, row_3_3 = to_merge

        imgs = [render_panel(row_1_1),
                render_panel(row_1_2),
                render_panel(row_1_3),
                render_panel(row_2_1),
                render_panel(row_2_2),
                render_panel(row_2_3),
                render_panel(row_3_1),
                render_panel(row_3_2),
                np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
        context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]

        modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
        answer_AoT = copy.deepcopy(row_3_3)
        candidates = [answer_AoT]
        for j in range(7):
            component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
            answer_j = copy.deepcopy(answer_AoT)
            answer_j.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)
            candidates.append(answer_j)

        random.shuffle(candidates)
        answers = []
        for candidate in candidates:
            answers.append(render_panel(candidate))
        imsave(merge_matrix_answer(imgs, answers), args.save_dir+"/fused/{}.jpg".format(k))
        target = candidates.index(answer_AoT)
        map_id2letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5:"F", 6:"G", 7:"H"}

        fuse_output_tsv = pd.concat([fuse_output_tsv, pd.DataFrame({"index": k, "image_path": "{}/fused/{}.jpg".format(args.save_dir, k), "question": "holder", "answer": map_id2letter[target], "type": tree_name, "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H"}, index=[0])], ignore_index=True)

        questions, answer = split_matrix_answer(imgs, answers)
        for ii, question in enumerate(questions):
            os.makedirs(args.save_dir+"/segmented/{}".format(k), exist_ok=True)
            imsave(question, args.save_dir+"/segmented/{}/q{}.jpg".format(k, ii))
        for ii, answer in enumerate(answer):
            imsave(answer, args.save_dir+"/segmented/{}/{}.jpg".format(k, map_id2letter[ii]))
        segment_output_tsv = pd.concat([segment_output_tsv, pd.DataFrame({"index": k, "image_path": "{}/segmented/{}.jpg".format(args.save_dir, k), "question": "holder", "answer": map_id2letter[target], "type": tree_name, "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H"}, index=[0]) ], ignore_index=True)
    fuse_output_tsv.to_csv("data/raven_fused.tsv", sep="\t", index=False)
    segment_output_tsv.to_csv("data/raven_segmented.tsv", sep="\t", index=False)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=20000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="~/Datasets/",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=1234,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")                             
    args = main_arg_parser.parse_args()

    all_configs = {"center_single": build_center_single(),
                   "distribute_four": build_distribute_four(),
                   "distribute_nine": build_distribute_nine(),
                   "left_center_single_right_center_single": build_left_center_single_right_center_single(),
                   "up_center_single_down_center_single": build_up_center_single_down_center_single(),
                   "in_center_single_out_center_single": build_in_center_single_out_center_single(),
                   "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()}

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if os.path.exists(os.path.join(args.save_dir, "fused")):
        os.system("rm -rf {}".format(os.path.join(args.save_dir, "fused")))
    if not os.path.exists(os.path.join(args.save_dir, "fused")):
        os.mkdir(os.path.join(args.save_dir, "fused"))
    fuse(args, all_configs)
if __name__ == "__main__":
    main()
