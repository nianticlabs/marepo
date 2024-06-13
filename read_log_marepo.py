#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import os.path as osp
import sys
from distutils.util import strtobool

Wayspots=['wayspots_bears', 'wayspots_cubes', 'wayspots_inscription', 'wayspots_lawn', 'wayspots_map',
          'wayspots_squarebench', 'wayspots_statue', 'wayspots_tendrils', 'wayspots_therock', 'wayspots_wintersign']


Mapfree_val = [f'mapfree_s{num:05d}' for num in range(410,460)]

Seven_Scenes=['7scenes_chess', '7scenes_fire', '7scenes_heads', '7scenes_office', '7scenes_pumpkin', '7scenes_redkitchen', '7scenes_stairs']

Seven_Scenes_Pgt=['pgt_7scenes_chess', 'pgt_7scenes_fire', 'pgt_7scenes_heads', 'pgt_7scenes_office', 'pgt_7scenes_pumpkin', 'pgt_7scenes_redkitchen', 'pgt_7scenes_stairs']

Twelve_Scenes=["12scenes_apt1_kitchen", "12scenes_apt1_living", "12scenes_apt2_bed", "12scenes_apt2_kitchen", "12scenes_apt2_living", "12scenes_apt2_luke", "12scenes_office1_gates362", "12scenes_office1_gates381", "12scenes_office1_lounge", "12scenes_office1_manolis", "12scenes_office2_5a", "12scenes_office2_5b"]

Twelve_Scenes_Pgt=["pgt_12scenes_apt1_kitchen", "pgt_12scenes_apt1_living", "pgt_12scenes_apt2_bed", "pgt_12scenes_apt2_kitchen", "pgt_12scenes_apt2_living", "pgt_12scenes_apt2_luke", "pgt_12scenes_office1_gates362", "pgt_12scenes_office1_gates381", "pgt_12scenes_office1_lounge", "pgt_12scenes_office1_manolis", "pgt_12scenes_office2_5a", "pgt_12scenes_office2_5b"]


_logger = logging.getLogger(__name__)

def _strtobool(x):
    return bool(strtobool(x))

def p2f(x):
    return float(x.strip('\n').strip('%')) / 100

def parse_line_from_file(log_file, pct_dict):

    n=10
    with open(log_file, 'r') as file:

        # read last n=10 lines, start from e.g. INFO:__main__:  5m/10deg: 97.22%
        Lines = file.readlines()[-n:]

        # start
        for i in range(n-3):
            this_line = Lines[i].split(" ") # ['INFO:__main__:\t5m/10deg:', '100.00%\n']

            this_line_0_front, this_line_0_rear = this_line[0].split('\t') # ['INFO:__main__:', '5m/10deg:']

            if this_line_0_rear == '5m/10deg:':
                pct_dict['pct500_10'] += p2f(this_line[1]) # add float number representing %, e.g. 97% -> 0.97

            if this_line_0_rear == '0.5m/5deg:':
                pct_dict['pct50_5'] += p2f(this_line[1])

            if this_line_0_rear == '0.25m/2deg:':
                pct_dict['pct25_2'] += p2f(this_line[1])
                
            if this_line_0_rear == '10cm/5deg:':
                pct_dict['pct10_5'] += p2f(this_line[1]) # add float number representing %, e.g. 97% -> 0.97

            if this_line_0_rear == '5cm/5deg:':
                pct_dict['pct5'] += p2f(this_line[1])

            if this_line_0_rear == '2cm/2deg:':
                pct_dict['pct2'] += p2f(this_line[1])
                
            if this_line_0_rear == '1cm/1deg:':
                pct_dict['pct1'] += p2f(this_line[1])
        
        # Split the median and mean error lines.
        def extract_r_t(line):
            # Line looks like: "INFO:__main__:Median Error: 0.00 deg, 0.00 cm"
            r, t = line.split(":")[-1].strip().split(",")
            return float(r.strip().split(" ")[0]), float(t.strip().split(" ")[0])

        # Extract median.
        median_r, median_t = extract_r_t(Lines[-3])
        pct_dict['median_r'] += median_r
        pct_dict['median_t'] += median_t

        mean_r, mean_t = extract_r_t(Lines[-2])
        pct_dict['mean_r'] += mean_r
        pct_dict['mean_t'] += mean_t

    return pct_dict

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Compute metrics for a pre-existing poses file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset', type=str, help='name of the dataset e.g. Wayspots, 7Scenes')
    parser.add_argument('model_path', type=str, help='path of model')
    parser.add_argument('datatype', type=str, help='choices: train, val, test')
    parser.add_argument('--finetune', type=_strtobool, default=False,
                        help='finetuned model using pretrained marepo')
    args = parser.parse_args()

    if args.dataset=='Wayspots' and args.datatype=='test':
        scene_dict = Wayspots
    elif args.dataset=='Wayspots' and args.datatype=='val':
        scene_dict = Mapfree_val
    elif args.dataset=='7Scenes':
        scene_dict = Seven_Scenes
    elif args.dataset=='7Scenes_pgt':
        scene_dict = Seven_Scenes_Pgt
    elif args.dataset=='12Scenes':
        scene_dict = Twelve_Scenes
    elif args.dataset == '12Scenes_pgt':
        scene_dict = Twelve_Scenes_Pgt
    else:
        print("unrecognized dataset, please check")
        NotImplementedError

        sys.exit()

    pct_dict = {
        'pct500_10': 0,
        'pct50_5': 0,
        'pct25_2': 0,
        'pct10_5': 0,
        'pct5': 0,
        'pct2': 0,
        'pct1': 0,
        'median_r': 0,
        'median_t': 0,
        'mean_r': 0,
        'mean_t': 0,
    }
    for scene in scene_dict:

        if args.finetune:
            log_file = osp.join(args.model_path, 'log_Finetune_Marepo_' + scene + '_' + args.datatype + '.txt')
        else:
            log_file=osp.join(args.model_path, 'log_Marepo_'+scene+'_'+args.datatype+'.txt')
        print(log_file)
        pct_dict = parse_line_from_file(log_file, pct_dict)
    # breakpoint()
    print(f"{args.dataset} {args.datatype} dataset mean accuracy:")
    print(f"5m/10deg: {pct_dict['pct500_10']*100/len(scene_dict):.2f}%" )
    print(f"0.5m/5deg: {pct_dict['pct50_5'] * 100 / len(scene_dict):.2f}%")
    print(f"0.25m/2deg: {pct_dict['pct25_2'] * 100 / len(scene_dict):.2f}%")
    print(f"10cm/5deg: {pct_dict['pct10_5'] * 100 / len(scene_dict):.2f}%")
    print(f"5cm/5deg: {pct_dict['pct5'] * 100 / len(scene_dict):.2f}%")
    print(f"2cm/2deg: {pct_dict['pct2'] * 100 / len(scene_dict):.2f}%")
    print(f"1cm/1deg: {pct_dict['pct1'] * 100 / len(scene_dict):.2f}%")
    print(f"Avg. Median Error: {pct_dict['median_r']/len(scene_dict):.2f} deg, {pct_dict['median_t']/len(scene_dict):.2f} cm")
    print(f"Avg. Mean Error: {pct_dict['mean_r']/len(scene_dict):.2f} deg, {pct_dict['mean_t']/len(scene_dict):.2f} cm")

