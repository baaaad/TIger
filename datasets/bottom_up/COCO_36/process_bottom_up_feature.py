from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
from ipykernel import kernelapp as app

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infiles = ['./karpathy_test_resnet101_faster_rcnn_genome.tsv',
#           './karpathy_val_resnet101_faster_rcnn_genome.tsv',\
#           './karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
#            './karpathy_train_resnet101_faster_rcnn_genome.tsv.1']
infiles = ['coco_36.tsv']



output_dir = './extracted'


if not os.path.exists(output_dir+'_att'):
    os.makedirs(output_dir+'_att')
if not os.path.exists(output_dir+'_fc'):
    os.makedirs(output_dir+'_fc')
if not os.path.exists(output_dir+'_box'):
    os.makedirs(output_dir+'_box')
if not os.path.exists(output_dir+'_wh'):
    os.makedirs(output_dir+'_wh')
cocobu_size = {}

from tqdm import tqdm

for infile in infiles:
    print('Reading ' + infile)

    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        # reader = pd.read_csv(tsv_in_file, sep='\t', names = FIELDNAMES)
        c = 0
        for item in tqdm(reader):
            #             if os.path.exists(os.path.join(output_dir+'_fc', str(item['image_id'])+'.npy')):
            # #                 print(os.path.join(output_dir+'_fc', str(item['image_id'])+'.npy'))
            #                 c+=1
            #                 continue
            #             print(item.keys())
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            item['image_w'] = int(item['image_w'])
            item['image_h'] = int(item['image_h'])
            #             print(item['image_w'])
            #             print(item['image_h'])
            #             if c == 246:
            #                 c+=1
            #                 continue
            for field in ['boxes', 'features']:
                #                 print(base64.decodestring(item[field].encode('ascii')))
                item[field] = np.frombuffer(base64.decodestring(item[field].encode('ascii')),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))

            #             c+=1
            np.savez_compressed(os.path.join(output_dir + '_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(output_dir+'_box', str(item['image_id'])), item['boxes'])
            np.savez(os.path.join(output_dir+'_wh', str(item['image_id'])), w = item['image_w'],h=item['image_h'])
