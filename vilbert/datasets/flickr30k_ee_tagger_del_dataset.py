import json
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
from pytorch_transformers.tokenization_bert import BertTokenizer
from .converter_for_inserter import compute_edits_and_insertions
import collections


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def read_label_map(
            path,
            use_str_keys=False):
        """Returns label map read from the given path.

        Args:
          path: Path to the label map file.
          use_str_keys: Whether to use label strings as keys instead of
            (base tag, num insertions) tuple keys. The latter is only used by
            FelixInsert.
        """
        label_map = {}
        label_map = json.load(open(path))
        if not use_str_keys:
            new_label_map = {}
            for key, val in label_map.items():
                if "|" in key:
                    pos_pipe = key.index("|")
                    new_key = (key[:pos_pipe], int(key[pos_pipe + 1:]))
                else:
                    new_key = (key, 0)
                new_label_map[new_key] = val
            label_map = new_label_map
        return label_map


def image_bp_features_reader(image_id):

    att_path = 'datasets/bottom_up/Flickr30K_36/extracted_att/' + str(image_id) + '.npz'
    box_path = 'datasets/bottom_up/Flickr30K_36/extracted_box/' + str(image_id) + '.npy'
    wh_path = 'datasets/bottom_up/Flickr30K_36/extracted_wh/' + str(image_id) + '.npz'

    features = np.load(att_path)['feat']
    # img = torch.FloatTensor(att_f)

    box_f = np.load(box_path)
    image_w = np.load(wh_path)['w']
    image_h = np.load(wh_path)['h']


    num_boxes = features.shape[0]

    # print(att_f[:2])
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1

    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((box_f.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = box_f
    image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
    )
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    return features, num_boxes, image_location


class Flickr30KEETaggerDELDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        padding_index=0,
        max_seq_length=32,
        max_region_num=37,
    ):

        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self._max_region_num = max_region_num
        self.num_labels = 2
        self._max_insertions_per_token = 20
        self._insert_after_token = True
        label_map_path = os.path.join(dataroot, 'label_map_tagger_del.json')
        self.label_map = read_label_map(label_map_path)

        self._entries = []
        self.dataset_size = len(self._entries)

        cache_path = os.path.join(dataroot, "Flickr30K_EE/cache_processed", task + "_" + split + ".pkl")

        print('split:', split)

        data_path = os.path.join(dataroot, 'Flickr30K_EE/Flickr30K_EE_' + split + ".json")

        logger.info("Loading from %s" % data_path)

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        if not os.path.exists(cache_path):
            self._entries = self.get_entries()
            self.dataset_size = len(self._entries)
            print('Valid sample:', self.dataset_size)
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))
            self.dataset_size = len(self._entries)
            print('Valid sample:', self.dataset_size)

    def get_entries(self):

        entries = []

        for i in range(len(self.data)):
            sample = self.data[i]
            image_id = sample['image_id']

            input_caption = ' '.join(sample['Ref-Cap_tokens']).strip()

            input_tokens = self._tokenizer.tokenize(input_caption)
            input_tokens = ["[CLS]"] + input_tokens[: self._max_seq_length - 2] + ["[SEP]"]

            input_ids = self._tokenizer.encode(input_caption)
            input_ids = input_ids[: self._max_seq_length - 2]
            input_ids = self._tokenizer.add_special_tokens_single_sentence(input_ids)

            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)

            if len(input_ids) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(input_ids))
                input_ids = input_ids + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(input_ids), self._max_seq_length)

            target = ' '.join(sample['GT-Cap_tokens'])
            output_tokens = self._tokenizer.tokenize(target)
            output_tokens = ["[CLS]"] + output_tokens[: self._max_seq_length - 2] + ["[SEP]"]

            edits_and_insertions = compute_edits_and_insertions(
                input_tokens, output_tokens, self._max_insertions_per_token,
                self._insert_after_token)

            if edits_and_insertions is None:
                continue
            else:
                edits, insertions = edits_and_insertions

            label_tokens = []  # Labels as strings.
            label_tuples = []  # Labels as (base_tag, num_insertions) tuples.
            labels = []  # Labels as IDs.
            for edit, insertion in zip(edits, insertions):
                label_token = edit
                if insertion:
                    label_token += f'|{len(insertion)}'
                label_tokens.append(label_token)
                label_tuple = (edit, len(insertion))
                label_tuples.append(label_tuple)
                if label_tuple in self.label_map:
                    labels.append(self.label_map[label_tuple])
                else:
                    raise KeyError(
                        f"Label map doesn't contain a computed label: {label_tuple}")

            lables_new = [0 if index % 2 == 0 else 1 if (index - 1) % 2 == 0 else index for index in labels]
            label_tokens_new = ['KEEP' if token[0] == 'K' else 'DELETE' if token[0] == 'D' else token for token in
                                label_tokens]

            labels = lables_new
            labels[0] = -1  # cls
            labels[-1] = -1 # sep
            label_tokens = label_tokens_new


            if len(labels) < self._max_seq_length:
                padding = [-1] * (self._max_seq_length - len(labels))
                labels = labels + padding

            entries.append(
                {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "labels": labels,
                    "label_tokens": label_tokens,
                    "image_id": image_id,
                    "source_text": input_caption,
                    "target_text": target,
                }
            )

        return entries

    def __getitem__(self, index):

        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes = image_bp_features_reader(image_id)

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        source_text = entry["source_text"]
        target_text = entry["target_text"]

        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        input_ids = torch.from_numpy(np.array(input_ids))
        input_mask = torch.from_numpy(np.array(input_mask))
        segment_ids = torch.from_numpy(np.array(segment_ids))

        labels = entry["labels"]
        label_tokens = entry["label_tokens"]
        labels = torch.from_numpy(np.array(labels))

        return (
            features,
            spatials,
            image_mask,
            input_ids,
            labels,
            input_mask,
            segment_ids,
            co_attention_mask,
            image_id,
            label_tokens,
            source_text,
            target_text,
        )

    def __len__(self):
        return self.dataset_size



