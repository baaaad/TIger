import json
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
from pytorch_transformers.tokenization_bert import BertTokenizer
from .converter_for_inserter import compute_edits_and_insertions
import operator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



"""Constants"""

# Edit operations.
KEEP = 'KEEP'
DELETE = 'DELETE'
PAD_TAG = 'PAD'


# Special tokens.
PAD = '[PAD]'
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

# Special tokens that indicate the start and end of a span of deleted tokens.
DELETE_SPAN_START = '[unused1]'
DELETE_SPAN_END = '[unused2]'

# For filtering out input tokens which are not used.
DELETED_TAGS = frozenset([DELETE, PAD_TAG, PAD])


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


def edit2sent(sent, edits, last=False):
    """
    Edit the sentence given the edit operations.
    :param sent: sentence to edit, list of string
    :param edits: a sequence of edits in ['KEEP','DEL','STOP']+INS_vocab_set
    :return: the new sentence, as the edit sequence is deterministic based on the edits labels
    """
    new_sent = []
    sent_pointer = 0 #counter the total of KEEP and DEL, then align with original sentence

    if len(edits) == 0 or len(sent) ==0: # edit_list empty, return original sent
        return sent

    for i, edit in enumerate(edits):
        if len(sent) > sent_pointer: #there are tokens left for editing
            if edit =="KEEP":
                new_sent.append(sent[sent_pointer])
                sent_pointer += 1
            elif edit =="DELETE":
                sent_pointer += 1
    if sent_pointer < len(sent):
        for i in range(sent_pointer,len(sent)):
            new_sent.append(sent[i])
    return new_sent


def edit2sent_add(sent, edits, last=False):
    """
    Edit the sentence given the edit operations.
    :param sent: sentence to edit, list of string
    :param edits: a sequence of edits in ['KEEP','DEL','STOP']+INS_vocab_set
    :return: the new sentence, as the edit sequence is deterministic based on the edits labels
    """
    new_sent = []
    sent_pointer = 0 #counter the total of KEEP and DEL, then align with original sentence

    if len(edits) == 0 or len(sent) ==0: # edit_list empty, return original sent
        return sent

    for i, edit in enumerate(edits):
        if len(sent) > sent_pointer: #there are tokens left for editing
            if edit =="KEEP":
                new_sent.append(sent[sent_pointer])
                sent_pointer += 1
            elif edit =="DELETE":
                sent_pointer += 1
            else: #insert the word in
                new_sent.append(edit)
    if sent_pointer < len(sent):
        for i in range(sent_pointer,len(sent)):
            new_sent.append(sent[i])
    return new_sent



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


class Flickr30KEEInserterDataset(Dataset):
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
        self._max_insertions_per_token = 10
        self.max_predictions_per_seq = 20

        self._max_insertions_per_token_first_iter = 20
        self._max_insertions_per_token_second_iter = 20
        self._insert_after_token = True

        label_map_file_first_iter = os.path.join(dataroot, 'label_map_tagger_del.json')
        self.label_map_first_iter = read_label_map(label_map_file_first_iter)
        label_map_file_second_iter = os.path.join(dataroot, 'label_map_tagger_add.json')
        self.label_map_second_iter = read_label_map(label_map_file_second_iter)

        data_path = os.path.join(dataroot, 'Flickr30K_EE/Flickr30K_EE_' + split + ".json")
        logger.info("Loading from %s" % data_path)

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print('Whole dataset with sample:', len(self.data))

        self._entries = []
        self.dataset_size = len(self._entries)

        cache_path = os.path.join(dataroot, "Flickr30K_EE/cache_processed", task + "_" + split + ".pkl")

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


    def build_insertion_tokens(
            self,
            source_tokens,
            labels,
            target_insertions=None,
    ):
        """Constructs the masked input Example for the insertion model.

        masked_tokens ['[CLS]', 'a', '[MASK]', 'man', '[MASK]', '[MASK]', '[MASK]', 'next', 'to', 'a', '[MASK]', '[MASK]', '[MASK]', '[SEP]']
        target_tokens ['[CLS]', 'a', 'young', 'man', 'holding', 'an', 'umbrella', 'next', 'to', 'a', 'herd', 'of', 'cattle', '[SEP]']
        Args:
          source_tokens: List of source tokens.
          labels: List of edit label tuples (base_tag, num_insertions).
          target_insertions: Inserted target tokens per source token. Only provided
            when constructing training examples.

        Returns:
          A tuple with a masked insertion token sequence and the unmasked target
          sequence.

        Raises:
          ValuError: If the number of insertions in a label doesn't match the
            number of elements in the corresponding target_insertions[i] list.
        """
        masked_tokens = []
        target_tokens = []
        deletion_started = False
        # Tokens to add after finishing the deleted span.
        mask_buffer = []
        target_buffer = []
        for i, (source_token,
                (base_tag,
                 num_insertions)) in enumerate(zip(source_tokens, labels)):
            masks = [MASK] * num_insertions
            if target_insertions is not None:
                insertions = target_insertions[i]
                if len(insertions) != num_insertions:
                    raise ValueError(f'Inserted token counts do not match at {i}: '
                                     f'{len(insertions)} vs. {num_insertions}')
            else:
                insertions = masks

            if base_tag == KEEP:
                if deletion_started:
                    deletion_started = False
                    masked_tokens.extend(mask_buffer)
                    target_tokens.extend(target_buffer)
                    mask_buffer = []
                    target_buffer = []
                if self._insert_after_token:
                    masked_tokens.extend([source_token] + masks)
                    target_tokens.extend([source_token] + insertions)
                else:
                    masked_tokens.extend(masks + [source_token])
                    target_tokens.extend(insertions + [source_token])
            else:  # base_tag == constants.DELETE
                if not deletion_started:
                    masked_tokens = masked_tokens
                    target_tokens = target_tokens
                deletion_started = True
                mask_buffer.extend(masks)
                target_buffer.extend(insertions)
                #masked_tokens.append(None)
                #target_tokens.append(None)
        if deletion_started:
            # Source ended with deletion so we need to add the buffered tokens.
            masked_tokens.extend(mask_buffer)
            target_tokens.extend(target_buffer)

        assert len(masked_tokens) == len(target_tokens), (
            f"Masked token count ({len(masked_tokens)}) doesn't match the "
            f"target token count ({len(target_tokens)}).")

        # Truncate lists. Note that although `source_tokens` is already guaranteed
        # to be truncated, `masked_tokens` can still be too long since it contains
        # added MASK tokens.
        masked_tokens = masked_tokens[:self._max_seq_length]
        target_tokens = target_tokens[:self._max_seq_length]
        # Don't truncate [SEP] to match BERT input format.
        if masked_tokens[-1] != SEP:
            masked_tokens[-1] = SEP
            target_tokens[-1] = SEP

        return masked_tokens, target_tokens

    def get_entries(self):

        entries = []

        for i in range(len(self.data)):

            sample = self.data[i]
            image_id = sample['image_id']

            input_caption = ' '.join(sample['Ref-Cap_tokens']).strip()

            input_tokens = self._tokenizer.tokenize(input_caption)
            input_tokens = ["[CLS]"] + input_tokens[: self._max_seq_length - 2] + ["[SEP]"]
            # print('input_tokens', input_tokens)

            input_ids = self._tokenizer.encode(input_caption)
            input_ids = input_ids[: self._max_seq_length - 2]
            input_ids = self._tokenizer.add_special_tokens_single_sentence(input_ids)

            target = ' '.join(sample['GT-Cap_tokens'])
            output_tokens = self._tokenizer.tokenize(target)
            output_tokens = ["[CLS]"] + output_tokens[: self._max_seq_length - 2] + ["[SEP]"]
            # print('output_tokens', output_tokens)

            edits_and_insertions = compute_edits_and_insertions(
                input_tokens, output_tokens, self._max_insertions_per_token_first_iter,
                self._insert_after_token)

            if edits_and_insertions is None:
                # print(i)
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
                if label_tuple in self.label_map_first_iter:
                    labels.append(self.label_map_first_iter[label_tuple])
                else:
                    raise KeyError(
                        f"Label map doesn't contain a computed label: {label_tuple}")

            lables_new = [0 if index % 2 == 0 else 1 if (index - 1) % 2 == 0 else index for index in labels]
            label_tokens_new = ['KEEP' if token[0] == 'K' else 'DELETE' if token[0] == 'D' else token for token in
                                label_tokens]
            labels = lables_new
            label_tokens = label_tokens_new

            input_tokens_second_iter = edit2sent(input_tokens, label_tokens_new)

            while (operator.eq(input_tokens_second_iter, output_tokens) != True):

                input_caption_second_iter = ' '.join(input_tokens_second_iter[1:-1])

                input_ids = self._tokenizer.encode(input_caption_second_iter)
                input_ids = input_ids[: self._max_seq_length - 2]
                input_ids = self._tokenizer.add_special_tokens_single_sentence(input_ids)

                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                if len(input_ids) < self._max_seq_length:
                    padding = [self._padding_index] * (self._max_seq_length - len(input_ids))
                    input_ids = input_ids + padding
                    input_mask += padding
                    segment_ids += padding

                assert_eq(len(input_ids), self._max_seq_length)

                edits_and_insertions_second_iter = compute_edits_and_insertions(
                    input_tokens_second_iter, output_tokens, self._max_insertions_per_token_second_iter,
                    self._insert_after_token)

                if edits_and_insertions_second_iter is None:
                    break
                else:
                    edits_second_iter, insertions_second_iter = edits_and_insertions_second_iter

                label_tokens = []  # Labels as strings.
                label_tuples = []  # Labels as (base_tag, num_insertions) tuples.
                labels = []  # Labels as IDs.

                label_tokens_add = []

                for edit, insertion in zip(edits_second_iter, insertions_second_iter):
                    label_token = edit
                    if insertion:
                        label_token += f'|{len(insertion)}'
                        label_tokens_add.append(edit)
                        label_tokens_add.append(insertion[0])
                        label_tuple = (edit, 1)
                        label_tuples.append(label_tuple)
                    else:
                        label_tokens_add.append(edit)
                        label_tuple = (edit, len(insertion))
                        label_tuples.append(label_tuple)
                    label_tokens.append(label_token)

                    if label_tuple in self.label_map_second_iter:
                        labels.append(self.label_map_second_iter[label_tuple])
                    else:
                        raise KeyError(
                            f"Label map doesn't contain a computed label: {label_tuple}")

                insertions = [[inser[0]] if inser else inser for inser in insertions_second_iter]

                masked_tokens, target_tokens = self.build_insertion_tokens(input_tokens_second_iter, label_tuples,
                                                                                 insertions)

                mask_position = []
                mask_target_id = []
                mask_target_weight = []
                masknum = 0

                for idx, token in enumerate(masked_tokens):
                    if token != MASK:
                        mask_position.append(0)
                        mask_target_id.append(-1)
                        mask_target_weight.append(0.0)
                        continue

                    mask_position.append(1)
                    masknum += 1
                    if target_tokens:
                        mask_target_id += self._tokenizer.convert_tokens_to_ids([target_tokens[idx]])
                    else:
                        mask_target_id.append(0)
                    mask_target_weight.append(1.0)

                if masknum > self.max_predictions_per_seq:
                    continue

                segment_ids = [0] * len(masked_tokens)
                input_mask = [1] * len(masked_tokens)
                input_ids = self._tokenizer.convert_tokens_to_ids(masked_tokens)

                assert len(segment_ids) == len(input_ids)
                while len(input_ids) < self._max_seq_length:
                    segment_ids.append(0)
                    input_ids.append(0)
                    input_mask.append(0)
                    mask_position.append(0)
                    mask_target_id.append(-1)
                    mask_target_weight.append(0.0)

                input_tokens_second_iter = edit2sent_add(input_tokens_second_iter, label_tokens_add)

                target = ' '.join(input_tokens_second_iter[1:-1])

                entries.append(
                    {
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "segment_ids": segment_ids,
                        "masked_lm_positions": mask_position,
                        "masked_lm_ids": mask_target_id,
                        "masked_lm_weights": mask_target_weight,
                        "image_id": image_id,
                        "masked_tokens": masked_tokens,
                        "target_tokens": target_tokens,
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

        masked_tokens = entry["masked_tokens"]
        target_tokens = entry["target_tokens"]

        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        input_ids = torch.from_numpy(np.array(input_ids))
        input_mask = torch.from_numpy(np.array(input_mask))
        segment_ids = torch.from_numpy(np.array(segment_ids))

        masked_lm_positions = entry["masked_lm_positions"]
        masked_lm_ids = entry["masked_lm_ids"]
        masked_lm_weights = entry["masked_lm_weights"]

        masked_lm_positions = torch.from_numpy(np.array(masked_lm_positions))
        masked_lm_ids = torch.from_numpy(np.array(masked_lm_ids))
        masked_lm_weights = torch.from_numpy(np.array(masked_lm_weights))

        return (
            features,
            spatials,
            image_mask,
            input_ids,
            masked_lm_ids,
            input_mask,
            segment_ids,
            co_attention_mask,
            image_id,
            masked_lm_weights,
            masked_lm_positions,
            masked_tokens,
            target_tokens,
        )

    def __len__(self):
        return self.dataset_size



