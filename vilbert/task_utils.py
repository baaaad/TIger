from io import open
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from .utils import build_insertion_tokens


logger = logging.getLogger(__name__)

LossMap = {'BCEWithLogitLoss': nn.BCEWithLogitsLoss(reduction='mean'),
           'CrossEntropyLoss': nn.CrossEntropyLoss(),
            }


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):

    # given the current task, decided whether to forward the model and forward with specific loss.
    if task_cfg[task_id]['name'] in ['COCOEETaggerDel', 'COCOEETaggerAdd', 'Flickr30KEETaggerDel',
                                     'Flickr30KEETaggerAdd']:
        batch_cuda = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-3])
        batch_list = tuple(t for t in batch[-3:])

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch_cuda
        )
        label_tokens, source_text, target_text = (
            batch_list
        )

    elif task_cfg[task_id]["name"] in ['COCOEEInserter', 'Flickr30KEEInserter']:
        batch_cuda = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-2])
        batch_list = tuple(t for t in batch[-2:])

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, masked_lm_weights, masked_lm_positions = (
            batch_cuda
        )
        masked_tokens, target_tokens = (
            batch_list
        )

    else:
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )

    batch_size = features.size(0)

    # get the model output
    linguisic_prediction, tagger_prediction_scores_t = \
        model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

    loss = 0.0
    batch_score = 0.0

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "Tagger":
        weight_label = torch.FloatTensor([args.tagger_loss_ratio, 1]).cuda()
        loss_func = nn.CrossEntropyLoss(ignore_index=-1, weight=weight_label)
        tagger_class = task_cfg[task_id]["tagger_class"]

        input_mask = input_mask.view(-1)
        loss = loss_func(tagger_prediction_scores_t.view(-1, tagger_class), target.view(-1))
        loss = loss.mean()
        preds = torch.max(tagger_prediction_scores_t, 2)[1].data.view(-1)
        preds = preds * input_mask
        target = target.view(-1)
        target_num = (target != -1).sum()
        batch_score = float((preds == target).sum()) / float(target_num)

    elif task_cfg[task_id]["type"] == "Inserter":
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        vocab_size = linguisic_prediction.shape[-1]

        masked_lm_positions = masked_lm_positions.view(-1)
        loss = loss_func(linguisic_prediction.view(-1, vocab_size), target.view(-1))
        loss = loss.mean()
        preds = torch.max(linguisic_prediction, 2)[1].data.view(-1)
        preds = preds * masked_lm_positions
        target = target.view(-1)
        target_num = (masked_lm_positions == 1).sum()
        batch_score = float((preds == target).sum()) / float(target_num)

    return loss, batch_score, batch_size


def ForwardModelsTrain(args, task_cfg, device, task_id, task_count, task_iter_train, task_dataloader_train, model, task_losses, task_start_iter):

    # given the current task, decided whether to forward the model and forward with specific loss.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])
    
    task_count[task_id] += 1

    if task_cfg[task_id]['name'] in ['COCOEETaggerDel', 'COCOEETaggerAdd', 'Flickr30KEETaggerDel', 'Flickr30KEETaggerAdd']:
        batch = task_iter_train[task_id].next()
        batch_cuda = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-3])
        batch_list = tuple(t for t in batch[-3:])

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch_cuda
        )
        label_tokens, source_text, target_text = (
            batch_list
        )

    elif task_cfg[task_id]["name"] in ['COCOEEInserter', 'Flickr30KEEInserter']:
        batch = task_iter_train[task_id].next()
        batch_cuda = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-2])
        batch_list = tuple(t for t in batch[-2:])

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, masked_lm_weights, masked_lm_positions = (
            batch_cuda
        )
        masked_tokens, target_tokens = (
            batch_list
        )

    else:
        batch = task_iter_train[task_id].next()
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch
        )

    batch_size = features.size(0)

    # get the model output
    linguisic_prediction, tagger_prediction_scores_t = \
            model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

    loss = 0.0
    batch_score = 0.0

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "Tagger":
        weight_label = torch.FloatTensor([args.tagger_loss_ratio, 1]).cuda()
        loss_func = nn.CrossEntropyLoss(ignore_index=-1, weight=weight_label)
        tagger_class = task_cfg[task_id]["tagger_class"]

        input_mask = input_mask.view(-1)
        loss = loss_func(tagger_prediction_scores_t.view(-1,tagger_class), target.view(-1))
        loss = loss.mean()
        preds =  torch.max(tagger_prediction_scores_t, 2)[1].data.view(-1)
        preds = preds * input_mask
        target = target.view(-1)
        target_num = (target!=-1).sum()
        batch_score = float((preds == target).sum()) / float(target_num)

    elif task_cfg[task_id]["type"] == "Inserter":
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        vocab_size = linguisic_prediction.shape[-1]

        masked_lm_positions = masked_lm_positions.view(-1)
        loss = loss_func(linguisic_prediction.view(-1, vocab_size), target.view(-1))
        loss = loss.mean()
        preds = torch.max(linguisic_prediction, 2)[1].data.view(-1)
        preds = preds * masked_lm_positions
        target = target.view(-1)
        target_num = (masked_lm_positions == 1).sum()
        batch_score = float((preds == target).sum()) / float(target_num)

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = 'TASK' + task_id
        model_type = task_cfg[task]['type']
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]['loss']]

    return losses


def LoadDatasets(args, task_cfg, ids, split='trainval'):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        if task_cfg[task]['features_h5path1'] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]['features_h5path1']] = None
        if task_cfg[task]['features_h5path2'] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]['features_h5path2']] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != '':
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(features_h5path, 
                                                                            args.in_memory)
    
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != '':
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(features_h5path, args.in_memory)
    
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        task_ids.append(task)
        batch_size = task_cfg[task]['batch_size'] // args.gradient_accumulation_steps 
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())
        
        # num_workers = int(num_workers / len(ids))
        logger.info("Loading %s Dataset with batch size %d" %(task_cfg[task]['name'], batch_size))
        
        task_datasets_train[task] = None
        if 'train' in split:
            task_datasets_train[task] = DatasetMapTrain[task](
                                task=task_cfg[task]['name'],
                                dataroot=task_cfg[task]['dataroot'],
                                annotations_jsonpath=task_cfg[task]['train_annotations_jsonpath'],
                                split=task_cfg[task]['train_split'],
                                image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                                gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                                tokenizer=tokenizer,
                                padding_index=0,
                                max_seq_length=task_cfg[task]['max_seq_length'],
                                max_region_num=task_cfg[task]['max_region_num'],
                                )

        task_datasets_val[task] = None
        if 'val' in split:
            task_datasets_val[task] = DatasetMapTrain[task](
                                task=task_cfg[task]['name'],
                                dataroot=task_cfg[task]['dataroot'],
                                annotations_jsonpath=task_cfg[task]['val_annotations_jsonpath'],
                                split=task_cfg[task]['val_split'],
                                image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                                gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                                tokenizer=tokenizer, 
                                padding_index=0,
                                max_seq_length=task_cfg[task]['max_seq_length'],
                                max_region_num=task_cfg[task]['max_region_num'])

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if 'train' in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                #TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])

            # num_workers = 1
            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                # shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if 'val' in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

    return task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val


def LoadDatasetTest(args, task_cfg, ids):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        if task_cfg[task]['features_h5path1'] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]['features_h5path1']] = None
        if task_cfg[task]['features_h5path2'] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]['features_h5path2']] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != '':
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(features_h5path,
                                                                          args.in_memory)

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != '':
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(features_h5path, args.in_memory)

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        task_ids.append(task)
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info("Loading %s Dataset with batch size %d" % (task_cfg[task]['name'], batch_size))

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]['test_split']

        task_datasets_val[task] = DatasetMapEval[task](
            task=task_cfg[task]['name'],
            dataroot=task_cfg[task]['dataroot'],
            annotations_jsonpath=task_cfg[task]['val_annotations_jsonpath'],
            split=eval_split,
            image_features_reader=task_feature_reader1[task_cfg[task]['features_h5path1']],
            gt_image_features_reader=task_feature_reader2[task_cfg[task]['features_h5path2']],
            tokenizer=tokenizer,
            padding_index=0,
            max_seq_length=task_cfg[task]['max_seq_length'],
            max_region_num=task_cfg[task]['max_region_num'])

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val


def LoadDatasetEval(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        if task_cfg[task]['features_h5path1'] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]['features_h5path1']] = None
        if task_cfg[task]['features_h5path2'] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]['features_h5path2']] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != '':
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(features_h5path, 
                                                                            args.in_memory)
    
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != '':
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(features_h5path, args.in_memory)
    
    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        task_ids.append(task)
        batch_size =  args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
        
        num_workers = int(args.num_workers / len(ids))
        logger.info("Loading %s Dataset with batch size %d" %(task_cfg[task]['name'], batch_size))

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]['val_split']

        task_datasets_val[task] = DatasetMapEval[task](
                            task=task_cfg[task]['name'],
                            dataroot=task_cfg[task]['dataroot'],
                            annotations_jsonpath=task_cfg[task]['val_annotations_jsonpath'],
                            split=eval_split,
                            image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                            gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                            tokenizer=tokenizer, 
                            padding_index=0,
                            max_seq_length=task_cfg[task]['max_seq_length'],
                            max_region_num=task_cfg[task]['max_region_num'])
        
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def read_label_map(
            path,
            use_str_keys=False):
        """Returns label map read from the given path.

        Args:
          path: Path to the label map file.
          use_str_keys: Whether to use label strings as keys instead of
            (base tag, num insertions) tuple keys.
        """
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


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True
)

label_map_file = './datasets/label_map_tagger_add_eval.json'
label_map = read_label_map(label_map_file)
inverse_label_map = {
    tag_id: tag for tag, tag_id in label_map.items()
}


def edit2sent(sent, edits, last=False):
    """
    Edit the sentence given the edit operations.
    :param sent: sentence to edit, list of string
    :param edits: a sequence of edits in ['KEEP','DEL','STOP']+INS_vocab_set
    :return: the new sentence, as the edit sequence is deterministic based on the edits labels
    """
    new_sent = []
    sent_pointer = 0  # counter the total of KEEP and DEL, then align with original sentence

    if len(edits) == 0 or len(sent) == 0:  # edit_list empty, return original sent
        return sent

    for i, edit in enumerate(edits):
        if len(sent) > sent_pointer:  # there are tokens left for editing
            if edit == "KEEP":
                new_sent.append(sent[sent_pointer])
                sent_pointer += 1
            elif edit == "DELETE":
                sent_pointer += 1
    if sent_pointer < len(sent):
        for i in range(sent_pointer, len(sent)):
            new_sent.append(sent[i])
    return new_sent


def processcap(sent):
    '''
    DELETE the '#' in generated captions
    :param sent: raw sentence which may contain '#'
    :return: the new sentence
    '''
    i = 0
    newsent = sent
    while i < len(newsent):
        if newsent[i][0] == '#' and i == 0:
            newsent[i] = newsent[i][2:]
            i+=1
        elif newsent[i][0] == '#' and i != 0:
            newsent[i - 1] = newsent[i - 1] + newsent[i][2:]
            newsent.pop(i)

        else:
            i+=1
    return newsent


def EvaluatingTiger(
        device,
        batch,
        tagger_del,
        tagger_add,
        inserter,
        results,
        edit_round,
):
    max_seq_length_tagger = 32
    max_seq_length_inserter = 32

    # get the batch
    batch_cuda = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-3])
    batch_list = tuple(t for t in batch[-3:])

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            batch_cuda
        )
    label_tokens, source_text, target_text = (
            batch_list
        )

    batch_size = features.size(0)

    gen_lables = [[] for _ in range(batch_size)]

    # tagger_del decides whether keep or delete
    with torch.no_grad():

        linguisic_prediction, tagger_prediction_scores_t = tagger_del(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
        )

    input_mask1 = input_mask.view(-1)

    preds = torch.max(tagger_prediction_scores_t, 2)[1].data.view(-1)
    preds = preds * input_mask1



    for i in range(batch_size):
        image_id = question_id[i].item()
        input_mask_i = input_mask[i]
        question_i = question[i]
        preds_i = preds[i * max_seq_length_tagger:(i + 1) * max_seq_length_tagger]
        seq_len = (input_mask_i == 1).sum()

        question_i = question_i[:seq_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(question_i)

        preds_i = preds_i[:seq_len].tolist()

        preds_i[0] = 0
        preds_i[-1] = 0

        generate_lables = []
        for j in range(len(preds_i)-1):
            tag = preds_i[j]
            if tag == 0:
                generate_lables.append('KEEP')
                gen_lables[i].append('KEEP')
            elif tag == 1:
                generate_lables.append('DELETE')
                gen_lables[i].append('DELETE')

        quesiton_new = edit2sent(question_i, generate_lables)

        segment_ids_new = [0] * len(quesiton_new)
        input_mask_new = [1] * len(quesiton_new)

        while len(quesiton_new) < max_seq_length_tagger:
            segment_ids_new.append(0)
            quesiton_new.append(0)
            input_mask_new.append(0)

        question[i] = torch.tensor(quesiton_new).unsqueeze(0).cuda()
        segment_ids[i] = torch.tensor(segment_ids_new).unsqueeze(0).cuda()
        input_mask[i] = torch.tensor(input_mask_new).unsqueeze(0).cuda()

    # edit n rounds
    for round_i in range(edit_round):

        # tagger_add decides where to add
        with torch.no_grad():

            linguisic_prediction, tagger_prediction_scores_t = tagger_add(
                question,
                features,
                spatials,
                segment_ids,
                input_mask,
                image_mask,
                co_attention_mask,
            )

        input_mask1 = input_mask.view(-1)

        preds = torch.max(tagger_prediction_scores_t, 2)[1].data.view(-1)
        preds = preds * input_mask1

        generate_lables2 = [[] for _ in range(batch_size)]
        mask_position = torch.zeros([batch_size, max_seq_length_tagger]).cuda()
        for i in range(batch_size):
            image_id = question_id[i].item()
            input_mask_i = input_mask[i]
            question_i = question[i]
            preds_i = preds[i * max_seq_length_tagger:(i + 1) * max_seq_length_tagger]

            seq_len = (input_mask_i == 1).sum()

            question_i = question_i[:seq_len].tolist()
            tokens = tokenizer.convert_ids_to_tokens(question_i)
            preds_i = preds_i[:seq_len].tolist()
            preds_i[-1]=0 #sep

            sample_i = i
            for j in range(len(preds_i)-1):
                tag = preds_i[j]
                if tag == 0:
                    generate_lables2[sample_i].append('KEEP')
                else:
                    generate_lables2[sample_i].append('KEEP')
                    generate_lables2[sample_i].append('ADD')
                    gen_lables[i].append('ADD')

            label_tuples = [inverse_label_map[int(tag)] for tag in preds_i]
            tokens_insert = build_insertion_tokens(tokens, label_tuples)

            if tokens_insert is None:
                continue

            masked_tokens = tokens_insert[0]
            mask_position_new = []
            mask_target_id = []

            for idx, token in enumerate(masked_tokens):
                if token != '[MASK]':
                    mask_position_new.append(0)
                    mask_target_id.append(-1)
                    continue

                mask_position_new.append(1)
                mask_target_id.append(0)

            segment_ids_new = [0] * len(masked_tokens)
            input_mask_new = [1] * len(masked_tokens)
            input_ids_new = tokenizer.convert_tokens_to_ids(masked_tokens)

            while len(input_ids_new) < max_seq_length_inserter:
                segment_ids_new.append(0)
                input_ids_new.append(0)
                input_mask_new.append(0)
                mask_position_new.append(0)

            question[i] = torch.tensor(input_ids_new).unsqueeze(0).cuda()
            segment_ids[i] = torch.tensor(segment_ids_new).unsqueeze(0).cuda()
            input_mask[i] = torch.tensor(input_mask_new).unsqueeze(0).cuda()
            mask_position[i] = torch.tensor(mask_position_new).unsqueeze(0).cuda()

        # inserter predicts specific token for [MASK]
        with torch.no_grad():

            linguisic_prediction, tagger_prediction_scores_t = inserter(
                question,
                features,
                spatials,
                segment_ids,
                input_mask,
                image_mask,
                co_attention_mask,
            )

        masked_lm_positions = mask_position.view(-1)

        preds = torch.max(linguisic_prediction, 2)[1].data.view(-1)
        preds = preds * masked_lm_positions

        for i in range(batch_size):

            image_id = question_id[i].item()
            input_mask_i = input_mask[i]
            question_i = question[i]
            preds_i = preds[i * max_seq_length_inserter:(i + 1) * max_seq_length_inserter]
            seq_len = (input_mask_i == 1).sum()

            question_i = question_i[:seq_len].tolist()
            tokens = tokenizer.convert_ids_to_tokens(question_i)
            preds_i = preds_i.tolist()

            current_mask = 0
            new_tokens = []
            gt_tokens = []
            in_deletion_bracket = False

            index = np.nonzero(preds_i)[0].tolist()

            for token in tokens:
                if token.lower() == '[unused2]':
                    in_deletion_bracket = False
                    continue
                elif in_deletion_bracket:
                    continue
                elif token.lower() == '[unused1]':
                    in_deletion_bracket = True
                    continue

                if token.lower() == '[MASK]'.lower():
                    new_tokens.append(
                        tokenizer.convert_ids_to_tokens(
                            [preds_i[index[current_mask]]])[0])
                    current_mask += 1
                else:
                    new_tokens.append(token)

                    gt_tokens.append(token)

            gen_sentence = ' '.join(processcap(new_tokens[1:-1]))

            input_ids_new = tokenizer.convert_tokens_to_ids(new_tokens)
            segment_ids_new = [0] * len(input_ids_new)
            input_mask_new = [1] * len(input_ids_new)

            while len(input_ids_new) < max_seq_length_tagger:
                segment_ids_new.append(0)
                input_ids_new.append(0)
                input_mask_new.append(0)

            item_dict = {"image_id": image_id, "ref_cap": source_text[i], "gt_cap": target_text[i], "caption": gen_sentence, "id": len(results[round_i])+1}
            #item_dict = {"image_id": image_id, "edit_operations": gen_lables[i], "caption": gen_sentence, "id": len(results[round_i]) + 1}

            results[round_i].append(item_dict)

            question[i] = torch.tensor(input_ids_new).unsqueeze(0).cuda()
            segment_ids[i] = torch.tensor(segment_ids_new).unsqueeze(0).cuda()
            input_mask[i] = torch.tensor(input_mask_new).unsqueeze(0).cuda()

    return results


