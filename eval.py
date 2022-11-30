import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from easydict import EasyDict as edict
from tqdm import tqdm
import torch
import yaml
from vilbert.task_utils import LoadDatasetEval, LoadDatasetTest, LoadLosses, EvaluatingTiger
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained_tagger_del",
        default="bert-base-uncased",
        type=str,
        help="The pre-trained tagger_del model.",
    )
    parser.add_argument(
        "--from_pretrained_tagger_add",
        default="bert-base-uncased",
        type=str,
        help="The pre-trained tagger_add model.",
    )
    parser.add_argument(
        "--from_pretrained_inserter",
        default="bert-base-uncased",
        type=str,
        help="The pre-trained inserter model.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the evaluation results will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for evaluation results.",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... task separate by -"
    )
    parser.add_argument(
        "--split", default="", type=str, help="which split to use."
    )
    parser.add_argument(
        "--edit_round", default="5", type=int, help="how many round to edit."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )

    args = parser.parse_args()
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from vilbert.vilbert import BertConfig
    from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

    savePath = os.path.join(args.output_dir, args.save_name)

    config = BertConfig.from_json_file(args.config_file)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info(
        "Evaluating Tiger with device: {} n_gpu: {}".format(
            device, n_gpu)
    )

    default_gpu = True

    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
        = LoadDatasetTest(args, task_cfg, args.tasks.split('-'))


    num_labels = max([dataset.num_labels for dataset in task_datasets_val.values()])

    model_tagger_del = VILBertForVLTasks.from_pretrained(
            args.from_pretrained_tagger_del, config, num_labels=num_labels, default_gpu=default_gpu
        )

    model_tagger_add = VILBertForVLTasks.from_pretrained(
            args.from_pretrained_tagger_add,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    model_inserter = VILBertForVLTasks.from_pretrained(
            args.from_pretrained_inserter,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    model_tagger_del.to(device)
    model_tagger_add.to(device)
    model_inserter.to(device)

    print("  Num Iters: ", task_num_iters)
    print("  Batch size: ", task_batch_size)

    model_tagger_del.eval()
    model_tagger_add.eval()
    model_inserter.eval()

    # when run evaluate, we run each task sequentially. here we only have one task
    for task_id in task_ids:

        edit_round = args.edit_round

        results = [[] for _ in range(edit_round)]

        for i, batch in enumerate(tqdm(task_dataloader_val[task_id])):

            results = EvaluatingTiger(
                device,
                batch,
                model_tagger_del,
                model_tagger_add,
                model_inserter,
                results,
                edit_round,
            )

        if args.split:
            json_path = os.path.join(savePath, args.split)
        else:
            json_path = os.path.join(savePath, task_cfg[task_id]["test_split"])

        for i in range(edit_round):

            json.dump(results[i], open(json_path + "_round_" + str(i + 1) + "_result.json", "w"))

            print('Evaluating Metrics for Editing Round ', i+1)

            gen = {}
            gts = {}
            for j, res in enumerate(results[i]):

                gts[j] = [res['gt_cap']]
                gen[j] = [res['caption']]

            gts_t = PTBTokenizer.tokenize(gts)
            gen_t = PTBTokenizer.tokenize(gen)

            val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
            method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
            for metric, score in zip(method, val_bleu):
                print(metric, score)

            # val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
            # print('METEOR', val_meteor)

            # val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
            # print('ROUGE_L', val_rouge)

            val_cider, _ = Cider().compute_score(gts_t, gen_t)
            print('CIDEr', val_cider)

            # val_spice, _ = Spice().compute_score(gts_t, gen_t)
            # print('SPICE', val_spice)


if __name__ == "__main__":
    main()
