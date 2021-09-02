# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling_roberta_subdict import RobertaForTokenClassification_v2
from data_utils import load_and_cache_examples_subdict, get_labels, OurRandomSampler
from model_utils import multi_source_label_refine, soft_frequency, mt_update, get_mt_loss, opt_grad
from eval import evaluate

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification_v2, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, model_class, config, t_total, epoch):

    models = [model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    ) for _ in range(args.num_head-1)]
    
    for model in models:
        model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizers = []
    schedulers = []
    for model in models:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
                eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    for model in models:
        model.zero_grad()
    return models, optimizers, schedulers

def train(args, train_datasets, model_class, config, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    o_label = {key:idx for idx, key in enumerate(labels)}['O']
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.saved_dir,'tfboard'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    n = len(train_datasets[0])
    shuffle_list = torch.randperm(n).tolist()
    train_dataloaders = []
    for idx in range(args.num_head-1):
        train_sampler = OurRandomSampler(train_datasets[idx], shuffle_list=shuffle_list, master=(idx==0)) if args.local_rank == -1 else DistributedSampler(train_datasets[idx])
        train_dataloader = DataLoader(train_datasets[idx], sampler=train_sampler, batch_size=args.train_batch_size)
        train_dataloaders.append(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    models, optimizers, schedulers = initialize(args, model_class, config, t_total, 0)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_datasets[0]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_losses, logging_losses = [0.0]* len(models), [0.0]*len(models)
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0, defaultdict(int), defaultdict(int), defaultdict(int)], [0, 0, 0, defaultdict(int), defaultdict(int), defaultdict(int)]
   
    self_training_teacher_models = models

    for epoch in train_iterator:
        epoch_iterators = []
        for idx in range(args.num_head-1):
            if idx==0:
                epoch_iterator = tqdm(train_dataloaders[idx], desc="Iteration", disable=args.local_rank not in [-1, 0])
            else:
                epoch_iterator = train_dataloaders[idx]
            epoch_iterators.append(epoch_iterator)

        # for step, batches in enumerate(zip(*epoch_iterators)):
        for step, batches in enumerate(zip(epoch_iterators[0], epoch_iterators[1], epoch_iterators[2], epoch_iterators[3])):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            for model in models:
                model.train()
            batches = [tuple(t.to(args.device) for t in batch) for batch in batches]
            positives = []
            unlabeleds = []
            # Update labels periodically after certain begin step
            if global_step >= args.self_training_begin_step:

                # Update a new teacher periodically
                delta = global_step - args.self_training_begin_step
                if delta % args.self_training_period == 0:
                    self_training_teacher_models = [copy.deepcopy(model) for model in models]
                    # self_training_teacher_model = copy.deepcopy(model)  # 这里对teacher model进行更新
                    for self_training_teacher_model in self_training_teacher_models:
                        self_training_teacher_model.eval()

                inputses = [{"input_ids": batch[0], "attention_mask": batch[1]} for batch in batches]
                with torch.no_grad():
                    outputses = [self_training_teacher_model(**inputs) 
                                    for self_training_teacher_model, inputs 
                                        in zip(self_training_teacher_models, inputses[1:])]

                label_mask = None
                if args.self_training_label_mode == "hard":
                    pred_labelses = []
                    label_masks = []
                    for batch, outputs in zip(batches, outputses):
                        pred_labels = torch.argmax(outputs[0], dim=2)
                        pred_labels, label_mask = multi_source_label_refine(args,batch[5],batch[3],pred_labels,pad_token_label_id,pred_logits=outputs[0])
                        pred_labelses += [pred_labels]
                        label_masks += [label_mask]
                elif args.self_training_label_mode == "soft":
                    pred_labelses = []
                    label_masks = []
                    for batch, outputs in zip(batches, outputses):
                        pred_labels = soft_frequency(logits=outputs[0], power=2)
                        pred_labels, label_mask = multi_source_label_refine(args,batch[5],batch[3],pred_labels,pad_token_label_id)
                        pred_labelses.append(pred_labels)
                        label_masks.append(label_mask)
                inputses = []
                for pred_labels, label_mask, batch in zip(pred_labelses, label_masks, batches):
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": pred_labels, "label_mask": label_mask}
                    inputses.append(inputs)
            else:
                inputses = []
                for batch in batches:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    positive = (inputs['labels']!=pad_token_label_id) & (inputs['labels'] != 0)
                    unlabeled = (inputs['labels']!=pad_token_label_id) & (inputs['labels'] == 0)
                    positives.append(positive)
                    unlabeleds.append(unlabeled)
                    inputses.append(inputs)
 
            for i in range(len(positives)):
                pi = positives[i]
                ui = unlabeleds[i]
                tmpp, tmpu = None, None
                for j in range(len(positives)):
                    if i==j: continue
                    tmpp = (ui & positives[j]) if tmpp is None else (tmpp | (ui & positives[j]))
                    tmpu = (pi & unlabeleds[j]) if tmpu is None else (tmpu | (pi & unlabeleds[j]))
                    curp = (ui & positives[j])
                    inputses[i]['labels'][curp] = inputses[j]['labels'][curp]

            losses = []
            logitses = []
            final_embedses = []
            for model, inputs in zip(models, inputses):
                outputs = model(**inputs)
                loss, logits, final_embeds = outputs[0], outputs[1], outputs[2] # model outputs are always tuple in pytorch-transformers (see doc)
                losses += [loss]
                logitses += [logits]
                final_embedses += [final_embeds]
            
            if args.lamb != 0:
                for i in range(len(losses)):
                    rep = final_embedses[i]
                    reg = 0.0
                    for j in range(len(losses)):
                        if i==j: continue
                        reg += ((rep-final_embedses[j].detach())**2).mean()
                    losses[i] = losses[i] + args.lamb * reg

            if args.gradient_accumulation_steps > 1:
                for idx in range(len(losses)):
                    losses[idx] = losses[idx] / args.gradient_accumulation_steps

            if args.fp16:
                for idx in range(len(losses)):
                    with amp.scale_loss(losses[idx], optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                for loss in losses:
                    loss.backward()
            for idx, loss in enumerate(losses):
                tr_losses[idx] += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    for optimizer in optimizers:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    for model in models:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info("***** Entropy loss: {} *****".format(",".join(["{}".format(loss.item()) for loss in losses])) )
                        results, _, best_dev, _ = evaluate(args, models[args.num], tokenizer, labels, pad_token_label_id, best_dev, mode="dev", prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, is_updated  = evaluate(args, models[args.num], tokenizer, labels, pad_token_label_id, best_test, mode="test", prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)

                        saved_dirs = []
                        if args.local_rank in [-1, 0] and is_updated:
                            updated_self_training_teacher = True
                            saved_dirs.append(os.path.join(args.saved_dir, "checkpoint-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            saved_dirs.append(os.path.join(args.saved_dir, "checkpoint-{}".format(global_step)))

                        if len(saved_dirs) > 0:
                            for saved_dir in saved_dirs:
                                for idx, model in enumerate(models):
                                    saved_dir = os.path.join(args.saved_dir, f"{idx}")
                                    logger.info("Saving model checkpoint to %s", args.saved_dir)
                                    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                    # They can then be reloaded using `from_pretrained()`
                                    if not os.path.exists(saved_dir):
                                        os.makedirs(saved_dir)
                                
                                    model_to_save = (
                                        model.module if hasattr(model, "module") else model
                                    )  # Take care of distributed/parallel training
                                    model_to_save.save_pretrained(saved_dir)
                                    tokenizer.save_pretrained(saved_dir)
                                    torch.save(args, os.path.join(saved_dir, "training_args.bin"))
                                    torch.save(model.state_dict(), os.path.join(saved_dir, "model.pt"))
                                    torch.save(optimizer.state_dict(), os.path.join(saved_dir, "optimizer.pt"))
                                    torch.save(scheduler.state_dict(), os.path.join(saved_dir, "scheduler.pt"))
                                    logger.info("Saving optimizer and scheduler states to %s", saved_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return models, global_step, [tr_loss / global_step for tr_loss in tr_losses], best_dev, best_test

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--saved_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_saved_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # mean teacher
    parser.add_argument('--mt', type = int, default = 0, help = 'mean teacher.')
    parser.add_argument('--mt_updatefreq', type=int, default=1, help = 'mean teacher update frequency')
    parser.add_argument('--mt_class', type=str, default="kl", help = 'mean teacher class, choices:[smart, prob, logit, kl(default), distill].')
    parser.add_argument('--mt_lambda', type=float, default=1, help= "trade off parameter of the consistent loss.")
    parser.add_argument('--mt_rampup', type=int, default=300, help="rampup iteration.")
    parser.add_argument('--mt_alpha1', default=0.99, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_alpha2', default=0.995, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_beta', default=10, type=float, help="coefficient of mt_loss term.")
    parser.add_argument('--mt_avg', default="exponential", type=str, help="moving average method, choices:[exponentail(default), simple, double_ema].")
    parser.add_argument('--mt_loss_type', default="logits", type=str, help="subject to measure model difference, choices:[embeds, logits(default)].")

    # virtual adversarial training
    parser.add_argument('--vat', type = int, default = 0, help = 'virtual adversarial training.')
    parser.add_argument('--vat_eps', type = float, default = 1e-3, help = 'perturbation size for virtual adversarial training.')
    parser.add_argument('--vat_lambda', type = float, default = 1, help = 'trade off parameter for virtual adversarial training.')
    parser.add_argument('--vat_beta', type = float, default = 1, help = 'coefficient of the virtual adversarial training loss term.')
    parser.add_argument('--vat_loss_type', default="logits", type=str, help="subject to measure model difference, choices = [embeds, logits(default)].")

    # self-training
    parser.add_argument('--self_training_reinit', type = int, default = 0, help = 're-initialize the student model if the teacher model is updated.')
    parser.add_argument('--self_training_begin_step', type = int, default = 900, help = 'the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type = str, default = "hard", help = 'pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type = int, default = 878, help = 'the self-training period.')
    parser.add_argument('--self_training_hp_label', type = float, default = 0, help = 'use high precision label.')
    parser.add_argument('--self_training_ensemble_label', type = int, default = 0, help = 'use ensemble label.')

    parser.add_argument("--num_head", dest="num_head", type=int, default=5)
    parser.add_argument("--lamb", dest="lamb", type=float, default=1.0)
    parser.add_argument("--num", dest="num", type=int, default=0)
    args = parser.parse_args()

    if (
        os.path.exists(args.saved_dir)
        and os.listdir(args.saved_dir)
        and args.do_train
        and not args.overwrite_saved_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_saved_dir to overcome.".format(
                args.saved_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.saved_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.saved_dir)
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(args.saved_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    labels = get_labels(args.data_dir)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    best_test = None

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_datasets = load_and_cache_examples_subdict(args, tokenizer, labels, pad_token_label_id, mode="train")
        models, global_step, tr_losses, best_dev, best_test = train(args, train_datasets, model_class, config, tokenizer, labels, pad_token_label_id)
        tr_loss = sum(tr_losses) / len(tr_losses)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.saved_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.saved_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.saved_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if not best_dev:
            best_dev = [0, 0, 0]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, best_dev, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_dev, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.saved_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        # models_ = []
        # for idx in range(1): 
        #     saved_dir = os.path.join(args.saved_dir, f"{idx}")
        #     tokenizer = tokenizer_class.from_pretrained(saved_dir, do_lower_case=args.do_lower_case)
        #     model = model_class.from_pretrained(saved_dir)
        #     model.to(args.device)
        #     models_.append(model)
        saved_dir = os.path.join(args.saved_dir, f"{args.num}")
        tokenizer = tokenizer_class.from_pretrained(saved_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(saved_dir)
        model.to(args.device)
        if not best_test:
            best_test = [0, 0, 0]
        result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.saved_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.saved_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.json"), "r") as f:
                example_id = 0
                data = json.load(f)
                for item in data:
                    output_line = str(item["str_words"]) + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                    example_id += 1

    return results


if __name__ == "__main__":
    main()
