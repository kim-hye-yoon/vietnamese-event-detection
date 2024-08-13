import logging
import argparse
import sys
import time
import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from utils import *
from models import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# draw confusion matrix
def evaluate_cfm(inps, in_sentence, sentence_ids, id2event, labels):
    inps = inps.tolist()
    in_sentence.tolist()
    sentence_ids = sentence_ids.tolist()
    flatten_labels = []
    flatten_preds = []

    result = []
    
    for i, in_sent in enumerate(in_sentence):
        inp = inps[i]
        events_of_sent = []

        start_sent = 0
        while in_sent[start_sent] == 0 and start_sent+1 < len(in_sent):
            start_sent += 1
        end_sent = start_sent
        while in_sent[end_sent] == 1 and end_sent+1 < len(in_sent):
            end_sent += 1
        
        inp = np.array(inp[start_sent:end_sent])
        label = labels[i][start_sent:end_sent].detach().cpu().numpy()
        
        flatten_labels.extend(label)
        flatten_preds.extend(inp)
    
    assert len(flatten_labels) == len(flatten_preds)
    # print(flatten_labels)
    # print(flatten_preds)
    plot_confusion_matrix(flatten_labels, flatten_preds, id2event)
    

def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(18, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Atual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

def make_prediction(inps, in_sentence, sentence_ids, id2event):
    inps = inps.tolist()
    in_sentence.tolist()
    sentence_ids = sentence_ids.tolist()

    result = []
    
    for i, in_sent in enumerate(in_sentence):
        inp = inps[i]
        events_of_sent = []

        start_sent = 0
        while in_sent[start_sent] == 0 and start_sent+1 < len(in_sent):
            start_sent += 1
        end_sent = start_sent
        while in_sent[end_sent] == 1 and end_sent+1 < len(in_sent):
            end_sent += 1
        
        inp = inp[start_sent:end_sent]
        j = 0
        while j < len(inp):
            event_id = inp[j]
            event_type = id2event[event_id]
            if event_type != 'O':
                tmp = [event_type, j]
                while j+1 < len(inp) and inp[j+1] == inp[j]:
                    j += 1
                tmp.append(j)
                events_of_sent.append(tuple(tmp))
            j += 1

        result.append(events_of_sent)
    
    return result


def evaluate(preds, labels):

    num_labels = 0.0
    num_preds = 0.0
    num_true_positives = 0.0

    for i in range(len(preds)):
        label = labels[i]
        pred = preds[i]
        num_labels += len(label)
        num_preds += len(pred)
        for p in pred:
            for l in label:
                if p == l:
                    num_true_positives += 1
                    break

    f1, precision, recall = 0, 0, 0
    if num_preds != 0:
        precision = 100.0 * num_true_positives / num_preds
    else:
        precision = 0
    if num_labels != 0:
        recall = 100.0 * num_true_positives / num_labels
    else:
        recall = 0
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    log_folder = 'log/phobert'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    weight_folder = 'weight/phobert'
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    out_folder = 'out/phobert'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    log_file = f'phobert-{args.nth_question}-th-question.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, log_file)))

    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_sents, train_events_of_sents, _, train_max_length = convert_dataset_to_list('data/train', args.level, False)
    dev_sents, dev_events_of_sents, _, dev_max_length = convert_dataset_to_list('data/dev', args.level, False)
    test_sents, test_events_of_sents, _, test_max_length = convert_dataset_to_list('data/test', args.level, False)
    max_length = max(train_max_length, dev_max_length, test_max_length) + 20

    event2id, id2event = build_vocab_event(train_events_of_sents)

    train_data = EventDataset(train_sents, train_events_of_sents, event2id, \
        tokenizer, args.level, args.nth_question, max_length)
    dev_data = EventDataset(dev_sents, dev_events_of_sents, event2id, \
        tokenizer, args.level, args.nth_question, max_length)
    test_data = EventDataset(test_sents, test_events_of_sents, event2id, \
        tokenizer, args.level, args.nth_question, max_length)

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dataset=dev_data, batch_size=args.eval_batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)

    num_train_optimization_steps = len(train_data_loader) // args.gradient_accumulation_steps * args.num_epochs

    lr = args.lr
    bert = AutoModel.from_pretrained(args.model)
    model = EventClassifyBert(bert, len(event2id))
    model.to(device)

    if not args.no_train:
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(args.warmup_proportion * float(num_train_optimization_steps)),
            num_training_steps=num_train_optimization_steps
        )

        max_dev_f1 = 0
        for epoch in range(args.num_epochs):
            train_loss = 0
            train_f1 = 0
            train_precision = 0
            train_recall = 0
            nb_train_steps = 0

            start_time = time.time()

            model.train()
            logger.info(f'Epoch {epoch+1}|{args.num_epochs}:')
            for batch, data in enumerate(train_data_loader):
                input_ids = data['input_ids'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                in_sentence = data['in_sentence'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                sentence_ids = data['sentence_id'].to(device)

                loss, logits = model(input_ids, in_sentence, token_type_ids, labels)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_loss += loss.item()
                nb_train_steps += 1

                loss.backward()
                if (batch + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                preds = torch.argmax(logits, dim=-1)
                preds = make_prediction(preds, in_sentence, sentence_ids, id2event)
                labels = make_prediction(labels, in_sentence, sentence_ids, id2event)
                f1, precision, recall = evaluate(preds, labels)
                
                train_f1 += f1
                train_precision += precision
                train_recall += recall

                if batch % args.log_step == 0 or batch+1 == len(train_data_loader):
                    logger.info(f'Batch {batch+1}|{len(train_data_loader)}: loss {loss.item():.4f} f1 {f1:.2f} precision {precision:.2f} recall {recall:.2f}')

            logger.info(f'Train: loss {train_loss/nb_train_steps:.4f} f1 {train_f1/nb_train_steps:.2f} precision {train_precision/nb_train_steps:.2f} recall {train_recall/nb_train_steps:.2f}')
            logger.info(f'Time: {time.time() - start_time:.2f}')

            dev_loss = 0
            nb_dev_steps = 0
            list_logits = []
            list_in_sentence = []
            list_sentence_ids = []

            model.eval()
            for batch, data in enumerate(dev_data_loader):
                input_ids = data['input_ids'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                in_sentence = data['in_sentence'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                sentence_ids = data['sentence_id'].to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids, in_sentence, token_type_ids, labels)

                dev_loss += loss.item()
                nb_dev_steps += 1

                list_logits.append(logits)
                list_in_sentence.append(in_sentence)
                list_sentence_ids.append(sentence_ids)

            logits = torch.cat(list_logits, dim=0)
            in_sentence = torch.cat(list_in_sentence, dim=0)
            sentence_ids = torch.cat(list_sentence_ids, dim=0)
            
            preds = torch.argmax(logits, dim=-1)
            preds = make_prediction(preds, in_sentence, sentence_ids, id2event)
            labels = dev_events_of_sents
            dev_f1, dev_precision, dev_recall = evaluate(preds, labels)

            logger.info(f'Dev: loss {dev_loss/nb_dev_steps:.4f} f1 {dev_f1:.2f} precision {dev_precision:.2f} recall {dev_recall:.2f}')

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                torch.save(model.state_dict(), os.path.join(weight_folder, f'phobert-{args.nth_question}-th-question.pth'))
                logger.info(f'Save model weight!')

            logger.info('')

    # result on test
    model.load_state_dict(torch.load(os.path.join(weight_folder, f'phobert-{args.nth_question}-th-question.pth'), map_location=device))
    logger.info('Restore best model!')
    list_logits = []
    list_labels = []
    list_in_sentence = []
    list_sentence_ids = []

    model.eval()
    for batch, data in enumerate(test_data_loader):
        input_ids = data['input_ids'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        in_sentence = data['in_sentence'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        sentence_ids = data['sentence_id'].to(device)

        with torch.no_grad():
            logits = model(input_ids, in_sentence, token_type_ids)

        list_logits.append(logits)
        list_labels.append(labels)
        list_in_sentence.append(in_sentence)
        list_sentence_ids.append(sentence_ids)

    logits = torch.cat(list_logits, dim=0)
    labels = torch.cat(list_labels, dim=0)
    in_sentence = torch.cat(list_in_sentence, dim=0)
    sentence_ids = torch.cat(list_sentence_ids, dim=0)
    
    preds = torch.argmax(logits, dim=-1)
    
    # print(f"labels: {labels}")
    # print(f"preds: {preds}")
    evaluate_cfm(preds, in_sentence, sentence_ids, id2event, labels)
    
    preds = make_prediction(preds, in_sentence, sentence_ids, id2event)
    labels = test_events_of_sents
    test_f1, test_precision, test_recall = evaluate(preds, labels)

    logger.info(f'Test: f1 {test_f1:.2f} precision {test_precision:.2f} recall {test_recall:.2f}')

    with open(os.path.join(out_folder, f'phobert-{args.nth_question}-th-question.pkl'), 'wb') as f:
        pickle.dump(preds, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--level', default=None, type=str, required=True)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--nth_question', default=None, type=int)

    args = parser.parse_args()

    main(args)