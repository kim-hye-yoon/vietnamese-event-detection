import logging
import argparse
import sys
import time
import os
import pickle
import numpy as np

from utils import *
from models import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def make_prediction(inps, in_sentence, sentence_ids, id2event):
    
    inps = inps.tolist()
    in_sentence = in_sentence.tolist()
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
    
    log_folder = 'log/bilstm'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    weight_folder = 'weight/bilstm'
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    out_folder = 'out/bilstm'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    log_file = f'bilstm-event-sentence-{args.use_pretrained}-word2vec-{args.use_postag}-postag.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, log_file)))

    logger.info(str(args))

    train_sents, train_postags_of_sents, train_events_of_sents, train_arguments_of_sents, train_max_length \
        = convert_dataset_to_list('data/train')
    dev_sents, dev_postags_of_sents, dev_events_of_sents, dev_arguments_of_sents, dev_max_length \
        = convert_dataset_to_list('data/dev')
    test_sents, test_postags_of_sents, test_events_of_sents, test_arguments_of_sents, test_max_length \
        = convert_dataset_to_list('data/test')
    max_length = max(train_max_length, dev_max_length, test_max_length)
    
    word2id, id2word = build_vocab_word(train_sents)
    postag2id, id2postag = build_vocab_postag(train_postags_of_sents)
    event2id, id2event = build_vocab_event(train_events_of_sents)
    
    train_data = EventSentenceDataset(train_sents, train_events_of_sents, train_postags_of_sents, \
        word2id, event2id, postag2id, max_length)
    dev_data = EventSentenceDataset(dev_sents, dev_events_of_sents, dev_postags_of_sents, \
        word2id, event2id, postag2id, max_length)
    test_data = EventSentenceDataset(test_sents, test_events_of_sents, test_postags_of_sents, \
        word2id, event2id, postag2id, max_length)

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dataset=dev_data, batch_size=args.eval_batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)

    lr = args.lr

    embedding_weight = None
    if args.use_pretrained:
        embedding_weight = np.load('weight/word2vec.npy')
        embedding_weight = torch.FloatTensor(embedding_weight)

    if not args.use_postag:
        model = EventSentenceBiLSTM(
            num_labels=len(event2id),
            num_words=len(word2id),
            word_embedding_dim=args.word_embedding_dim,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            use_pretrained=args.use_pretrained,
            embedding_weight=embedding_weight
        )
    else:
        model = EventSentenceBiLSTMWithPostag(
            num_labels=len(event2id),
            num_words=len(word2id),
            word_embedding_dim=args.word_embedding_dim,
            num_postags=len(postag2id),
            postag_embedding_dim=args.postag_embedding_dim,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            use_pretrained=args.use_pretrained,
            embedding_weight=embedding_weight
        )
    model.to(device)

    if not args.no_train:
        optimizer = Adam(model.parameters(), lr=lr)

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
            for (batch, data) in enumerate(train_data_loader):
                input_ids = data['input_ids'].to(device)
                postag_ids = data['postag_ids'].to(device)
                in_sentence = data['in_sentence'].to(device)
                labels = data['event_ids'].to(device)
                sentence_ids = data['sentence_id'].to(device)

                if not args.use_postag:
                    loss, logits = model(input_ids, in_sentence, labels)
                else:
                    loss, logits = model(input_ids, postag_ids, in_sentence, labels)
                
                train_loss += loss.item()
                nb_train_steps += 1

                loss.backward()
                optimizer.step()
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
            for (batch, data) in enumerate(dev_data_loader):
                input_ids = data['input_ids'].to(device)
                postag_ids = data['postag_ids'].to(device)
                in_sentence = data['in_sentence'].to(device)
                labels = data['event_ids'].to(device)
                sentence_ids = data['sentence_id'].to(device)

                with torch.no_grad():
                    if not args.use_postag:
                        loss, logits = model(input_ids, in_sentence, labels)
                    else:
                        loss, logits = model(input_ids, postag_ids, in_sentence, labels)
                    
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
                torch.save(model.state_dict(), os.path.join(weight_folder, f'bilstm-event-sentence-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pth'))
                logger.info(f'Save model weight!')
        
            logger.info('')
    
    # result on test
    model.load_state_dict(torch.load(os.path.join(weight_folder, f'bilstm-event-sentence-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pth'), map_location=device))
    logger.info('Restore best model !')
    
    list_logits = []
    list_in_sentence = []
    list_sentence_ids = []

    model.eval()
    for (batch, data) in enumerate(test_data_loader):
        input_ids = data['input_ids'].to(device)
        postag_ids = data['postag_ids'].to(device)
        in_sentence = data['in_sentence'].to(device)
        labels = data['event_ids'].to(device)
        sentence_ids = data['sentence_id'].to(device)

        with torch.no_grad():
            if not args.use_postag:
                logits = model(input_ids, in_sentence)
            else:
                logits = model(input_ids, postag_ids, in_sentence)

        list_logits.append(logits)
        list_in_sentence.append(in_sentence)
        list_sentence_ids.append(sentence_ids)

    logits = torch.cat(list_logits, dim=0)
    in_sentence = torch.cat(list_in_sentence, dim=0)
    sentence_ids = torch.cat(list_sentence_ids, dim=0)
    
    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction(preds, in_sentence, sentence_ids, id2event)
    labels = test_events_of_sents
    test_f1, test_precision, test_recall = evaluate(preds, labels)

    logger.info(f'Test: f1 {test_f1:.2f} precision {test_precision:.2f} recall {test_recall:.2f}')
    
    with open(os.path.join(out_folder, f'bilstm-event-sentence-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pkl'), 'wb') as f:
        pickle.dump(preds, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--word_embedding_dim', default=300, type=int)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--use_postag', action='store_true')
    parser.add_argument('--postag_embedding_dim', default=25, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)

    args = parser.parse_args()

    main(args)