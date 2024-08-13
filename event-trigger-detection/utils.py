import os

from pyvi import ViPosTagger

import torch
from torch.utils.data import Dataset

word_questions = [['sự_kiện'], ['hành_động'], ['sự_kiện', 'gì', 'xảy_ra', '?'], ['động_từ']]
syllable_questions = [['sự', 'kiện'], ['hành', 'động'], ['sự', 'kiện', 'gì', 'xảy', 'ra', '?'], ['động', 'từ']]

def read_data(txt_file, ann_file):
    txt_reader = open(txt_file, 'r', encoding='utf-8')
    txt = txt_reader.read().replace(u'\xa0', u' ')
    sents = txt.split('\n')[:-1]
    
    ann_reader = open(ann_file, 'r', encoding='utf-8')
    anns = ann_reader.read().split('\n')[:-1]
    
    txt_reader.close()
    ann_reader.close()
    
    data = []
    
    ann_dict = dict()
    for ann in anns:
        items = ann.split('\t')
        key = items[0]
        ann_dict[key] = items[1:]
    
    start_sent = -1
    end_sent = -1
    for sent in sents:
        start_sent = end_sent + 1
        end_sent = start_sent + len(sent)
        
        tmp = dict()
        tmp['text'] = sent
        tmp['annotation'] = []
        
        for key in ann_dict:
            if key[0] == 'E':
                args = ann_dict[key][0].split()
                event, event_key = args[0].split(':')[0:2]
                start_trigger, end_trigger = ann_dict[event_key][0].split()[1:3]
                start_trigger = int(start_trigger)
                end_trigger = int(end_trigger)
                if start_trigger >= start_sent and end_trigger <= end_sent:
                    tmp_ann = dict()
                    
                    trigger = ann_dict[event_key][1]
                    
                    tmp_ann['event'] = event
                    tmp_ann['trigger'] = trigger
                    tmp_ann['start_trigger'] = start_trigger
                    tmp_ann['end_trigger'] = end_trigger
                    # if trigger != txt[start_trigger:end_trigger]:
                    #     print(txt_file + '\t' + trigger + '\t' + txt[start_trigger:end_trigger])
                    #     continue
                    tmp_ann['argument'] = []
                    
                    for i in range(1, len(args)):
                        tmp_arg = dict()
                        type, type_key = args[i].split(':')[0:2]
                        arg_text = ann_dict[type_key][1]
                        entity, start_arg, end_arg = ann_dict[type_key][0].split()[0:3]
                        
                        tmp_arg['text'] = arg_text
                        tmp_arg['type'] = type.split('-')[0]
                        tmp_arg['entity'] = entity
                        tmp_arg['start_arg'] = int(start_arg)
                        tmp_arg['end_arg']= int(end_arg)
                        if tmp_arg['text'] != arg_text:
                            print(txt_file + '\t' + arg_text + '\t' + tmp_arg['text'])
                            continue
                        tmp_ann['argument'].append(tmp_arg)
                    
                    tmp['annotation'].append(tmp_ann)
                else:
                    continue
        data.append(tmp)

    l = 0
    for i in range(len(data)):
        if len(data[i]['annotation']) > 0:
            for j in range(len(data[i]['annotation'])):
                data[i]['annotation'][j]['start_trigger'] -= l
                data[i]['annotation'][j]['end_trigger'] -= l
                for k in range(len(data[i]['annotation'][j]['argument'])):
                    data[i]['annotation'][j]['argument'][k]['start_arg'] -= l
                    data[i]['annotation'][j]['argument'][k]['end_arg'] -= l
        l += len(data[i]['text']) + 1

    return data


def read_dataset(folder):
    res = []
    files = os.listdir(folder)
    files = sorted(files)
    for file in files:
        if file.endswith('.txt'):
            txt_file = file
            ann_file = file[:-4] + '.ann'
            tmp = read_data(os.path.join(folder, txt_file), os.path.join(folder, ann_file))
            res.extend(tmp)
    return res

def convert_data_to_list(data, level):
    sent = data['text']
    events_of_sent = []
    arguments_of_sent = []
   
    sent = sent.split(' ')
    if level == 'syllable':
        new_sent = []
        for word in sent:
            syllables = word.split('_')
            for syllable in syllables:
                new_sent.append(syllable)
        sent = new_sent
 
    anns = data['annotation']
    for ann in anns:
        start_trigger = ann['start_trigger']
        end_trigger = ann['end_trigger']
        event = ann['event']
        event_tmp = [event]
        start = -1
        end = -1
        event_valid = False
        for i, word in enumerate(sent):
            start = end + 1
            end = start + len(word)
           
            if start == start_trigger:
                event_tmp.append(i)
            if end == end_trigger:
                event_tmp.append(i)

        if len(event_tmp) == 3:
            events_of_sent.append(tuple(event_tmp))
            event_valid = True

        args = ann['argument']
        for arg in args:
            start_arg = arg['start_arg']
            end_arg = arg['end_arg']
            type = arg['type']

            argument_tmp = [type]
            start = -1
            end = -1
            for i, word in enumerate(sent):
                start = end + 1
                end = start + len(word)

                if start == start_arg:
                    argument_tmp.append(i)
                if end == end_arg:
                    argument_tmp.append(i)

            if len(argument_tmp) == 3:
                event_tmp.append(tuple(argument_tmp))

        if event_valid:
            arguments_of_sent.append(tuple(event_tmp))

    return sent, events_of_sent, arguments_of_sent

def convert_data_to_list_with_postag(data):
    sent = data['text']
    events_of_sent = []
    arguments_of_sent = []

    sent, postags_of_sent = ViPosTagger.postagging(sent)
    assert len(sent) == len(postags_of_sent)
 
    anns = data['annotation']
    for ann in anns:
        start_trigger = ann['start_trigger']
        end_trigger = ann['end_trigger']
        event = ann['event']
        event_tmp = [event]
        start = -1
        end = -1
        event_valid = False
        for i, word in enumerate(sent):
            start = end + 1
            end = start + len(word)
           
            if start == start_trigger:
                event_tmp.append(i)
            if end == end_trigger:
                event_tmp.append(i)

        if len(event_tmp) == 3:
            events_of_sent.append(tuple(event_tmp))
            event_valid = True

        args = ann['argument']
        for arg in args:
            start_arg = arg['start_arg']
            end_arg = arg['end_arg']
            type = arg['type']

            argument_tmp = [type]
            start = -1
            end = -1
            for i, word in enumerate(sent):
                start = end + 1
                end = start + len(word)

                if start == start_arg:
                    argument_tmp.append(i)
                if end == end_arg:
                    argument_tmp.append(i)

            if len(argument_tmp) == 3:
                event_tmp.append(tuple(argument_tmp))

        if event_valid:
            arguments_of_sent.append(tuple(event_tmp))

    return sent, postags_of_sent, events_of_sent, arguments_of_sent

def convert_dataset_to_list(folder, level="word", with_postag=True):
    sents = []
    postags_of_sents = []
    events_of_sents = []
    arguments_of_sents = []
    max_length = 0

    data = read_dataset(folder)
    for d in data:
        if with_postag:
            sent, postags_of_sent, events_of_sent, arguments_of_sent = convert_data_to_list_with_postag(d)
            postags_of_sents.append(postags_of_sent)
        else:
            sent, events_of_sent, arguments_of_sent = convert_data_to_list(d, level)
        max_length = max(max_length, len(sent))
        sents.append(sent)
        events_of_sents.append(events_of_sent)
        arguments_of_sents.append(arguments_of_sent)
    
    if with_postag:
      return sents, postags_of_sents, events_of_sents, arguments_of_sents, max_length
    else:
      return sents, events_of_sents, arguments_of_sents, max_length

def build_vocab_word(sents):
    word2id = {'<pad>': 0, '<unk>': 1}
    id_ = len(word2id)
    for sent in sents:
        for word in sent:
            if word not in word2id:
                word2id[word] = id_
                id_ += 1
    
    id2word = []
    for key in word2id:
        id2word.append(key)
    
    return word2id, id2word

def build_vocab_postag(postags_of_sents):
    postag2id = {'O': 0}
    id_ = len(postag2id)
    for postags_of_sent in postags_of_sents:
        for postag in postags_of_sent:
            if postag not in postag2id:
                postag2id[postag] = id_
                id_ += 1
    
    id2postag = []
    for key in postag2id:
        id2postag.append(key)
    
    return postag2id, id2postag


def build_vocab_event(events_of_sents):
    event2id = {'O': 0}
    id_ = len(event2id)
    for events_of_sent in events_of_sents:
        for event_tuple in events_of_sent:
            event_type = event_tuple[0]
            if event_type not in event2id:
                event2id[event_type] = id_
                id_ += 1
    
    id2event = []
    for key in event2id:
        id2event.append(key)

    return event2id, id2event

def build_vocab_argument(arguments_of_sents):
    argument2id = {'O': 0}
    id_ = len(argument2id)
    for arguments_of_sent in arguments_of_sents:
        for argument_event_tuple in arguments_of_sent:
            for argument_tuple in argument_event_tuple[3:]:
                argument_type = argument_tuple[0]
                if argument_type not in argument2id:
                    argument2id[argument_type] = id_
                    id_ += 1

    id2argument = []
    for key in argument2id:
        id2argument.append(key)

    return argument2id, id2argument

class EventWordDataset(Dataset):
    def __init__(self, sents, events_of_sents, postags_of_sents,
                word2id, event2id, postag2id, window_size):

        data = []
        for i, sent in enumerate(sents):
            events_of_sent = events_of_sents[i]
            postags_of_sent = postags_of_sents[i]

            for pos, word in enumerate(sent):
                input_ids = []
                position_ids = []
                postag_ids = []
                for j in range(pos-window_size, pos+window_size+1):
                    if j < 0 or j >= len(sent):
                        input_ids.append(word2id['<pad>'])
                        postag_ids.append(postag2id['O'])
                    else:
                        if sent[j] in word2id:
                            input_ids.append(word2id[sent[j]])
                        else:
                            input_ids.append(word2id['<unk>'])
                        postag_ids.append(postag2id[postags_of_sent[j]])
                    position_ids.append(abs(j-pos))
                
                event_id = event2id['O']
                for event_tuple in events_of_sent:
                    start_trigger = event_tuple[1]
                    end_trigger = event_tuple[2]
                    event_type = event_tuple[0]
                    if pos >= start_trigger and pos <= end_trigger:
                        event_id = event2id[event_type]
                
                tmp = dict()
                tmp['input_ids'] = torch.tensor(input_ids)
                tmp['position_ids'] = torch.tensor(position_ids)
                tmp['postag_ids'] = torch.tensor(postag_ids)
                tmp['event_id'] = torch.tensor(event_id)
                tmp['sentence_id'] = torch.tensor(i)
                tmp['current_position_id'] = torch.tensor(pos)
                data.append(tmp)
        
        self.data = data
        self.sents = sents
        self.events_of_sents = events_of_sents
        self.postags_of_sents = postags_of_sents
        self.word2id = word2id
        self.event2id = event2id
        self.postag2id = postag2id
        self.window_size = window_size
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class EventSentenceDataset(Dataset):
    def __init__(self, sents, events_of_sents, postags_of_sents,
                word2id, event2id, postag2id, max_length):

        data = []
        for i, sent in enumerate(sents):
            events_of_sent = events_of_sents[i]
            postags_of_sent = postags_of_sents[i]
            
            input_ids = []
            postag_ids = []
            in_sentence = []
            for j in range(max_length):
                if j < len(sent):
                    if sent[j] in word2id:
                        input_ids.append(word2id[sent[j]])
                    else:
                        input_ids.append(word2id['<unk>'])
                    postag_ids.append(postag2id[postags_of_sent[j]])
                    in_sentence.append(1)
                else:
                    input_ids.append(word2id['<pad>'])
                    postag_ids.append(postag2id['O'])
                    in_sentence.append(0)
            
            event_ids = [event2id['O'] for _ in range(max_length)]
            for event_tuple in events_of_sent:
                event_type = event_tuple[0]
                start_trigger = event_tuple[1]
                end_trigger = event_tuple[2]
                for j in range(start_trigger, end_trigger+1):
                    event_ids[j] = event2id[event_type]
            
            tmp = dict()
            tmp['input_ids'] = torch.tensor(input_ids)
            tmp['postag_ids'] = torch.tensor(postag_ids)
            tmp['in_sentence'] = torch.tensor(in_sentence)
            tmp['event_ids'] = torch.tensor(event_ids)
            tmp['sentence_id'] = torch.tensor(i)
            data.append(tmp)
        
        self.data = data
        self.sents = sents
        self.events_of_sents = events_of_sents
        self.postags_of_sents = postags_of_sents
        self.word2id = word2id
        self.event2id = event2id
        self.postag2id = postag2id
        self.max_length = max_length
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class EventDataset(Dataset):
    def __init__(self, sents, events_of_sents, event2id, 
                tokenizer, level, nth_question, max_length):

        data = []
        for i, sent in enumerate(sents):
            offset_category = dict()
            events_of_sent = events_of_sents[i]
            for event_tuple in events_of_sent:
                event_type = event_tuple[0]
                start_trigger = event_tuple[1]
                end_trigger = event_tuple[2]
                for j in range(start_trigger, end_trigger+1):
                    offset_category[j] = event_type

            tokens = []
            token_type_ids = []
            in_sentence = []
            labels = []

            # add <s>
            tokens.append('<s>')
            token_type_ids.append(0)
            in_sentence.append(0)
            labels.append(event2id['O'])

            if nth_question is not None:
                # add question
                question = word_questions[nth_question]
                if level == 'syllable':
                    question = syllable_questions[nth_question]
                for j, token in enumerate(question):
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.append(sub_tokens[0])
                    token_type_ids.append(0)
                    in_sentence.append(0)
                    labels.append(event2id['O'])

                # add </s> and </s>
                tokens.extend(['</s>', '</s>'])
                token_type_ids.extend([0, 0])
                in_sentence.extend([0, 0])
                labels.extend([event2id['O'], event2id['O']])

            # add sentence
            for j, token in enumerate(sent):
                if token.strip() != '':
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.append(sub_tokens[0])
                    token_type_ids.append(0)
                    in_sentence.append(1)
                else:
                    tokens.append(tokenizer.unk_token_id)
                    token_type_ids.append(0)
                    in_sentence.append(1)
                
                if j in offset_category:
                    labels.append(event2id[offset_category[j]])
                else:
                    labels.append(event2id['O'])

            # add </s>
            tokens.append('</s>')
            token_type_ids.append(0)
            in_sentence.append(0)
            labels.append(event2id['O'])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)
                token_type_ids.append(0)
                in_sentence.append(0)
                labels.append(event2id['O'])

            assert len(input_ids) == max_length
            assert len(token_type_ids) == max_length
            assert len(in_sentence) == max_length
            assert len(attention_mask) == max_length
            assert len(labels) == max_length

            tmp = dict()
            tmp['input_ids'] = torch.tensor(input_ids)
            tmp['token_type_ids'] = torch.tensor(token_type_ids)
            tmp['in_sentence'] = torch.tensor(in_sentence)
            tmp['attention_mask'] = torch.tensor(attention_mask)
            tmp['labels'] = torch.tensor(labels)
            tmp['sentence_id'] = torch.tensor(i)

            data.append(tmp)

        self.data = data
        self.sents = sents
        self.events_of_sents = events_of_sents
        self.event2id = event2id 
        self.tokenizer = tokenizer
        self.level = level
        self.nth_question = nth_question
        self.max_length = max_length

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)