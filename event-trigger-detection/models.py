import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class EventWordCNN(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, window_size, position_embedding_dim,
                kernel_sizes, num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(EventWordCNN, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.window_size = window_size
        self.position_embedding_dim = position_embedding_dim  
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=position_embedding_dim)
        self.embedding_dim = word_embedding_dim + position_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, position_ids, labels=None):
        
        word_emb = self.word_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        emb = torch.cat((word_emb, position_emb), 2)
        
        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        


class EventWordCNNWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, window_size, 
                position_embedding_dim, num_postags, postag_embedding_dim, kernel_sizes, 
                num_filters, dropout_rate, use_pretrained=False, embedding_weight=None):
        
        super(EventWordCNNWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.window_size = window_size
        self.position_embedding_dim = position_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim  
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=position_embedding_dim)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)    
        self.embedding_dim = word_embedding_dim + position_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, [kernel_size, self.embedding_dim])
                                    for kernel_size in kernel_sizes])

        self.linear = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, position_ids, postag_ids, labels=None):
        
        word_emb = self.word_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, position_emb, postag_emb), 2)

        emb = torch.unsqueeze(emb, 1)

        cnns = []
        for conv in self.convs:
            cnn = F.relu(conv(emb))
            cnn = torch.squeeze(cnn, -1)
            cnn = F.max_pool1d(cnn, cnn.size(2))
            cnns.append(cnn)
        
        cnn = torch.cat(cnns, 2)
        flat = cnn.view(cnn.size(0), -1)
        flat = self.dropout(flat)
        logits = self.linear(flat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class EventSentenceBiLSTM(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, hidden_size,
                dropout_rate, use_pretrained=False, embedding_weight=None):

        super(EventSentenceBiLSTM, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.embedding_dim = word_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        emb = word_emb
        # emb = torch.cat((word_emb, entity_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class EventSentenceBiLSTMWithPostag(nn.Module):
    def __init__(self, num_labels, num_words, word_embedding_dim, num_postags, postag_embedding_dim, \
                hidden_size, dropout_rate, use_pretrained=False, embedding_weight=None):

        super(EventSentenceBiLSTMWithPostag, self).__init__()

        self.num_labels = num_labels
        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.num_postags = num_postags
        self.postag_embedding_dim = postag_embedding_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if not use_pretrained:
            self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_weight, freeze=False)
        self.postag_embedding = nn.Embedding(num_embeddings=num_postags, embedding_dim=postag_embedding_dim)
        self.embedding_dim = word_embedding_dim + postag_embedding_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, postag_ids, attention_mask=None, labels=None):

        word_emb = self.word_embedding(input_ids)
        postag_emb = self.postag_embedding(postag_ids)
        emb = torch.cat((word_emb, postag_emb), 2)

        hidden_states, _ = self.bilstm(emb)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


class EventClassifyBert(nn.Module):
    def __init__(self, bert, num_labels):
        super(EventClassifyBert, self).__init__()

        self.config = bert.config
        self.num_labels = num_labels
        self.bert = bert
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        sequence_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits