import sys
import torch

from layers.EmbeddingLayer import EmbeddingLayer
from layers.LSTMLayer import LSTMLayer
from layers.CharLSTMEncoder import CharLSTMEncoder
from layers.TimeConvLayer import TimeConvLayer
sys.path.append("..")


class DetectionModel(torch.nn.Module):

    def __init__(self, **kwargs):

        super(DetectionModel, self).__init__()

        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        self.word_embedding = EmbeddingLayer(dim=self.word_embedding_dim, trainable=self.embedding_trainable)
        self.word_embedding.load_from_pretrain(self.pretrain_file, self.word2id)

        self.char_encoder = CharLSTMEncoder(char_embedding_dim=self.char_embedding_dim,
                                            word_encoding_dim=self.word_encoding_dim,
                                            num_vocab=len(self.char2id))

        self.pos_embedding = EmbeddingLayer(dim=self.pos_embedding_dim, trainable=True)
        self.pos_embedding.initialize_with_random(len(self.pos2id))

        self.rnn_layer = LSTMLayer(D_in=self.word_embedding_dim + self.pos_embedding_dim + self.word_encoding_dim,
                                   D_out=self.hidden_dim,
                                   n_layers=1,
                                   dropout=0,
                                   bidirectional=True)
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate)

        self.dense_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.conv_layer = TimeConvLayer(self.hidden_dim, self.hidden_dim, 3)
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, do_softmax=False, **kwargs):

        words = kwargs["words"]
        poss = kwargs["poss"]
        seq_len = kwargs["seq_len"]
        chars = kwargs["chars"]
        char_len = kwargs["char_len"]

        embed_words = self.word_embedding(words)
        embed_poss = self.pos_embedding(poss)
        embed_chars = self.char_encoder(chars, char_len)
        concat_embedding = torch.cat((embed_words, embed_poss, embed_chars), dim=2)

        hidden, _ = self.rnn_layer(concat_embedding, seq_len, total_length=self.max_seq_len)

        hidden = torch.nn.functional.relu(self.dense_layer(hidden))
        hidden = self.dropout_layer(hidden)
        output = self.output_layer(hidden)

        if do_softmax:
            output = torch.nn.functional.softmax(output, dim=2)
        return output
