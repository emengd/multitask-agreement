import os
import pickle
from warnings import warn

import numpy as np

#import inflect

from keras.models import Model, load_model
from keras.layers import *
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

def load_base_model(dir_name):
    with open(dir_name + '/rest.pkl', 'rb') as pkl:
        model = pickle.load(pkl)
    model.model = load_model(dir_name + '/model.keras')
    return model

class BaseModel:

    def __init__(self):
        pass
    
    def save(self, dir_name, save_data=False):
        os.makedirs(dir_name, exist_ok=True)
        self.model.save(dir_name + '/model.keras')
        model_bkup = self.model
        self.model = None
        if not save_data:
            if hasattr(self, 'Xtrain'):
                Xtrain_bkup = self.Xtrain
                self.Xtrain = None
            if hasattr(self, 'Xval'):
                Xval_bkup = self.Xval
                self.Xval = None
        with open(dir_name + '/rest.pkl', 'wb') as pkl:
            pickle.dump(self, pkl)
        self.model = model_bkup
        if not save_data:
            if hasattr(self, 'Xtrain'):
                self.Xtrain = Xtrain_bkup
            if hasattr(self, 'Xval'):
                self.Xval = Xval_bkup
    
    def fit(self, Xtrain, Xval, nb_epoch=10, batch_size=32, early_stopping=False,
            verbose=0, save_dir=None, save_on_epoch=False, nb_worker=1):
        self.ntrain = len(Xtrain)
        self.nval = len(Xval)
        self.Xtrain = Xtrain
        if self.nval > 0:
            self.Xval = Xval
        
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping())
        if save_on_epoch:
            if save_dir is None:
                save_dir = 'rnd{:04d}'.format(np.random.randint(10000))
                warn('No save directory provided; saving to {}.'.format(save_dir))
            os.makedirs(save_dir, exist_ok=True)
            callbacks.append(ModelCheckpoint(save_dir+'/autosave.{epoch:03d}.keras'))
        gen = self._generator(self.Xtrain, batch_size)
        gen_val = self._generator(self.Xval, batch_size)
        results = self.model.fit_generator(gen, self.ntrain, nb_epoch=nb_epoch, validation_data=gen_val,
            nb_val_samples=self.nval, nb_worker=nb_worker, pickle_safe=True, verbose=verbose,
            callbacks=callbacks, max_q_size=1)
        if save_dir is not None and not save_on_epoch:
            self.save(save_dir)
        return results
    
    def evaluate(self, X=None, batch_size=32, verbose=0, nb_worker=1):
        if X is None:
            X = self.Xval
        gen = self._generator(X, batch_size)
        results = self.model.evaluate_generator(gen, len(X), nb_worker=nb_worker, pickle_safe=True, max_q_size=1, verbose=verbose)
        return results
    
    def predict(self, X=None, batch_size=32, nb_worker=1):
        if X is None:
            X = self.Xval
        gen = self._predict_generator(X, batch_size)
        results = self.model.predict_generator(gen, len(X), nb_worker=nb_worker, pickle_safe=True, max_q_size=1)
        return results
        
    def train_on_batch(self, X):
        data = self._preprocess(X)
        if len(data) > 2:
            return self.model.train_on_batch(data[0], data[1], data[2])
        else:
            return self.model.train_on_batch(data[0], data[1])
     
    def test_on_batch(self, X):
        data = self._preprocess(X)
        if len(data) > 2:
            return self.model.test_on_batch(data[0], data[1], data[2])
        else:
            return self.model.test_on_batch(data[0], data[1])
    
    def _generator(self, X=None, batch_size=32):
        if X is None:
            X = self.Xtrain
        def gen():
            cur = 0
            while True:
                if cur >= len(X):
                    cur = 0
                res = self._preprocess(X[cur:cur+batch_size])
                cur += batch_size
                yield res
        return gen()
    
    def _predict_generator(self, X=None, batch_size=32, acc=None):
        if X is None:
            X = self.Xtrain
        cur = 0
        while True:
            if cur >= len(X):
                cur = 0
            res = self._preprocess(X[cur:cur+batch_size])
            cur += batch_size
            if acc is not None and len(res) > 2:
                acc.append(res[2])
            yield res[0]
        
    
class LanguageModel(BaseModel):

    def __init__(self, nwords, maxlen, state_size, id2word, word2id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = Input(shape=(maxlen,), dtype='int32')
        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen, name='embedding')(input)
        rep = LSTM(state_size, input_length=maxlen, return_sequences=True, name='representation')(embedding)
        lm = Convolution1D(nwords+1, 1, name='lm')(rep)
        lm_p = Activation('softmax', name='lm_p')(lm)
        self.model = Model(input=input, output=lm_p)
        self.model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', sample_weight_mode='temporal')
        self.maxlen = maxlen
        self.id2word = id2word
        self.word2id = word2id
    
    def _preprocess(self, X):
        aux = np.zeros((len(X), self.maxlen+1))
        w = np.zeros((len(X), self.maxlen))
        for i, s in enumerate(X):
            tokens = np.asarray([self.word2id[w]+1 for w in s['word']] + [0])
            aux[i, -len(tokens):] = tokens
            w[i, -len(tokens):] = 1
        inp = aux[:, :-1]
        outplm = aux[:, 1:, None]
        if np.sum(w) < 1:
            w[0, 0] = K.epsilon()
        return inp, outplm, w

class AgreementLM(BaseModel):

    def __init__(self, nwords, maxlen, state_size, loss_weights, id2word, word2id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = Input(shape=(maxlen,), dtype='int32')
        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen, name='embedding')(input)
        rep = LSTM(state_size, input_length=maxlen, return_sequences=True, name='representation')(embedding)
        lm = Convolution1D(nwords+1, 1, name='lm')(rep)
        lm_p = Activation('softmax', name='lm_p')(lm)
        agreement = Convolution1D(1, 1, activation='sigmoid', name='agreement')(rep)
        self.model = Model(input=input, output=[lm_p, agreement])
        self.model.compile(optimizer='adagrad', loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
            loss_weights=list(np.asarray(loss_weights)/np.sum(loss_weights)), sample_weight_mode='temporal')
        self.maxlen = maxlen
        self.id2word = id2word
        self.word2id = word2id
    
    def _preprocess(self, X):
        aux = np.zeros((len(X), self.maxlen+1))
        w = [np.zeros((len(X), self.maxlen)) for i in range(2)]
        outagree = np.zeros((len(X), self.maxlen))
        for i, s in enumerate(X):
            tokens = np.asarray([self.word2id[w]+1 for w in s['word']] + [0])
            aux[i, -len(tokens):] = tokens
            w[0][i, -len(tokens):] = 1
            if np.sum(s['verb']) > 0:
                j = np.flatnonzero(s['verb'])[0]
                ind = -len(s) + j - 1
                outagree[i, ind] = s['pos'][j] == 'VBP'
                w[1][i, ind] = 1
        inp = aux[:, :-1]
        outplm = aux[:, 1:, None]
        outagree = outagree[:, :, None]
        for w0 in w:
            if np.sum(w0) < 1:
                w0[0, 0] = K.epsilon()
        return inp, [outplm, outagree], w

#class DoubleAgrLM(BaseModel):

#    def __init__(self, nwords, maxlen, state_size, loss_weights, id2word, word2id, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        input = Input(shape=(maxlen,), dtype='int32')
#        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen)(input)
#        rep = LSTM(state_size, input_length=maxlen, return_sequences=True)(embedding)
#        lm = Convolution1D(nwords+1, 1)(rep)
#        lm_p = Activation('softmax', name='lm_p')(lm)
#        agreement = Convolution1D(1, 1, activation='sigmoid', name='agr_verb')(rep)
#        agreement2 = Convolution1D(1, 1, activation='sigmoid', name='agr_refl')(rep)
#        self.model = Model(input=input, output=[lm_p, agreement, agreement2])
#        self.model.compile(optimizer='adagrad', loss=['sparse_categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
#            loss_weights=list(np.asarray(loss_weights)/np.sum(loss_weights)), sample_weight_mode='temporal')
#        self.maxlen = maxlen
#        self.id2word = id2word
#        self.word2id = word2id
##        self.inflect = inflect.engine()
#    
#    def _preprocess(self, X):
#        aux = np.zeros((len(X), self.maxlen+1))
#        w = [np.zeros((len(X), self.maxlen)) for i in range(3)]
#        outagree_vb = np.zeros((len(X), self.maxlen))
#        outagree_rf = np.zeros((len(X), self.maxlen))
#        for i, s in enumerate(X):
#            tokens = np.asarray([self.word2id[w]+1 for w in s['sentence'].split()] + [0])
#            aux[i, -len(tokens):] = tokens
#            w[0][i, -len(tokens):] = 1
#            for ind, pos in zip(s['verb_index'], s['verb_pos']):
#                outagree_vb[i, -len(tokens)+ind-1] = pos == 'VBP'
#                w[1][i, -len(tokens)+ind-1] = 1
#            for ind, number in zip(s['refl_index'], s['refl_number']):
#                outagree_rf[i, -len(tokens)+ind-1] = number == 'pl'
#                w[2][i, -len(tokens)+ind-1] = 1
#            
#        inp = aux[:, :-1]
#        outplm = aux[:, 1:, None]
#        outagree_vb = outagree_vb[:, :, None]
#        outagree_rf = outagree_rf[:, :, None]
#        for w0 in w:
#            if np.sum(w0) < 1:
#                w0[0, 0] = K.epsilon()
#        return inp, [outplm, outagree_vb, outagree_rf], w

class Supertagger(BaseModel):

    def __init__(self, nwords, ntags, maxlen, state_size, id2word, word2id, id2tag, tag2id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = Input(shape=(maxlen,), dtype='int32')
        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen, name='embedding')(input)
        rep_layer = LSTM(state_size, input_length=maxlen, return_sequences=True, name='representation')
        rep = rep_layer(embedding)
        supertags = Convolution1D(ntags, 1, name='supertags')(rep)
        supertags_p = Activation('softmax', name='supertags_p')(supertags)
        self.model = Model(input=input, output=supertags_p)
        self.model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', sample_weight_mode='temporal')
        self.maxlen = maxlen
        self.id2word = id2word
        self.word2id = word2id
        self.id2tag = id2tag
        self.tag2id = tag2id
    
    def _preprocess(self, X):
        toks = np.zeros((len(X), self.maxlen))
        tags = np.zeros_like(toks)
        wtags = np.zeros_like(toks)
        for i, s in enumerate(X):
            wtags[i, -len(s):] = 1
            for j in range(len(s)):
                toks[i, -len(s)+j] = self.word2id[s['word'][j]] + 1
                if s['tag'][j] != '_':
                    tags[i, -len(s)+j] = self.tag2id[s['tag'][j]]
                else:
                    wtags[i, -len(s)+j] = 0
        tags = tags[:, :, None]
        if np.sum(wtags) < 1:
            wtags[0, 0] = K.epsilon()
        return toks, tags, wtags

class AgrSupertagger(BaseModel):

    def __init__(self, nwords, ntags, maxlen, state_size, loss_weights, id2word, word2id, id2tag, tag2id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = Input(shape=(maxlen,), dtype='int32')
        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen, name='embedding')(input)
        rep_layer = LSTM(state_size, input_length=maxlen, return_sequences=True, name='representation')
        rep = rep_layer(embedding)
        supertags = Convolution1D(ntags, 1, name='supertags')(rep)
        supertags_p = Activation('softmax', name='supertags_p')(supertags)
        agreement = Convolution1D(1, 1, activation='sigmoid', name='agreement')(rep)
        self.model = Model(input=input, output=[supertags_p, agreement])
        self.model.compile(optimizer='adagrad', loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
            loss_weights=list(np.asarray(loss_weights)/np.sum(loss_weights)), sample_weight_mode='temporal')
        self.maxlen = maxlen
        self.id2word = id2word
        self.word2id = word2id
        self.id2tag = id2tag
        self.tag2id = tag2id
    
    def _preprocess(self, X):
        toks = np.zeros((len(X), self.maxlen))
        tags = np.zeros_like(toks)
        wtags = np.zeros_like(toks)
        wagr = np.zeros_like(toks)
        outagree = np.zeros((len(X), self.maxlen))
        for i, s in enumerate(X):
            wtags[i, -len(s):] = 1
            for j in range(len(s)):
                toks[i, -len(s)+j] = self.word2id[s['word'][j]] + 1
                if s['tag'][j] != '_' and len(s['tag'][j]) > 0:
                    tags[i, -len(s)+j] = self.tag2id[s['tag'][j]]
                else:
                    wtags[i, -len(s)+j] = 0
                if s['verb'][j] == 1 and j > 0:
                    ind = -len(s) + j - 1
                    outagree[i, ind] = s['pos'][j] == 'VBP'
                    wagr[i, ind] = 1
        outagree = outagree[:, :, None]
        tags = tags[:, :, None]
        for w0 in [wtags, wagr]:
            if np.sum(w0) < 1:
                w0[0, 0] = K.epsilon()
        return toks, [tags, outagree], [wtags, wagr]

class Agreement(BaseModel):

    def __init__(self, nwords, maxlen, state_size, id2word, word2id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input = Input(shape=(maxlen,), dtype='int32')
        embedding = Embedding(input_dim=nwords+1, output_dim=state_size, input_length=maxlen, name='embedding')(input)
        rep_layer = LSTM(state_size, input_length=maxlen, return_sequences=True, name='representation')
        rep = rep_layer(embedding)
        agreement = Convolution1D(1, 1, activation='sigmoid', name='agreement')(rep)
        self.model = Model(input=input, output=agreement)
        self.model.compile(optimizer='adagrad', loss='binary_crossentropy', sample_weight_mode='temporal')
        self.maxlen = maxlen
        self.id2word = id2word
        self.word2id = word2id
    
    def _preprocess(self, X):
        toks = np.zeros((len(X), self.maxlen))
        wagr = np.zeros_like(toks)
        outagree = np.zeros((len(X), self.maxlen))
        for i, s in enumerate(X):
            for j in range(len(s)):
                toks[i, -len(s)+j] = self.word2id[s['word'][j]] + 1
                if s['verb'][j] == 1 and j > 0:
                    ind = -len(s)+j-1
                    outagree[i, ind] = s['pos'][j] == 'VBP'
                    wagr[i, ind] = 1
        outagree = outagree[:, :, None]
        for w0 in [wagr]:
            if np.sum(w0) < 1:
                w0[0, 0] = K.epsilon()
        return toks, outagree, wagr


