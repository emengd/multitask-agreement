import glob
import re
from collections import Counter, defaultdict

import numpy as np

def tsv_to_numpy(data):
    def aux(s):
        words = s['orig_sentence'].split()
        pos = s['pos_sentence'].split()
        n = len(words)
        res = np.zeros(n, dtype=np.dtype([('word', 'O'), ('pos', 'O'), ('verb', 'int'), ('subj', 'int')]))
        res['word'][:] = words
        res['pos'][:] = pos
        res['subj'][s['subj_index']-1] = 1
        res['verb'][s['verb_index']-1] = 1
        return res
    return [aux(s) for s in data]

def build_lexicon(data, key='word'):
    lexicon = Counter()
    for s in data:
        for w in s[key]:
            lexicon[w] += 1
    return lexicon

def build_vocab(data = None, thres = 0, nwords = None, lexicon = None, key='word', verbose=False):
    if lexicon is None:
        lexicon = build_lexicon(data, key)
    aux = lexicon.most_common()
    freqs = np.asarray([p[1] for p in aux], dtype='int')
    words = [p[0] for p in aux]
    if nwords is None:
        nwords = np.sum(freqs >= thres)
    if verbose:
        print('Removing {:d} tokens out of {:d} accounting for {:04.1f}% of occurrences.'.format(
            len(lexicon)-nwords, len(lexicon), 100 * np.sum(freqs[nwords:]) / np.sum(freqs))
    vocab = set(words[:nwords])
    return vocab

def apply_threshold_pos(data, thres = 0, nwords = None, vocab = None, lexicon = None, verbose=False):
    if vocab is None:
        vocab = build_vocab(data, lexicon=lexicon, thres=thres, nwords=nwords, verbose=verbose)
    nwords = len(vocab)
    new_vocab = vocab.copy()
    
    def aux(s):
        new_s = s.copy()
        for i in range(len(s)):
            if s['word'][i] not in vocab:
                new_s['word'][i] = s['pos'][i]
                new_vocab.add(s['pos'][i])
        return new_s
    
    res = [aux(s) for s in data]
    return (res, new_vocab)

def apply_threshold_void(data, thres = 0, nwords = None, key='tag', vocab = None, lexicon = None, verbose=False):
    if vocab is None:
        vocab = build_vocab(data, lexicon=lexicon, key=key, thres=thres, nwords=nwords, verbose=verbose)
    nwords = len(vocab)
    new_vocab = vocab.copy()
    added = False
    
    def aux(s):
        nonlocal added
        new_s = s.copy()
        for i in range(len(s)):
            if s[key][i] not in vocab:
                new_s[key][i] = '_'
                added = True
        return new_s
    
    res = [aux(s) for s in data]
    if added:
        new_vocab.add('_')
    return (res, new_vocab)

def build_dicts(vocab):
    id = 0
    tok2id = dict()
    for tok in vocab:
        tok2id[tok] = id
        id += 1
    id2tok = np.zeros(len(tok2id), dtype='O')
    for tok, i in tok2id.items():
        id2tok[i] = tok
    return (id2tok, tok2id)

def apply_length_threshold(data, thres, verbose=False):
    res = [s for s in data if len(s) <= thres]
    if verbose:
        print('Removing {} out of {} data points ({:04.1f}%).'.format(len(data)-len(res), len(data),
            100*(len(data)-len(res))/len(data)))
    return res

def extract_ccg(folder):
    nsections = 25
    leafregex = re.compile(r'<L([^>]*)>')
    sections = [dict() for i in range(nsections)]
    for i in range(nsections):
        subfolder = '{}/{:02d}'.format(folder, i)
        files = glob.glob(subfolder + '/*.auto')
        for f in files:
            with open(f, 'r') as inp:
                for line in inp:
                    if line.startswith('ID='):
                        id = line.split()[0][3:]
                        continue
                    words = [l.strip().split() for l in leafregex.findall(line)]
                    sentence = np.zeros(len(words), dtype=[('word', 'O'), ('pos', 'O'),
                        ('tag', 'O'), ('subj', 'int'), ('verb', 'int')])
                    sentence['word'][:] = [w[3] for w in words]
                    sentence['tag'][:] = [w[1] for w in words]
                    sentence['pos'][:] = [w[0] for w in words]
                    sections[i][id] = sentence
    return sections
                    
    


