This repository contains the code associated to the paper REF.

## Summary of files

* `model.py`: models

* `utils.py`: function to read the data from previous paper

* `data.py`: other data-processing functions

Dependencies:  Keras, pandas, inflect, numpy

## Details: models

The file model.py contains the implementation of a number of single or multi-task models.

`LanguageModel`: Basic language model.

`AgreementLM`: LM + Agreement.

`Supertagger`: Tagging (POS, CCG, etc).

`AgrSupertagger`: Tagging + Agreement.

`Agreement`: Agreement.

---

Models accept whichever of those parameters are relevant:

* `nwords`: vocabulary size

* `ntags`: number of tags

* `maxlen`: maximum sentence length

* `state_size`: parameter D from the article

* `loss_weights`: weights of both losses

* `id2word`: mapping of integer tokens to words (typically a NP array) 

* `word2id`: mapping of words to integer tokens (typically a dict)
    Beware: in the embedding layers or the LM output, the 0th dimension is for the null token.
    To get the embedding or probability of the i-th word you need to look at dimension i+1.

* `id2tag`: mapping of integer tokens to tags.

* `tag2id`: mapping of tags to integer tokens.

---

Data is expected to have the following numpy format:

    dtype([('word', 'O'), ('pos', 'O'), ('tag', 'O'), ('subj', 'int'), ('verb', 'int')])

The column `word` contains string tokens, the column `pos` contains Penn Treebank POS tags in string
format, the column `tag` contains target tags in string format, the columns `subj` and `verb` have
only zeroes except at the position of the subject and the verb respectively.

Note that models do not actually use most columns in most cases.

---

Models support the following functions, calqued on Keras functions of the same name:

* `predict`

* `fit`

* `evaluate`

They take the data in a list or NP array as first parameter, and a parameter `batch_size`. There is
also a parameter `nb_worker` which you should set to 1 to avoid weird bugs. `fit` accepts both
training and testing data in that order, and it also takes the arguments `nb_epoch` (maximum number
of training epochs), `early_stopping` (equivalent to the Keras callback of the same name),
`verbose` (also transferred to Keras).

Models can be saved and loaded back to folders using the `save` method and the `load_base_model`
function.

## Details: data

The function `utils.deps_from_tsv` is designed to read
[the following file](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz)
provided [here](https://github.com/TalLinzen/rnn_agreement)
(which is were `utils.py` is from as well).

In the file `data.py`, you'll find among others the following functions:

* `tsv_to_numpy(data)`: transforms the output of deps_to_tsv into our format (as a `list` of numpy arrays)

* `extract_ccg(folder)`: if `folder` is the "data" subfolder of the ccg-bank, extract the sentences
    and the CCG tags. Returns a `list` of `dict`'s of numpy arrays (sentences are organised in
    sections and then by ID).

* `apply_threshold_pos(data, thres)`: removes words less frequent than `thres` (`int`) and replaces
    them with their POS tag. Returns a pair of the new data and the new lexicon (as a `set`).

* `apply_threshold_void(data, thres, key='tag')`: same as above but replaces infrequent words with
    `'_'` and can apply to any key.

* `apply_length_threshold(data, thres)`: remove sentences longer than `thres`.

* `build_dicts(vocab)`: returns a pair of an int-to-word (`np.array`) and a word-to-int (`dict`)
    token mapping from a lexicon (`set` or other iterable).
