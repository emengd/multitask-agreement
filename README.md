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

Models can be saved and loaded back to fodlers using the `save` method and the `load_base_model`
function.

## Details: data


