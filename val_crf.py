import pickle
import lstm_model

with open('model/chars.pkl', 'rb') as inp:
    chars = pickle.load(inp)
word_size = 128
maxlen = 32

model = lstm_model.create_model_crf(maxlen, chars, word_size, True)
model.load_weights('model/model_CRF.h5', by_name=True)

import re
import numpy as np


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]))[0][:len(s)]
        print(r)
    #     nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
    #     words = []
    #     for i in range(len(s)):
    #         if t[i] in ['s', 'b']:
    #             words.append(s[i])
    #         else:
    #             words[-1] += s[i]
    #     return words
    # else:
    #     return []


not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')


def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


print(cut_word('学习出一个模型，然后再预测出一条指定'))
