from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
from keras_contrib.layers import CRF


def create_model(maxlen, chars, word_size):
    """

    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model_crf(maxlen, chars, word_size):
    """

    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'), )(blstm)
    crf = CRF(5, sparse_target=True)
    output = crf(output)
    model = Model(input=sequence, output=output)
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    return model
