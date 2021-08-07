import re
from tensorflow import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Embedding, Conv2D, MaxPooling2D, \
    GlobalMaxPooling2D, Reshape, Conv2DTranspose, UpSampling2D

# Seq2Seq by using LSTM
from tensorflow.python.keras.utils.vis_utils import plot_model


def seq2seq_model_builder(vocab_size, HIDDEN_DIM=300):

    SEQUENCE_LENGHT = 40
    VOCAB_SIZE = vocab_size

    encoder_inputs = Input(shape=(SEQUENCE_LENGHT, ), dtype='int32',)
    encoder_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=SEQUENCE_LENGHT)(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=(SEQUENCE_LENGHT, ), dtype='int32',)
    decoder_embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=SEQUENCE_LENGHT)(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model

#model = seq2seq_model_builder(vocab_size=64, HIDDEN_DIM=300)
#model.summary()
#plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# ============================================
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# def padding(encoder_sequences, decoder_sequences, MAX_LEN):
#
#     encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
#     decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
#
#     return encoder_input_data, decoder_input_data
#
# encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN)
