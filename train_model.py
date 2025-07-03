import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical

with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

sequence_length = 100
pitches = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(pitches)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitches))
network_output = to_categorical(network_output)

model = Sequential([
    LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(512),
    Dropout(0.3),
    Dense(256),
    Dropout(0.3),
    Dense(len(pitches)),
    Activation("softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(network_input, network_output, epochs=100, batch_size=64)

model.save("model/music_model.h5")

