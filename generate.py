import os
import pickle
import numpy as np
from music21 import stream, note, chord, instrument, pitch
from tensorflow.keras.models import load_model
os.makedirs("output", exist_ok=True)

model = load_model("model/music_model.h5")

with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

sequence_length = 100
start_index = np.random.randint(0, len(notes) - sequence_length - 1)
pattern = notes[start_index:start_index + sequence_length]

# Generate
generated_notes = []
for _ in range(200):
    input_seq = [note_to_int[n] for n in pattern]
    input_seq = np.reshape(input_seq, (1, sequence_length, 1))
    input_seq = input_seq / float(len(pitchnames))

    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    generated_notes.append(result)

    pattern.append(result)
    pattern = pattern[1:]

# Convert to MIDI
output_notes = []
offset = 0
for item in generated_notes:
    if '.' in item or item.isdigit():
        elements = item.split('.')
        notes_list = []
        for n in elements:
            try:
                notes_list.append(note.Note(n))
            except:
                print(f"Skipping invalid note in chord: {n}")
        if notes_list:
            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)
    else:
        try:
            new_note = note.Note(item)
            new_note.offset = offset
            output_notes.append(new_note)
        except:
            print(f"Skipping invalid note: {item}")
    offset += 0.5


# Save to MIDI
midi_stream = stream.Stream(output_notes)
midi_stream.insert(0, instrument.Piano())
midi_stream.write("midi", fp="output/generated_music.mid")
