from music21 import converter, instrument, note, chord
import pickle

midi = converter.parse("sample_music.mid")
notes = []

parts = instrument.partitionByInstrument(midi)
if parts:  
    notes_to_parse = parts.parts[0].recurse()
else:  
    notes_to_parse = midi.flat.notes

for element in notes_to_parse:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))  
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n.pitch) for n in element.notes))  

with open("data/notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Notes extracted and saved!")

