import logging
from functools import cmp_to_key, partial
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pm
import torch

from .consts import MIDI_IDX_TO_NATURAL_IDX

ANNOT_KEYS = [
    "beats",
    "downbeats",
    "time_signatures",
    "key_signatures",
    "onsets_musical",
    "note_value",
    "hands",
]
NO_ASAP_KEYS = set(["onsets_musical", "note_value", "hands"])

logger = logging.getLogger(__name__)
_n_processes = min(32, cpu_count())


def read_full_dataset(
    dataset_dir: Path, pickle_file: Path, workers: int = _n_processes
):
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map

    # Missing values are loaded as NaN, and we'll convert them to None
    meta = pd.read_csv(dataset_dir / "metadata.csv").replace({np.nan: None})
    if pickle_file.exists():
        data = torch.load(pickle_file)
        assert isinstance(data, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in data)
    else:
        logger.info("Preparing features...")
        rows = [row for _, row in meta.iterrows()]
        worker_f = partial(_prepare_one_feature, prefix=dataset_dir)
        if workers > 0:
            data = process_map(worker_f, rows, max_workers=workers, chunksize=4)
        else:
            data = [worker_f(row) for row in tqdm(rows)]
        torch.save(data, pickle_file)
    return meta, data


def _prepare_one_feature(row, prefix: Path):
    if row["annot_file"] is None:
        # get note sequence and annots dict
        # (beats, downbeats, key signatures, time signatures, musical onset times, note value in beats, hand parts)
        notes, annots = read_midi_notes_and_annots(prefix / row["perf_midi_file"])
    else:
        # get note sequence
        notes = read_midi_notes(prefix / row["perf_midi_file"])
        # get annots dict (beats, downbeats, key signatures, time signatures)
        annots = read_annot_file(prefix / row["annot_file"])
    return notes, annots


def read_midi_notes(midi_file: Path | str):
    """
    Get note sequence from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in torch.array.
    """
    midi_data = pm.PrettyMIDI(str(midi_file))
    notes = [note for inst in midi_data.instruments for note in inst.notes]
    notes = sorted(notes, key=cmp_to_key(compare_note_order))
    # conver to numpy array
    notes = [
        (note.pitch, note.start, note.end - note.start, note.velocity) for note in notes
    ]
    return torch.tensor(notes).float()


def read_annot_file(annot_file: Path):
    """
    Get annots from annotation file in ASAP dataset.
    annotations in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples
    }, all in torch.tensor.
    """
    annot_data = pd.read_csv(annot_file.as_posix(), header=None, sep="\t")
    beats = annot_data[0].to_numpy()
    downbeats, key_sigs, time_sigs = [], [], []
    for onset, annot_str in annot_data[[0, 2]].to_numpy():
        annots = annot_str.split(",")
        # downbeats
        if annots[0] == "db":
            downbeats.append(onset)
        # time_signatures
        if len(annots) >= 2 and annots[1] != "":
            numer, denom = annots[1].split("/")
            time_sigs.append((onset, int(numer), int(denom)))
        # key_signatures
        if len(annots) == 3 and annots[2] != "":
            key_sigs.append((onset, MIDI_IDX_TO_NATURAL_IDX[int(annots[2])]))
    return {
        "beats": torch.from_numpy(beats).float(),
        "downbeats": torch.tensor(downbeats).float(),
        "time_signatures": torch.tensor(time_sigs).float(),
        "key_signatures": torch.tensor(key_sigs).float(),
        # "onsets_musical", "note_value", "hands" are not available in ASAP dataset
    }


def read_midi_notes_and_annots(midi_file: Path):
    """
    Get beat sequence and annots from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    annots in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples,
        onsets_musical: list of onsets in musical time for each note (within a beat),
        note_value: list of note values (in beats),
        hands: list of hand part for each note (0: left, 1: right)
    """
    midi_data = pm.PrettyMIDI(midi_file.as_posix())

    # note sequence and hands
    if len(midi_data.instruments) == 2:
        # two hand parts
        notes_hands = [
            (note, hand)
            for hand, inst in enumerate(midi_data.instruments)
            for note in inst.notes
        ]
        notes_hands = sorted(
            notes_hands, key=cmp_to_key(lambda x, y: compare_note_order(x[0], y[0]))
        )
        notes, hands = zip(*notes_hands)
    else:
        # ignore data with other numbers of hand parts
        notes = [note for inst in midi_data.instruments for note in inst.notes]
        notes = sorted(notes, key=cmp_to_key(compare_note_order))
        hands = None
    notes = [
        [note.pitch, note.start, note.end - note.start, note.velocity] for note in notes
    ]
    notes = torch.tensor(notes).float()

    # beats, downbeats
    beats = torch.from_numpy(midi_data.get_beats()).float()
    downbeats = torch.from_numpy(midi_data.get_downbeats()).float()
    # time_signatures
    time_signatures = [
        (t.time, t.numerator, t.denominator) for t in midi_data.time_signature_changes
    ]
    # key_signatures
    key_signatures = [(k.time, k.key_number) for k in midi_data.key_signature_changes]

    note_onsets = notes[:, 1]
    note_beat = torch.searchsorted(beats, note_onsets, right=True)
    # If a note lands before the first beat, we assign it to beats[0] and beats[1];
    note_beat[note_beat == 0] = 1
    # if after the last beat, we assign it to beats[-2] and beats[-1].
    note_beat[note_beat == len(beats)] = -1

    ttt = midi_data.time_to_tick
    note_onset_ticks = torch.tensor([ttt(note) for note in note_onsets])
    note_end_ticks = torch.tensor([ttt(onset + dur) for _, onset, dur, _ in notes])
    beat_ticks = torch.tensor([ttt(beat) for beat in beats])
    beat_tick_diff = beat_ticks[note_beat] - beat_ticks[note_beat - 1]

    return notes, {
        "beats": beats,
        "downbeats": downbeats,
        "time_signatures": torch.tensor(time_signatures).float(),
        "key_signatures": torch.tensor(key_signatures).float(),
        "onsets_musical": note_onset_ticks / beat_tick_diff,
        "note_value": (note_end_ticks - note_onset_ticks) / beat_tick_diff,
        "hands": torch.tensor(hands).float()
        if hands is not None
        else torch.zeros(0, 2),
    }


def compare_note_order(note1, note2):
    """
    Compare two notes by firstly onset and then pitch.
    """
    if note1.start < note2.start:
        return -1
    elif note1.start == note2.start:
        if note1.pitch < note2.pitch:
            return -1
        elif note1.pitch == note2.pitch:
            return 0
        else:
            return 1
    else:
        return 1
