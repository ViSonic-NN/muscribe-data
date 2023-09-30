import logging
from functools import cmp_to_key, partial, reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pm
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .augment import RandomAddRemoveNotes, RandomPitchShift, RandomTempoChange
from .consts import BEAT_EPS, KEYSIG_N_SHARPS, KEYSIG_NOTE

logger = logging.getLogger(__name__)
ANNOT_KEYS = [
    "beats",
    "downbeats",
    "time_signatures",
    "key_signatures",
    "onsets_musical",
    "note_value",
    "hands",
]


class MIDIDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        feature_pickle: Path,
        split: str | list[str],
        annot_kinds: str | list[str] | None = None,
    ):
        from multiprocessing import cpu_count

        meta = pd.read_csv(dataset_dir / "metadata.csv")  # type: ignore
        n_process = min(cpu_count(), 32)
        dataset = prepare_features(dataset_dir, feature_pickle, n_process)
        assert len(meta) == len(dataset)

        selection = np.nonzero(meta["split"] == split)[0]
        self.metadata = meta.iloc[selection].reset_index()
        # int() is only for type checking
        self.dataset = [dataset[int(i)] for i in selection]
        self.split = split
        if annot_kinds is None:
            self.annots = ANNOT_KEYS  # Request all features
        elif isinstance(annot_kinds, str):
            self.annots = [annot_kinds]
        else:
            self.annots = annot_kinds
        self.augments = nn.Sequential(
            # RandomPitchShift(),
            # RandomTempoChange(),
            # RandomAddRemoveNotes(),
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[Tensor, list[Tensor]]:
        notes, annots = self.dataset[index]
        assert notes.dtype == torch.float32
        notes, annots = self.augments(notes, annots)
        annots = [annots[k] for k in self.annots]
        return notes, annots


def prepare_features(dataset_dir: Path, pickle_file: Path, workers: int):
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map

    if pickle_file.exists():
        ret = torch.load(pickle_file)
        assert isinstance(ret, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in ret)
    else:
        logger.info("Preparing features...")
        # Missing values are loaded as NaN, and we'll convert them to None
        meta = pd.read_csv(dataset_dir / "metadata.csv").replace({np.nan: None})
        rows = [row for _, row in meta.iterrows()]
        worker_f = partial(_prepare_one_feature, prefix=dataset_dir)
        if workers > 0:
            ret = process_map(worker_f, rows, max_workers=workers, chunksize=4)
        else:
            ret = [worker_f(row) for row in tqdm(rows)]
        torch.save(ret, pickle_file)
    return ret


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


def read_midi_notes(midi_file: Path):
    """
    Get note sequence from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    """
    midi_data = pm.PrettyMIDI(midi_file.as_posix())
    notes = reduce(lambda x, y: x + y, [inst.notes for inst in midi_data.instruments])
    notes = sorted(notes, key=cmp_to_key(compare_note_order))
    # conver to numpy array
    notes = torch.tensor(
        [
            (note.pitch, note.start, note.end - note.start, note.velocity)
            for note in notes
        ]
    )
    return notes


def read_annot_file(annot_file: Path):
    """
    Get annots from annotation file in ASAP dataset.
    annotatioins in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples
    }, all in torch.tensor.
    """
    annot_data = pd.read_csv(annot_file.as_posix(), header=None, sep="\t")

    name_to_pitch = {
        name: idx for idx, name in enumerate(KEYSIG_NOTE)
    }
    nsharp_to_pitch = {
        n_sharps: name_to_pitch[name] for name, n_sharps in KEYSIG_N_SHARPS.items()
    }
    beats, downbeats, key_signatures, time_signatures = [], [], [], []
    for _, row in annot_data.iterrows():
        a = row[2].split(",")
        # beats
        beats.append(row[0])
        # downbeats
        if a[0] == "db":
            downbeats.append(row[0])
        # time_signatures
        if len(a) >= 2 and a[1] != "":
            numerator, denominator = a[1].split("/")
            time_signatures.append((row[0], int(numerator), int(denominator)))
        # key_signatures
        if len(a) == 3 and a[2] != "":
            key_signatures.append((row[0], nsharp_to_pitch[int(a[2])]))

    # save as annotation dict
    annots = {
        "beats": torch.tensor(beats).float(),
        "downbeats": torch.tensor(downbeats).float(),
        "time_signatures": torch.tensor(time_signatures).float(),
        "key_signatures": torch.tensor(key_signatures).float(),
        "onsets_musical": torch.zeros(0),
        "note_value": torch.zeros(0),
        # Shape matches the output of read_midi_notes_and_annots
        "hands": torch.zeros(0, 2),
    }
    return annots


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
        note_sequence_with_hand = []
        for hand, inst in enumerate(midi_data.instruments):
            for note in inst.notes:
                note_sequence_with_hand.append((note, hand))

        def compare_note_with_hand(x, y):
            return compare_note_order(x[0], y[0])

        note_sequence_with_hand = sorted(
            note_sequence_with_hand, key=cmp_to_key(compare_note_with_hand)
        )

        notes, hands = [], []
        for note, hand in note_sequence_with_hand:
            notes.append(note)
            hands.append(hand)
    else:
        # ignore data with other numbers of hand parts
        notes = reduce(
            lambda x, y: x + y, [inst.notes for inst in midi_data.instruments]
        )
        notes = sorted(notes, key=cmp_to_key(compare_note_order))
        hands = None

    # beats
    beats = midi_data.get_beats()
    # downbeats
    downbeats = midi_data.get_downbeats()
    # time_signatures
    time_signatures = [
        (t.time, t.numerator, t.denominator) for t in midi_data.time_signature_changes
    ]
    # key_signatures
    key_signatures = [(k.time, k.key_number) for k in midi_data.key_signature_changes]

    # onsets_musical and note_values
    TT = midi_data.time_to_tick

    def times2note_value(start, end):
        # convert start and end times to note value (unit: beat, range: 0-4)
        this = np.where(beats - start <= BEAT_EPS)[0][-1]
        this, next = (this, this + 1) if this + 1 < len(beats) else (-2, -1)
        return (TT(end) - TT(start)) / (TT(beats[next]) - TT(beats[this]))

    time2pos = lambda t: times2note_value(
        t, t
    )  # convert time to position in musical time within a beat (unit: beat, range: 0-1)

    # get onsets_musical and note_values
    # filter out small negative values (they are usually caused by errors in time_to_tick convertion)
    onsets_musical = [
        min(1, max(0, time2pos(note.start))) for note in notes
    ]  # in range 0-1
    note_values = [max(0, times2note_value(note.start, note.end)) for note in notes]

    # conver to Tensor
    notes = torch.tensor(
        [
            [note.pitch, note.start, note.end - note.start, note.velocity]
            for note in notes
        ]
    )
    # save as annotation dict
    annots = {
        "beats": torch.tensor(beats).float(),
        "downbeats": torch.tensor(downbeats).float(),
        "time_signatures": torch.tensor(time_signatures).float(),
        "key_signatures": torch.tensor(key_signatures).float(),
        "onsets_musical": torch.tensor(onsets_musical).float(),
        "note_value": torch.tensor(note_values).float(),
        "hands": torch.tensor(hands).float()
        if hands is not None
        else torch.zeros(0, 2),
    }
    return notes, annots


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