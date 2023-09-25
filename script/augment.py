import logging

import torch
from torch import nn
from .consts import BEAT_EPS

logger = logging.getLogger(__name__)


class RandomTempoChange(nn.Module):
    def __init__(
        self, prob: float = 0.5, min_ratio: float = 0.8, max_ratio: float = 1.2
    ):
        super().__init__()
        self.prob = prob
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, inputs):
        notes, annots = inputs
        change, ratio = torch.rand(2).tolist()
        if change > self.prob:
            return notes, annots
        ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * ratio
        inv_ratio = 1 / ratio
        notes[:, 1:3] *= inv_ratio
        annots["beats"] *= inv_ratio
        annots["downbeats"] *= inv_ratio
        annots["time_signatures"][:, 0] *= inv_ratio
        annots["key_signatures"][:, 0] *= inv_ratio
        return notes, annots


class RandomPitchShift(nn.Module):
    def __init__(self, prob: float = 0.5, min_shift: int = -12, max_shift: int = 12):
        super().__init__()
        self.prob = prob
        self.min_shift = min_shift
        self.max_shift = max_shift

    def forward(self, inputs):
        notes, annots = inputs
        change = torch.rand(1).item()
        if change > self.prob:
            return notes, annots
        shift = torch.randint(self.min_shift, self.max_shift, (1,)).item()
        notes[:, 0] += shift
        keysig = annots["key_signatures"]
        keysig[:, 1] = (keysig[:, 1] + shift) % 24
        return notes, annots


class RandomAddRemoveNotes(nn.Module):
    def __init__(self, add_note_p: float = 0.5, add_ratio: float = 0.2, remove_note_p: float = 0.5, remove_ratio: float = 0.2):
        super().__init__()
        if add_note_p + remove_note_p > 1.0:
            total = add_note_p + remove_note_p
            logger.warning(
                "extra_note_prob (%.3f) + missing_note_p (%.3f) > 1; reset to %.3f, %.3f respectively",
                add_note_p,
                remove_note_p,
                (add_note_p := add_note_p / total),
                (remove_note_p := remove_note_p / total),
            )
        self.add_note_p = add_note_p
        self.add_ratio = add_ratio
        self.remove_note_p = remove_note_p
        self.remove_ratio = remove_ratio

    def forward(self, inputs):
        notes, annots = inputs
        rand = torch.rand(1).item()
        if rand < self.add_note_p:
            notes, annots = self.add_notes(notes, annots)
        elif rand > 1.0 - self.remove_note_p:
            notes, annots = self.remove_notes(notes, annots)
        return notes, annots

    def add_notes(self, notes, annots):
        new_seq = torch.repeat_interleave(notes, 2, dim=0)
        # pitch shift for extra notes (+-12)
        shift = torch.randint(-12, 12 + 1, (len(new_seq),))
        shift[::2] = 0
        new_seq[:, 0] += shift
        new_seq[:, 0][new_seq[:, 0] < 0] += 12
        new_seq[:, 0][new_seq[:, 0] > 127] -= 12

        # keep a random ratio of extra notes
        probs = torch.rand(len(new_seq))
        probs[::2] = 0.0
        kept = probs < self.add_ratio
        new_seq = new_seq[kept]
        for key in ["onsets_musical", "note_value", "hands"]:
            if annots[key] is not None:
                annots[key] = torch.repeat_interleave(annots[key], 2, dim=0)[kept]
        if annots["hands"] is not None:
            # mask out hand for extra notes during training
            hands_mask = torch.ones(len(notes) * 2)
            hands_mask[1::2] = 0
            annots["hands_mask"] = hands_mask[kept]
        return new_seq, annots

    def remove_notes(self, notes, annots):
        # find successive concurrent notes
        candidates = torch.diff(notes[:, 1]) < BEAT_EPS
        # randomly select a ratio of candidates to be removed
        candidates_probs = candidates * torch.rand(len(candidates))
        kept = torch.cat(
            [torch.tensor([True]), candidates_probs < (1 - self.remove_ratio)]
        )
        # remove selected candidates
        notes = notes[kept]
        for key in ["onsets_musical", "note_value", "hands"]:
            if annots[key] is not None:
                annots[key] = annots[key][kept]
        return notes, annots
