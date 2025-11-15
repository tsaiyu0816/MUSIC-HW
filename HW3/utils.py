# utils.py
import numpy as np
import miditoolkit

# ----- input parameters -----
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)  # 60 ~ 3840 ticks
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# ----- output parameters -----
DEFAULT_RESOLUTION = 480  # ticks per beat

# ---------- generic containers ----------
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
    def __repr__(self):
        return f'Item(name={self.name}, start={self.start}, end={self.end}, velocity={self.velocity}, pitch={self.pitch})'

class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
    def __repr__(self):
        return f'Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})'


# ---------- MIDI -> Items ----------
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)

    # notes：取第一軌（簡化）
    note_items = []
    if len(midi_obj.instruments) == 0:
        return [], []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item('Note', note.start, note.end, note.velocity, note.pitch))
    note_items.sort(key=lambda x: x.start)

    # tempos
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item('Tempo', tempo.time, None, None, int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)

    if len(note_items) == 0:
        return note_items, tempo_items

    # 展開 tempo 到每個 beat
    max_tick = max(note_items[-1].end, tempo_items[-1].start if tempo_items else 0)
    existing = {it.start: it.pitch for it in tempo_items} if tempo_items else {}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    out = []
    last_bpm = 120
    for t in wanted_ticks:
        if t in existing:
            last_bpm = existing[t]
        out.append(Item('Tempo', t, None, None, last_bpm))
    tempo_items = out
    return note_items, tempo_items

def quantize_items(items, ticks=120):
    # 480/4 = 120 → 1/16 bar（4/4 下）
    if not items:
        return items
    grids = np.arange(0, items[-1].start + 1, ticks, dtype=int)
    for item in items:
        idx = int(np.argmin(np.abs(grids - item.start)))
        shift = int(grids[idx] - item.start)
        item.start += shift
        if item.end is not None:
            item.end += shift
    return items


# ---------- 簡易和絃抽取（同一 1/16 起音 >=3 音則標記） ----------
_PITCH_CLASS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def _triad_label(pitches):
    pcs = sorted({p % 12 for p in pitches})
    if not pcs: return None
    root = min(pcs)
    def has(p, intervals): return all(((p+i) % 12) in pcs for i in intervals)
    quality = None
    for r in pcs:
        if has(r, [4,7]): quality = ''; root = r; break  # major
        if has(r, [3,7]): quality = 'm'; root = r; break # minor
    name = _PITCH_CLASS[root] + (quality if quality is not None else '')
    return name

def extract_chords(note_items):
    if not note_items:
        return []
    chords = []
    ticks_per_bar = DEFAULT_RESOLUTION * 4

    def pos_tick_of(tick):
        bar_st = (tick // ticks_per_bar) * ticks_per_bar
        flags = np.linspace(bar_st, bar_st + ticks_per_bar, DEFAULT_FRACTION, endpoint=False, dtype=int)
        idx = int(np.argmin(np.abs(flags - tick)))
        return int(flags[idx])

    onset_map = {}
    for n in note_items:
        st = pos_tick_of(n.start)
        onset_map.setdefault(st, []).append(n.pitch)

    for st, plist in onset_map.items():
        uniq = sorted(set(plist))
        if len(uniq) >= 3:
            label = _triad_label(uniq)
            if label is not None:
                chords.append(Item('Chord', st, None, None, label))
    chords.sort(key=lambda x: x.start)
    return chords


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [item for item in items if (item.start >= db1) and (item.start < db2)]
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups


# ---------- Items -> Events ----------
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        body = groups[i][1:-1]
        if 'Note' not in [it.name for it in body]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event('Bar', None, None, str(n_downbeat)))

        for item in body:
            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = int(np.argmin(np.abs(flags - item.start)))
            events.append(Event('Position', item.start, f'{index + 1}/{DEFAULT_FRACTION}', str(item.start)))

            if item.name == 'Note':
                # Velocity
                vel_idx = int(np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1)
                events.append(Event('Note Velocity', item.start, vel_idx, f'{item.velocity}/{DEFAULT_VELOCITY_BINS[vel_idx]}'))
                # Pitch
                events.append(Event('Note On', item.start, item.pitch, str(item.pitch)))
                # Duration
                duration = int(item.end - item.start)
                dur_idx = int(np.argmin(np.abs(DEFAULT_DURATION_BINS - duration)))
                events.append(Event('Note Duration', item.start, dur_idx, f'{duration}/{DEFAULT_DURATION_BINS[dur_idx]}'))

            elif item.name == 'Chord':
                # value 存字串標籤（如 "C", "Am"）
                events.append(Event('Chord', item.start, item.pitch, str(item.pitch)))

            elif item.name == 'Tempo':
                tempo = int(item.pitch)
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, tempo - DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                else:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)
    return events


# ---------- token <-> events ----------
def word_to_event(words, word2event):
    events = []
    for w in words:
        ev = word2event.get(int(w))
        name, value = ev.split('_')
        events.append(Event(name, None, value, None))
    return events


# ---------- write MIDI ----------
def write_midi(*, words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)

    temp_notes, temp_chords, temp_tempos = [], [], []
    for i in range(len(events) - 3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar'); temp_chords.append('Bar'); temp_tempos.append('Bar')

        elif (events[i].name == 'Position'
              and events[i + 1].name == 'Note Velocity'
              and events[i + 2].name == 'Note On'
              and events[i + 3].name == 'Note Duration'):
            pos = int(events[i].value.split('/')[0]) - 1
            vel = int(DEFAULT_VELOCITY_BINS[int(events[i + 1].value)])
            pitch = int(events[i + 2].value)
            dur = DEFAULT_DURATION_BINS[int(events[i + 3].value)]
            temp_notes.append([pos, vel, pitch, dur])

        elif events[i].name == 'Position' and events[i + 1].name == 'Chord':
            pos = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([pos, events[i + 1].value])  # "C", "Am", ...

        elif (events[i].name == 'Position'
              and events[i + 1].name == 'Tempo Class'
              and events[i + 2].name == 'Tempo Value'):
            pos = int(events[i].value.split('/')[0]) - 1
            if events[i + 1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i + 2].value)
            elif events[i + 1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i + 2].value)
            else:
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i + 2].value)
            temp_tempos.append([pos, tempo])

    ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4

    # Notes -> absolute
    notes = []
    cur_bar = 0
    for n in temp_notes:
        if n == 'Bar':
            cur_bar += 1; continue
        pos, vel, pitch, dur = n
        bar_st = cur_bar * ticks_per_bar
        flags = np.linspace(bar_st, bar_st + ticks_per_bar, DEFAULT_FRACTION, endpoint=False, dtype=int)
        st = int(flags[pos]); et = int(st + dur)
        notes.append(miditoolkit.Note(vel, pitch, st, et))

    # Chords -> markers
    chords = []
    if temp_chords:
        cur_bar = 0
        for c in temp_chords:
            if c == 'Bar':
                cur_bar += 1; continue
            pos, name = c
            bar_st = cur_bar * ticks_per_bar
            flags = np.linspace(bar_st, bar_st + ticks_per_bar, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = int(flags[pos])
            chords.append([st, str(name)])

    # Tempos -> absolute
    tempos = []
    cur_bar = 0
    for t in temp_tempos:
        if t == 'Bar':
            cur_bar += 1; continue
        pos, bpm = t
        bar_st = cur_bar * ticks_per_bar
        flags = np.linspace(bar_st, bar_st + ticks_per_bar, DEFAULT_FRACTION, endpoint=False, dtype=int)
        st = int(flags[pos])
        tempos.append([st, int(bpm)])

    # 組 MIDI
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        last_time = DEFAULT_RESOLUTION * 4 * 4  # 接在 4 小節之後
        for note in notes:
            note.start += last_time; note.end += last_time
        if not midi.instruments:
            midi.instruments.append(miditoolkit.midi.containers.Instrument(0, is_drum=False))
        midi.instruments[0].notes.extend(notes)
        kept = [t for t in midi.tempo_changes if t.time < last_time]
        for st, bpm in tempos:
            kept.append(miditoolkit.midi.containers.TempoChange(bpm, st + last_time))
        midi.tempo_changes = kept
        for st, name in chords:
            midi.markers.append(miditoolkit.midi.containers.Marker(text=name, time=st + last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        midi.tempo_changes = [miditoolkit.midi.containers.TempoChange(bpm, st) for st, bpm in tempos]
        for st, name in chords:
            midi.markers.append(miditoolkit.midi.containers.Marker(text=name, time=st))

    midi.dump(output_path)
