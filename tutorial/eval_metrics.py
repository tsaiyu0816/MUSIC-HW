import numpy as np
from glob import glob
import random, itertools
import pickle
import utils
import pandas as pd
import os
import scipy.stats
import tqdm
from tqdm import tqdm
import argparse
from miditoolkit import MidiFile

from musdr.side_utils import (
#   get_event_seq, 
  get_bars_crop, 
  get_pitch_histogram, 
  compute_histogram_entropy, 
  get_onset_xor_distance,
  get_chord_sequence,
  read_fitness_mat
)

#############################################################################
'''
Default event encodings (ones used by the Jazz Transformer).
You may override the defaults in function arguments to suit your own vocabulary.
'''
BAR_EV = 0                  # the ID of ``Bar`` event
POS_EVS = range(1, 17)      # the IDs of ``Position`` events
PITCH_EVS = range(99, 185)  # the ID of Pitch => Note on events
#############################################################################

def parse_opt():
    parser = argparse.ArgumentParser()
    # training opts
    parser.add_argument('--dict_path', type=str,
                        help='the dictionary path', required=True)
    parser.add_argument('--output_file_path', type=str,
                        help='the output file path.', required=True)
    args = parser.parse_args()
    return args
  
opt = parse_opt()


event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))


def extract_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end

    items = tempo_items + note_items

    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

def prepare_data(midi_path):
    # extract events
    events = extract_events(midi_path)
    # event to word
    words = []
    for event in events:
        e = '{}_{}'.format(event.name, event.value)
        if e in event2word:
            words.append(event2word[e])
        else:
            # OOV
            if event.name == 'Note Velocity':
                # replace with max velocity based on our training data
                words.append(event2word['Note Velocity_21'])
            else:
                # something is wrong
                # you should handle it for your own purpose
                print('something is wrong! {}'.format(e))
    return words

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id=BAR_EV, pitch_evs=PITCH_EVS, verbose=False):
  '''
  Computes the average pitch-class histogram entropy of a piece.
  (Metric ``H``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when a crop contains no notes.

  Returns:
    float: the average n-bar pitch-class histogram entropy of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  n_bars = piece_ev_seq.count(bar_ev_id)
  if window_size > n_bars:
    print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
    window_size = n_bars

  # compute entropy of all possible segments
  pitch_ents = []
  for st_bar in range(0, n_bars - window_size + 1):
    seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)

    pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
    if pitch_hist is None:
      if verbose:
        print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
      continue

    pitch_ents.append( compute_histogram_entropy(pitch_hist) )

  return np.mean(pitch_ents)

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id=BAR_EV, pos_evs=POS_EVS, pitch_evs=PITCH_EVS, max_pairs=1000):
  '''
  Computes the average grooving pattern similarity between all pairs of bars of a piece.
  (Metric ``GS``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    max_pairs (int): maximum #(pairs) considered, to save computation overhead.

  Returns:
    float: 0~1, the average grooving pattern similarity of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  # get every single bar & compute indices of bar pairs
  n_bars = piece_ev_seq.count(bar_ev_id)
  bar_seqs = []
  for b in range(n_bars):
    bar_seqs.append( get_bars_crop(piece_ev_seq, b, b, bar_ev_id) )
  pairs = list( itertools.combinations(range(n_bars), 2) )
  if len(pairs) > max_pairs:
    pairs = random.sample(pairs, max_pairs)

  # compute pairwise grooving similarities
  grv_sims = []
  for p in pairs:
    grv_sims.append(
      1. - get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
    )

  return np.mean(grv_sims)


if __name__ == "__main__":
  # codes below are for testing
  test_pieces = sorted(glob(os.path.join(opt.output_file_path, '*.mid')))

  # print (test_pieces)

  result_dict = {
      'piece_name': [],
      'H1': [],
      'H4': [],
      'GS': []
  }

  for p in tqdm(test_pieces):
      result_dict['piece_name'].append(p.replace('\\', '/').split('/')[-1])
      seq = prepare_data(p)

      h1 = compute_piece_pitch_entropy(seq, 1)
      result_dict['H1'].append(h1)
      h4 = compute_piece_pitch_entropy(seq, 4)
      result_dict['H4'].append(h4)
      gs = compute_piece_groove_similarity(seq)
      result_dict['GS'].append(gs)

  if len(result_dict):
      df = pd.DataFrame.from_dict(result_dict)
      df.to_csv('pop1k7.csv', index=False, encoding='utf-8')