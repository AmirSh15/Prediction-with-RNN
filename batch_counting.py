
"""Simple batch counter: grabs chunks of indices, repermuted after every pass"""

import numpy as np

class batchcounter():

  def __init__(self, gap, length, shuffle=True, seed=None):
    self.gap = gap
    self.length = length
    self.order = np.arange(length)
    if shuffle:
      np.random.seed(seed=seed)
      np.random.shuffle(self.order)
    self.start = 0
    self.wraps = 0

  def next_inds(self, seed=None):
    start = self.start
    end = start + self.gap
    if end >  self.length:
      self.wraps += 1
      self.start = start = 0
      end = start + self.gap
      np.random.shuffle(self.order)
    self.start += self.gap
    return self.order[start:end]

