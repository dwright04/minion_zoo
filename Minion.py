import numpy as np

class Minion(object):
  def __init__(self):
    raise NotImplementedError
  def classify(self):
    raise NotImplementedError

class ExpertMinion(Minion):
  def __init__(self, id):
    self.id = id
  def classify(self, subject_id, gold_label):
    return (subject_id, gold_label)

class AllTheSingleLabelsMinion(Minion):
  def __init__(self, id, label):
    self.id = id
    self.label = label
  def classify(self, subject_id):
    return (subject_id, self.label)

class RandomMinion(Minion):
  def __init__(self, id):
    self.id = id
  def classify(self, subject_id):
    return (subject_id, np.random.randint(0,2))

class ConfusionMatricMinion(Minion):
  def __init__(self, id, confusion_matrix):
    self.id = id
    self.confusion_matrix = confusion_matrix
  def classify(self, subject_id, gold_label):
    if np.random.randn() < self.confusion_matrix[gold_label]:
      return (subject_id, gold_label)
    else:
      return (subject_id, gold_label == 0)

class MachineLearningMinion(Minion):
  def __init__(self, id):
    raise NotImplementedError
  def classify(self, subject_id):
    raise NotImplementedError

def main():

  try:
    m = Minion()
  except NotImplementedError:
    pass

  m = ExpertMinion(1)
  assert m.classify(1, 0)[1] == 0

  m = AllTheSingleLabelsMinion(2, 0)
  assert m.classify(1)[1] == 0

  m = AllTheSingleLabelsMinion(3, 1)
  assert m.classify(1)[1] == 1

  m = RandomMinion(4)
  # this will fail 1 in 2^100 times
  x = []
  for i in range(100):
    x.append(m.classify(1)[1])
  x = set(x)
  assert 0 in x
  assert 1 in x

if __name__ == '__main__':
  main()
