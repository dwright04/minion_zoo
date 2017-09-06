"""Example robot classifiers.

This module defines a set of Minion objects - robots designed to mimic various
classification behaviour.  All Minions must implement the Minion.classify() 
class method to determine how they assign classifications to subjects.

Note
----
The current implementation has the
`Supernova Hunters <https://www.zooniverse.org/projects/dwright04/supernova-hunters>`_ citizen science project specifically in mind.  But is intended 
to generalise to other projects including multiclass classification projects.
The NoisyMinion class currently only works for binary classification tasks as
randomly flipping labels for multiclass classification problems should likely
consider the entire confusion matrix and not just return a random label.
"""

import numpy as np

class Minion(object):
  """Abstract Minion class.
  
  Methods
  -------
  classify(subject_id)
    classify the given subject.
  """
  def classify(self, subject_id):
    """Abstract class method.
    
    Must be overriden by subclasses.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Raises
    ------
    NotImplementedError
      this class method must be overridden.
    """
    raise NotImplementedError

class ExpertMinion(Minion):
  """Expert classifier always returns the correct label.
  
  Classifies a given subject with the expert provided label.
  
  Attributes
  ----------
  id : int
    unique Minion id.
    
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the provided gold label.
  """
  def __init__(self, id):
    """
    Parameters
    ----------
    id : int
      unique Minion id.
    """
    self.id = id
  def classify(self, subject_id, gold_label):
    """Classify the given subject with the expert provided gold label.
    
    Note
    ----
    The provided gold_label must be valid for the classification task.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
    gold_label : int
      expert provided gold label.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, gold_label)

class AllTheSingleLabelsMinion(Minion):
  """Classifier returning a single label only.
  
  Classifies all given subjects with the same label.
  
  Attributes
  ----------
  id : int
    unique Minion id.
  label : int
    label that this classifier will return for all subjects.
      
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the provided gold label.
  """
  def __init__(self, id, label):
    """
    Parameters
    ----------
    id : int
      unique Minion id.
    label : int
      label that this classifier will return for all subjects.
      
    Note
    ----
    The provided label must be valid for the classification task.
    """
    self.id = id
    self.label = label
  def classify(self, subject_id):
    """Classify the given subject with the label for this classifier.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, self.label)

class RandomMinion(Minion):
  """Classifier that returns a random label for a given subject.
  
  Classifies each subject with a random label drawn from a provided list of 
  valid labels.

  Attributes
  ----------
  id : int
    unique Minion id.
  labels : array-like, shape (n_labels,)
    list of valid labels for the classification task.
      
  Methods
  -------
  classify(subject_id)
    classify the given subject with the provided gold label.
  """

  def __init__(self, id, labels):
    """
    Parameters
    ----------
    id : int
      unique Minion id.
    labels : array-like, shape (n_labels,)
      list of valid labels for the classification task.
      
    Note
    ----
    The provided labels must be valid for the classification task.
    """
    self.id = id
    self.labels = labels
  def classify(self, subject_id):
    """Classify the given subject with a label selected randomly from the 
    provided labels.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, np.random.choice(labels))

class NoisyMinion(Minion):
  """Classifier returns the correct label a specified fraction of the time.
  
  The provided gold standard label is flipped based on the specified noise for
  this classifier defined in its confusion matrix.
  
  Note
  ----
  The NoisyMinion class is currently only implemented for binary classification
  problems.
  
  The ExpertMinion can be replicated with this class by defining both confusion
  matrix elements to be 1 corresponding to a perfectly astute classifier.
  
  Likewise perfectly obtuse, pessimistic and optimisitic classifiers can be 
  created by setting the corresponding confusion matrix elements to 0 or 1.
  
  Attributes
  ----------
  id : int
    unique Minion id.
  confusion_matrix : array-like, shape (2,)
    array of confusion matrix elements.  First element corresponds to 0 class,
    second element corresponds to 1 class.
      
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the gold label adding noise based on the 
    confusion matrix.
  """
  def __init__(self, id, confusion_matrix):
    """
    Parameters
    ----------
    id : int
      unique Minion id.
    confusion_matrix : array-like, shape (2,)
      array of confusion matrix elements.  First element corresponds to 0 
      class, second element corresponds to 1 class.
    
    Raises
    ------
    ValueError
      if all confusion matrix elements are not in the interval [0,1].
      
    Note
    ----
    The confusion matrix elements must be in the interval [0,1]
    """
    self.id = id
    if (confusion_matrix < 0).any() and (confusion_matrix > 1).any():
      raise ValueError('All confusion matrix elements must be in the' \
                    +  'interval [0,1].')
    self.confusion_matrix = confusion_matrix
  def classify(self, subject_id, gold_label):
    """Classify the given subject with the gold label adding noise based on the
    confusion matrix.

    Note
    ----
    The provided gold_label must be valid for the classification task.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
    gold_label : int
      expert provided gold label.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    if np.random.randn() < self.confusion_matrix[gold_label]:
      return (subject_id, gold_label)
    else:
      return (subject_id, gold_label == 0)

# TODO : Implement a machine learning based classifier.
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
