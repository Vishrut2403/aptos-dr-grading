"""The common evaluation protocol.

Every model in the study is scored by this one function, so the comparison
table in the report cannot drift between members. Nobody should be computing
their own accuracy.

evaluate(y_true, y_pred) -> dict with these keys:

  qwk           quadratic weighted kappa, the headline metric. Penalises being
                wrong by three grades far more than by one, which plain
                accuracy does not. sklearn: cohen_kappa_score(weights="quadratic")
  accuracy      plain accuracy, reported for comparison, not as the headline
  macro_f1      unweighted mean F1 over the five grades, so the minority
                grades count as much as No DR
  mae           mean absolute difference between predicted and true grade
  referable_f1  F1 on the binary question (grade >= 2), which is what decides
                whether a patient is referred to an ophthalmologist
  confusion     5x5 confusion matrix as a nested list, for the error analysis
                in Objective 3

Watch out: a model that predicts No DR for every image scores 49% accuracy on
this dataset but a quadratic weighted kappa of exactly 0. Make sure the tests
cover that case.

comparison_table(results) renders the single cross-model table the report is
graded on, from the list of result dicts written by src/train.py.
"""

import numpy as np

NUM_CLASSES = 5


def evaluate(y_true, y_pred):
    raise NotImplementedError("evaluate: not implemented yet")


def comparison_table(results):
    raise NotImplementedError("comparison_table: not implemented yet")
