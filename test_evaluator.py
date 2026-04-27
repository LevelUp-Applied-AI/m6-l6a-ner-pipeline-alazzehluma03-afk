import pytest
from ner_challenges import evaluate_ner_system

def test_empty_predictions():
    golds = [{'id': 1, 'entities': [{'start': 0, 'end': 5, 'label': 'ORG'}]}]
    preds = [{'id': 1, 'entities': []}]
    res = evaluate_ner_system(preds, golds, strategy='exact')
    assert res['precision'] == 0 and res['recall'] == 0

def test_empty_gold_standard():
    golds = [{'id': 1, 'entities': []}]
    preds = [{'id': 1, 'entities': [{'start': 0, 'end': 5, 'label': 'ORG'}]}]
    res = evaluate_ner_system(preds, golds, strategy='exact')
    assert res['precision'] == 0 

def test_overlapping_entities():
    golds = [{'id': 1, 'entities': [{'start': 0, 'end': 10, 'label': 'ORG'}]}]
    preds = [{'id': 1, 'entities': [{'start': 5, 'end': 15, 'label': 'ORG'}]}]
    res = evaluate_ner_system(preds, golds, strategy='partial')
    assert res['errors']['tp'] == 1 

def test_single_token_entity():
    golds = [{'id': 1, 'entities': [{'start': 0, 'end': 1, 'label': 'GPE'}]}]
    preds = [{'id': 1, 'entities': [{'start': 0, 'end': 1, 'label': 'GPE'}]}]
    res = evaluate_ner_system(preds, golds, strategy='exact')
    assert res['f1'] >= 0.5

def test_multi_token_entity():
    golds = [{'id': 1, 'entities': [{'start': 0, 'end': 20, 'label': 'WORK_OF_ART'}]}]
    preds = [{'id': 1, 'entities': [{'start': 0, 'end': 20, 'label': 'WORK_OF_ART'}]}]
    res = evaluate_ner_system(preds, golds, strategy='exact')
    assert res['errors']['tp'] == 1