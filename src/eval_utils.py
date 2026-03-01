
# This script handles the decoding functions and performance measurement

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


import re
import numpy as np
from sentence_transformers import util
import torch


def extract_spans_para(seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'null'
            if at.lower() == 'implicit':
                at = 'null'
            if ot.lower() == 'implicit':
                ot = 'null'

        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))

    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    if verbose:
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, verbose=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(gold_seqs[i], 'gold')
        pred_list = extract_spans_para(pred_seqs[i], 'pred')
        if verbose and i < 10:

            print("gold ", gold_seqs[i])
            print("pred ", pred_seqs[i])
            print()

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds


def compute_semantic_f1(pred_tuples, gold_tuples, model):
    """
    计算两个【已处理好的元组列表】之间的“语义F1分数”。
    这个函数是一个纯粹的计算模块。

    Args:
        pred_tuples (list): 预测的元组列表，例如 [('aspect', 'opinion', 'sentiment'),...]。
        gold_tuples (list): 用于比较的元组列表。
        model: 预加载的 sentence-transformer 模型，用于将词语转为向量。
    """
    num_preds = len(pred_tuples)
    num_golds = len(gold_tuples)

    if num_preds == 0 or num_golds == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    pred_aspects = [t for t in pred_tuples]
    pred_opinions = [t[1] for t in pred_tuples]
    gold_aspects = [t for t in gold_tuples]
    gold_opinions = [t[1] for t in gold_tuples]

    all_terms = list(set(pred_aspects + pred_opinions + gold_aspects + gold_opinions))
    if not all_terms:
        is_match = 1.0 if num_preds == num_golds else 0.0
        return {'precision': is_match * 100, 'recall': is_match * 100, 'f1': is_match * 100}

    embeddings = model.encode(all_terms, convert_to_tensor=True)
    emb_dict = {term: emb for term, emb in zip(all_terms, embeddings)}

    similarity_matrix = torch.zeros(num_preds, num_golds)
    for i in range(num_preds):
        for j in range(num_golds):
            pred_a, pred_o, pred_s = pred_tuples[i]
            gold_a, gold_o, gold_s = gold_tuples[j]

            s_sim = 1.0 if pred_s == gold_s else 0.0

            pred_a_emb = emb_dict.get(pred_a)
            gold_a_emb = emb_dict.get(gold_a)
            pred_o_emb = emb_dict.get(pred_o)
            gold_o_emb = emb_dict.get(gold_o)

            a_sim = 1.0 if pred_a == gold_a else (util.cos_sim(pred_a_emb,
                                                               gold_a_emb).item() if pred_a_emb is not None and gold_a_emb is not None else 0.0)
            o_sim = 1.0 if pred_o == gold_o else (util.cos_sim(pred_o_emb,
                                                               gold_o_emb).item() if pred_o_emb is not None and gold_o_emb is not None else 0.0)

            triplet_sim = s_sim * ((a_sim + o_sim) / 2.0)
            similarity_matrix[i][j] = triplet_sim

    soft_tp = 0
    temp_sim_matrix = similarity_matrix.clone()
    for _ in range(min(num_preds, num_golds)):
        max_val = temp_sim_matrix.max()
        if max_val <= 0: break
        indices = (temp_sim_matrix == max_val).nonzero(as_tuple=False)

        i, j = indices[0]

        soft_tp += max_val.item()
        temp_sim_matrix[i, :] = -1
        temp_sim_matrix[:, j] = -1

    soft_precision = soft_tp / num_preds if num_preds > 0 else 0.0
    soft_recall = soft_tp / num_golds if num_golds > 0 else 0.0

    f1 = 0.0
    if soft_precision + soft_recall > 0:
        f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall)

    scores = {
        'precision': soft_precision * 100,
        'recall': soft_recall * 100,
        'f1': f1 * 100
    }
    return scores

