"""Pure workspace and retrieval helpers, independent of the UI and model runtime."""
from __future__ import annotations

import re
import secrets
import unicodedata

import numpy as np


def normalize(text):
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans('يىك', 'ییک'))
    return re.sub(r'\s+', ' ', text.replace('\u200c', ' ')).strip()


def contextual_query(question, messages, document_ids):
    """Use only the previous user question in the same document scope."""
    words = normalize(question).split()
    followup = any(w in normalize(question) for w in ('این', 'آن', 'همین', 'قبلی', 'بیشتر', 'مهلتش', 'مسئولش', 'ادامه')) or len(words) <= 4
    if followup:
        for message in reversed(messages):
            if message['role'] == 'user' and set(message.get('document_ids', [])) == set(document_ids):
                return normalize(message.get('search_question',message['content'])[-400:] + ' ' + question)
    return normalize(question)


def rank_chunks(question, rows, query_vector, k):
    """Hybrid relevance with duplicate suppression and modest page diversity."""
    matrix = np.stack([np.frombuffer(r['embedding'], dtype=np.float32) for r in rows])
    semantic = matrix @ np.asarray(query_vector, dtype=np.float32)
    stop = {'از', 'به', 'در', 'را', 'و', 'که', 'با', 'این', 'است', 'رو', 'چیست', 'بگو', 'چه'}
    terms = set(re.findall(r'\w+', normalize(question))) - stop
    lexical = np.array([len(terms.intersection(re.findall(r'\w+', normalize(r['text'])))) / max(1,len(terms)) for r in rows])
    scores = 0.78 * semantic + 0.22 * lexical
    candidates = list(np.argsort(scores)[::-1])
    selected, seen, pages = [], set(), set()
    while candidates and len(selected) < k:
        best = max(candidates, key=lambda i: scores[i] - (0.06 if (rows[i]['document_id'], rows[i]['page']) in pages else 0))
        candidates.remove(best)
        row = rows[best]
        fingerprint = normalize(row['text'])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        pages.add((row['document_id'], row['page']))
        selected.append({**row, 'score': float(scores[best])})
    return selected


def summary_sample(rows, limit):
    groups = {}
    for row in rows:
        groups.setdefault(row['document_id'], []).append(row)
    queues = [[group[i] for i in np.linspace(0,len(group)-1,min(limit,len(group)),dtype=int)] for group in groups.values()]
    chosen = []
    while queues and len(chosen) < limit:
        for queue in list(queues):
            if len(chosen) == limit:
                break
            chosen.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return chosen


def pack_context(rows, budget, question=''):
    """Reserve source headers and take windows around question terms."""
    remaining = max(0,budget)
    packed = []
    for index, row in enumerate(rows):
        header_cost = len(row['filename']) + 60
        allowance = min(len(row['text']), remaining // (len(rows)-index) - header_cost)
        if allowance <= 0:
            continue
        text = row['text']
        # Source text remains unchanged; only choose a relevant substring.
        terms = [term for term in re.findall(r'\w+',question) if len(term)>2]
        positions = [text.find(t) for t in terms if text.find(t)>=0]
        start = max(0,min(positions)-allowance//4) if positions else 0
        start = min(start,max(0,len(text)-allowance))
        snippet = text[start:start+allowance]
        packed.append({**row,'text':snippet})
        remaining -= len(snippet)+header_cost
    return packed


def new_thread():
    return secrets.token_hex(8), {'title':'گفتگوی تازه','messages':[]}


def ensure_threads(state, permission_key):
    old = state.get('permission_key')
    keep = old and old[:3] == permission_key[:3] and {d[0] for d in old[3]}.issubset({d[0] for d in permission_key[3]})
    if (old != permission_key and not keep) or not state.get('threads'):
        key, thread = new_thread()
        state['threads'] = {key:thread}
        state['active_thread'] = key
        state['permission_key'] = permission_key
        state.pop('retrieval_cache',None)
        state.pop('query_cache',None)
    state['permission_key'] = permission_key
    state['messages'] = state['threads'][state['active_thread']]['messages']


def add_thread(state):
    key, thread = new_thread()
    state['threads'][key] = thread
    state['active_thread'] = key
    state['messages'] = thread['messages']
    # Keep per-session memory bounded; old threads are evicted only at the limit.
    if len(state['threads']) > 20:
        state['threads'].pop(next(iter(state['threads'])))
    return key
