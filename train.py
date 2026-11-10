#!/usr/bin/env python3
"""Tiny causal Transformer, trained with NumPy and explicit backpropagation.
Run: python3 train.py [--steps 1600] [--check-grad]
Exports model.json and parity.json for the dependency-free browser engine.
Architecture: one post-LN decoder block, 8 features, 2 attention heads, FFN 16.
No dropout; no encoder or cross-attention in this small causal demonstration.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

SENTENCES = [
    'the cat sat on the mat', 'the cat slept on the mat',
    'the dog sat on the rug', 'the dog slept on the rug',
    'the bird sat on the branch', 'the bird stood on the branch',
]
VOCAB = ['<bos>', '<eos>'] + sorted(set(' '.join(SENTENCES).split()))
D, HEADS, FF, EPS = 8, 2, 16, 1e-5


def softmax(x):
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def positional(length):
    p = np.arange(length)[:, None]
    i = np.arange(D)[None, :]
    a = p / (10000 ** (2 * (i // 2) / D))
    return np.where(i % 2 == 0, np.sin(a), np.cos(a))


def initialize(seed=42):
    rng = np.random.default_rng(seed)
    shapes = {'embedding': (len(VOCAB), D), 'wq': (D, D), 'wk': (D, D),
              'wv': (D, D), 'wo': (D, D), 'w1': (D, FF), 'w2': (FF, D),
              'readout': (D, len(VOCAB))}
    p = {k: rng.normal(0, .2, shape) for k, shape in shapes.items()}
    for name, size in [('g1', D), ('g2', D)]: p[name] = np.ones(size)
    for name, size in [('n1', D), ('n2', D), ('b1', FF), ('b2', D), ('bias', len(VOCAB))]: p[name] = np.zeros(size)
    return p


def norm(x, g, b):
    inv = 1 / np.sqrt(x.var(axis=-1, keepdims=True) + EPS)
    z = (x - x.mean(axis=-1, keepdims=True)) * inv
    return z * g + b, (z, inv, g)


def norm_back(dy, cache):
    z, inv, g = cache
    v = dy * g
    dx = inv * (v - v.mean(axis=-1, keepdims=True) - z * (v * z).mean(axis=-1, keepdims=True))
    return dx, (dy * z).sum(axis=(0, 1)), dy.sum(axis=(0, 1))


def forward(p, ids, targets=None, backward=False):
    batch, length = ids.shape
    x = p['embedding'][ids] * math.sqrt(D) + positional(length)
    q, k, v = [ (x @ p[name]).reshape(batch, length, HEADS, D//HEADS).transpose(0, 2, 1, 3)
                 for name in ['wq', 'wk', 'wv'] ]
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D//HEADS)
    scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -1e30, scores)
    attention = softmax(scores)
    merged = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, D)
    y, cache1 = norm(x + merged @ p['wo'], p['g1'], p['n1'])
    pre = y @ p['w1'] + p['b1']
    hidden = np.maximum(0, pre)
    z, cache2 = norm(y + hidden @ p['w2'] + p['b2'], p['g2'], p['n2'])
    logits = z @ p['readout'] + p['bias']
    if targets is None: return logits
    probs = softmax(logits)
    loss = -np.log(probs.reshape(-1, len(VOCAB))[np.arange(batch*length), targets.ravel()]).mean()
    if not backward: return loss
    grads = {}
    dl = probs.copy().reshape(-1, len(VOCAB))
    dl[np.arange(batch*length), targets.ravel()] -= 1
    dl = dl.reshape(batch, length, len(VOCAB)) / (batch*length)
    def weight_grad(a, b): return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])
    grads['readout'], grads['bias'] = weight_grad(z, dl), dl.sum(axis=(0, 1))
    ds2, grads['g2'], grads['n2'] = norm_back(dl @ p['readout'].T, cache2)
    grads['w2'], grads['b2'] = weight_grad(hidden, ds2), ds2.sum(axis=(0, 1))
    dp = (ds2 @ p['w2'].T) * (pre > 0)
    grads['w1'], grads['b1'] = weight_grad(y, dp), dp.sum(axis=(0, 1))
    ds1, grads['g1'], grads['n1'] = norm_back(ds2 + dp @ p['w1'].T, cache1)
    grads['wo'] = weight_grad(merged, ds1)
    dm = (ds1 @ p['wo'].T).reshape(batch, length, HEADS, D//HEADS).transpose(0, 2, 1, 3)
    da = dm @ v.transpose(0, 1, 3, 2)
    dv = attention.transpose(0, 1, 3, 2) @ dm
    ds = attention * (da - (da * attention).sum(axis=-1, keepdims=True)) / math.sqrt(D//HEADS)
    dq, dk = ds @ k, ds.transpose(0, 1, 3, 2) @ q
    dx = ds1.copy()
    for name, derivative in [('wq', dq), ('wk', dk), ('wv', dv)]:
        derivative = derivative.transpose(0, 2, 1, 3).reshape(batch, length, D)
        grads[name] = weight_grad(x, derivative)
        dx += derivative @ p[name].T
    grads['embedding'] = np.zeros_like(p['embedding'])
    np.add.at(grads['embedding'], ids.ravel(), dx.reshape(-1, D) * math.sqrt(D))
    return loss, grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--check-grad', action='store_true')
    args = parser.parse_args()
    encoded = [[0] + [VOCAB.index(w) for w in sentence.split()] + [1] for sentence in SENTENCES]
    inputs, targets = np.array([x[:-1] for x in encoded]), np.array([x[1:] for x in encoded])
    p = initialize()
    untrained = {k:v.copy() for k,v in p.items()}
    initial = float(forward(p, inputs, targets))
    if args.check_grad:
        _, gradients = forward(p, inputs, targets, True)
        for name in ['embedding', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'readout', 'g1', 'g2']:
            index = tuple(0 for _ in p[name].shape)
            value = p[name][index]
            p[name][index] = value + 1e-5; plus = forward(p, inputs, targets)
            p[name][index] = value - 1e-5; minus = forward(p, inputs, targets)
            p[name][index] = value
            numeric = (plus - minus) / 2e-5
            assert abs(numeric - gradients[name][index]) < 1e-6, (name, numeric, gradients[name][index])
        print('Finite-difference gradient checks passed for 10 parameter groups.')
    m, v = {k:np.zeros_like(x) for k,x in p.items()}, {k:np.zeros_like(x) for k,x in p.items()}
    history = [{'step':0,'loss':initial}]
    for step in range(1, args.steps+1):
        loss, gradients = forward(p, inputs, targets, True)
        magnitude = np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
        for key in p:
            g = gradients[key] / max(1, magnitude)
            m[key] = .9*m[key] + .1*g
            v[key] = .999*v[key] + .001*g*g
            p[key] -= .015*(m[key]/(1-.9**step)) / (np.sqrt(v[key]/(1-.999**step)) + 1e-8)
        if step % 100 == 0: history.append({'step':step,'loss':float(forward(p,inputs,targets))})
    final = float(forward(p, inputs, targets))
    print(f'Cross-entropy: {initial:.5f} → {final:.5f} nats/token.')
    assert final < .4
    root = Path(__file__).parent
    result = {'config':{'dimension':D,'heads':HEADS,'ff':FF,'epsilon':EPS,'context':8,'seed':42},
              'vocab':VOCAB,'sentences':SENTENCES,'history':history,
              'weights':{k:x.tolist() for k,x in p.items()},
              'untrained':{k:x.tolist() for k,x in untrained.items()}}
    (root/'model.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    prompts = ['the cat sat on the','the dog slept on the','the bird stood on the']
    cases = []
    for prompt in prompts:
        ids = [0] + [VOCAB.index(w) for w in prompt.split()]
        logits = forward(p, np.array([ids]))[0]
        cases.append({'prompt':prompt,'logits':logits.tolist()})
        print(prompt, '→', VOCAB[int(np.argmax(logits[-1]))])
    (root/'parity.json').write_text(json.dumps(cases,separators=(',',':'))+'\n')


if __name__ == '__main__': main()
