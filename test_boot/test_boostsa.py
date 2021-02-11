#!/usr/bin/env python
import sys
sys.path.append('/home/runner/work/boostsa/boostsa/')

from boostsa import Bootstrap
boot = Bootstrap()
boot.test(targs='test_boot/h0.0/targs.txt', h0_preds='test_boot/h0.0/preds.txt', h1_preds='test_boot/h1.0/preds.txt')

boot = Bootstrap()

boot.feed(h0='h0',          exp_idx='h0.0', preds='test_boot/h0.0/preds.txt', targs='test_boot/h0.0/targs.txt', idxs='test_boot/h0.0/idxs.txt')
boot.feed(h0='h0',          exp_idx='h0.1', preds='test_boot/h0.1/preds.txt', targs='test_boot/h0.1/targs.txt', idxs='test_boot/h0.1/idxs.txt')
boot.feed(h0='h0', h1='h1', exp_idx='h1.0', preds='test_boot/h1.0/preds.txt', targs='test_boot/h1.0/targs.txt', idxs='test_boot/h1.0/idxs.txt')
boot.feed(h0='h0', h1='h1', exp_idx='h1.1', preds='test_boot/h1.1/preds.txt', targs='test_boot/h1.1/targs.txt', idxs='test_boot/h1.1/idxs.txt')
boot.run(n_loops=100, sample_size=.2, verbose=True)

next_boot = Bootstrap()
next_boot.loadjson('outcomes.json')

next_boot.feed(h0='h0', h1='h2', exp_idx='h2.0', preds='test_boot/h2.0/preds.txt', targs='test_boot/h2.0/targs.txt', idxs='test_boot/h2.0/idxs.txt')
next_boot.feed(h0='h0', h1='h2', exp_idx='h2.1', preds='test_boot/h2.1/preds.txt', targs='test_boot/h2.1/targs.txt', idxs='test_boot/h2.1/idxs.txt')
next_boot.run(n_loops=100, sample_size=.2, verbose=True)



