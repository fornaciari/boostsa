boostsa - BOOtSTrap SAmpling in pyhton
======================================

.. image:: https://img.shields.io/pypi/v/boostsa.svg
        :target: https://pypi.python.org/pypi/boostsa

.. image:: https://img.shields.io/github/license/fornaciari/boostsa
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://github.com/fornaciari/boostsa/workflows/Python%20Package/badge.svg
        :target: https://github.com/fornaciari/boostsa/actions

.. image:: https://readthedocs.org/projects/boostsa/badge/?version=latest
        :target: https://boostsa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1pkbjouxjub9ve0PlVZaW_we_r1hz6Hf-#scrollTo=TGj4udXVb6Ji
    :alt: Open In Colab

Intro
-----

boostsa - BOOtSTrap SAmpinlg - is a tool to compute bootstrap sampling significance test,
even in the pipeline of a complex experimental design...

- Free software: MIT license
- Documentation: https://boostsa.readthedocs.io.

Google colab
------------

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1pkbjouxjub9ve0PlVZaW_we_r1hz6Hf-#scrollTo=TGj4udXVb6Ji
    :alt: Open In Colab

+----------------------------------------------------------------+--------------------+
| Name                                                           | Link               |
+================================================================+====================+
| You can try boostsa here:                                      | |colab1|           |
+----------------------------------------------------------------+--------------------+


Installation
------------

.. code-block:: bash

    pip install -U boostsa

Getting started
---------------

First, import ``boostsa``:

.. code-block:: python

    from boostsa import Bootstrap

Then, create a boostrap instance. You will use it to store your experiments' results and to compute the bootstrap sampling significance test:

.. code-block:: python

    boot = Bootstrap()


Inputs
^^^^^^

The assumption is that you ran at least two classification task experiments, which you want compare.

One is your *baseline*, or *control*, or *hypothesis 0* (*h0*).

The other one is the *experimental condition* that hopefully beats the baseline, or *treatment*, or *hypothesis 1* (*h1*).

You compare the *h0* and *h1* predictions against the same targets.

Therefore, *h0 predictions*, *h1 predictions* and *targets* will be the your ``Bootstrap`` instance's data inputs.


Outputs
^^^^^^^

By defalut, boostsa produces two output files:

- ``results.tsv``, that contains the experiments' performance and the (possible) significance levels;
- ``outcomes.json``, that contains targets and predictions for all the experimental conditions.

You can define the outputs when you create the instance, using the following parameters:

- ``save_results``, type: ``bool``, default: ``True``. This determines if you want to save the results.
- ``save_outcomes``, type: ``bool``, default: ``True``. This determines if you want to save the experiments' outcomes..
- ``dir_out``, type: ``str``, default: ``''``, that is your working directory. This indicates the directory where to save the results.

For example, if you want to save only the results in a particular folder, you will create an instance like this:

.. code-block:: python

    boot = Bootstrap(save_outcomes=False, dir_out='my/favourite/directory/')


Test function
-------------

In the simplest conditions, you will run the bootstrap sampling significance test with the ``test`` function.
It takes the following inputs:

- ``targs``, type: ``list`` or ``str``. They are the targets, or *gold standard*, that you use as benchmark to measure the *h0* and *h1* predictions' performance. They can be a **list of integers**, representing the labels' indexes for each data point, or a string. In such case, the string will be interpreted as the **path** to a text file containing a single integer in each row, having the same meaning as for the list input.
- ``h0_preds``, type: ``list`` or ``str``. The *h0* predictions, in the same formats of ``targs``.
- ``h1_preds``, type: ``list`` or ``str``. The *h1* predictions, in the same formats as above.
- ``h0_name``, type: ``str``, default: ``h0``. Expression to describe the *h0* condition.
- ``h1_name``, type: ``str``, default: ``h1``. Expression to describe the *h1* condition.
- ``n_loops``, type: ``int``, default: ``100``. Number of iterations for computing the bootstrap sampling.
- ``sample_size``, type: ``float``, default: ``.1``. Percentage of data points sampled, with respect to their whole set. The admitted values range between 0.05 (5%) and 0.5 (50%).
- ``verbose``, type: ``bool``, default: ``False``. If true, the experiments' performance is shown.

For example:

.. code-block:: python

    boot.test(targs='../test_boot/h0.0/targs.txt', h0_preds='../test_boot/h0.0/preds.txt', h1_preds='../test_boot/h1.0/preds.txt', n_loops=1000, sample_size=.2, verbose=True)

The ouput will be:

.. sourcecode::

    total size............... 1000
    sample size.............. 200
    targs count:              ['class 0 freq 465 perc 46.50%', 'class 1 freq 535 perc 53.50%']
    h0 preds count:           ['class 0 freq 339 perc 33.90%', 'class 1 freq 661 perc 66.10%']
    h1 preds count:           ['class 0 freq 500 perc 50.00%', 'class 1 freq 500 perc 50.00%']
    h0 F-measure............. 67.76   h1 F-measure............. 74.07   diff... 6.31
    h0 accuracy.............. 69.0    h1 accuracy.............. 74.1    diff... 5.1
    h0 precision............. 69.94   h1 precision............. 74.1    diff... 4.16
    h0 recall................ 67.96   h1 recall................ 74.22   diff... 6.26
    bootstrap: 100%|███████████████████████████| 1000/1000 [00:07<00:00, 139.84it/s]
    count sample diff f1   is twice tot diff f1....... 37   / 1000    p < 0.037  *
    count sample diff acc  is twice tot diff acc...... 73   / 1000    p < 0.073
    count sample diff prec is twice tot diff prec..... 111  / 1000    p < 0.111
    count sample diff rec  is twice tot diff rec ..... 27   / 1000    p < 0.027  *
    Out[3]:
           f1 diff_f1 sign_f1   acc diff_acc sign_acc   prec diff_prec sign_prec    rec diff_rec sign_rec
    h0  67.76                  69.0                    69.94                      67.96
    h1  74.07    6.31       *  74.1      5.1           74.10      4.16            74.22     6.26        *

That's it!

For more complex experimental designs and technical/ethical considerations, please refer to the documentation page.

