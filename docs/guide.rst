Guide
=====

.. image:: https://img.shields.io/github/license/fornaciari/boostsa
        :target: https://lbesson.mit-license.org/
        :alt: License

Intro
-----

boostsa - BOOtSTrap SAmpinlg - is a tool to compute bootstrap sampling significance test,
even in the pipeline of a complex experimental design.

For the theoretical aspects of Bootstrap sampling, please refer to the paper:

 Søgaard, A., Johannsen, A., Plank, B., Hovy, D., & Alonso, H. M. (2014, June).
 *What’s in a p-value in NLP?*.
 In Proceedings of the eighteenth conference on computational natural language learning (pp. 1-10).

See also :ref:`tethics`.

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
Where you see two asterisks **\*\*** you have a significance with :math:`p \le .01`; one asterisk **\*** indicates siginficance with :math:`p \le .05`.


boostsa in a pipeline
---------------------

Your use case is probably much more complex than that of the previous example.
You probably run multiple experiments, where you want to compare several *h0* baselines with many *h>0* (*h1*, *h2*...) experimental conditions.
Also, for each baseline/experimental condition you are maybe running many experiments, let's say to reduce the random initialization variability.

You would like to store the experiments' results directly when you run them, and to compute bootstrap sampling in the same pipeline.
You can do so with the functions ``feed`` and ``run``.

The ``feed`` function takes the following inputs:

- ``h0``, type: ``str``. This is an expression that gives a name to the *h0* experiment. It must be provided both for the *h0* experiments, and for the *h>0* experiments which have to be compared with that *h0* condition.
- ``h1``, type: ``str``, default: ``None``. This is an expression that gives a name to the *h>0* experiment.
- ``exp_idx``, type: ``str``, default: ``None``. This is an expression that identifies the single experiment, in case multiple experiments are carried out within the same experimental condition. It could contain, for example, the directory containing the outputs of such experiments.
- ``targs``, type: ``list`` or ``str``. Similarly to the ``test`` function, they are the targets and can be a **list of integers**, representing the labels' indexes for each data point, or a string. In such case, the string will be interpreted as the **path** to a text file containing a single integer in each row.
- ``preds``, type: ``list`` or ``str``. The predictions, in the same formats of ``targs``.
- ``idxs``, type: ``list`` or ``str``. Similar to the other inputs, it can be a list or a string representing the path to a file containing an integer number in each row. During the training, you could have shuffled your data points. The data points order does not affect the bootstrap sampling, but you could want to store the shuffled indexes, to link your predictions to your original data points in a second moment. You can provide these indexes to this parameter.
- ``epochs``, type:``int``. This is an integer number, corresponding to the number of epochs of the experiment. This variable will be included in the bootstrap outputs. In case of multiple experiments for experimental condition, with early stopping at different epochs, the average will be reported.

The ``run`` function takes the three inputs:

- ``n_loops``, type: ``int``, default: ``100``. Number of iterations for computing the bootstrap sampling.
- ``sample_size``, type: ``float``, default: ``.1``. Percentage of data points sampled, with respect to their whole set. The admitted values range between 0.05 (5%) and 0.5 (50%).
- ``verbose``, type: ``bool``, default: ``False``. If true, the experiments' performance is shown.

This is an example of these functions' use:

.. code-block:: python

    # you load the package

    from boostsa import Bootstrap

    # you create a bootstrap instance:

    boot = Bootstrap()

    # You run your first experiment, to compute your baseline performance.
    # You have your targets list 'tatgets', and you obtain your predictions list 'h0_exp1_preidctions'
    # You feed your bootstrap instance with your lists:

    boot.feed(h0='h0', exp_idx='h0.1', preds=h0_exp1_preidctions, targs=tatgets)

    # You could have re-run the same experiment, with different weigths' random initialization.
    # You keep on feeding your bootstrap instance with your outputs:

    boot.feed(h0='h0', exp_idx='h0.2', preds=h0_exp2_preidctions, targs=targets)

    # Following the h0 experiments, you run the experiments that you want to compare with the first ones.
    # Note that, in these cases, you have to label both the experimental condition and the baseline you want to compare with.

    boot.feed(h0='h0', h1='h1', exp_idx='h1.1', preds=h1_exp1_preidctions, targs=targets)
    boot.feed(h0='h0', h1='h1', exp_idx='h1.2', preds=h1_exp2_preidctions, targs=targets)

    # When you ran all the experiments, you can compute the bootstrap sampling test:

    boot.run(n_loops=1000, sample_size=.2, verbose=True)

The output will look like this:

.. sourcecode::

    ################################################################################
    start: 2021/02/09 16:21:26
    ################################################################################
    h0.0                                                         acc 69.0    F 67.76
    h0.1                                                         acc 72.6    F 72.59
    ################################################################################
    h0   vs   h1
    h1.0                                                         acc 74.1    F 74.07
    h1.1                                                         acc 73.0    F 72.99
    total size............... 2000
    sample size.............. 400
    targs count:              ['class 0 freq 930 perc 46.50%', 'class 1 freq 1070 perc 53.50%']
    h0 preds count:           ['class 0 freq 892 perc 44.60%', 'class 1 freq 1108 perc 55.40%']
    h1 preds count:           ['class 0 freq 1051 perc 52.55%', 'class 1 freq 949 perc 47.45%']
    h0 F-measure............. 70.57   h1 F-measure............. 73.55   diff... 2.98
    h0 accuracy.............. 70.8    h1 accuracy.............. 73.55   diff... 2.75
    h0 precision............. 70.66   h1 precision............. 73.79   diff... 3.13
    h0 recall................ 70.52   h1 recall................ 73.85   diff... 3.33
    bootstrap: 100%|███████████████████████████| 1000/1000 [00:08<00:00, 123.85it/s]
    count sample diff f1   is twice tot diff f1....... 61   / 1000    p < 0.061
    count sample diff acc  is twice tot diff acc...... 67   / 1000    p < 0.067
    count sample diff prec is twice tot diff prec..... 54   / 1000    p < 0.054
    count sample diff rec  is twice tot diff rec ..... 44   / 1000    p < 0.044  *
       mean_epochs    acc diff_acc sign_acc   prec diff_prec sign_prec    rec diff_rec sign_rec     f1 diff_f1 sign_f1
    h0        None  70.80                    70.66                      70.52                    70.57
    h1        None  73.55     2.75           73.79      3.13            73.85     3.33        *  73.55    2.98
    ################################################################################
    end: 2021/02/09 16:21:34  - time elapsed: 00:00:08
    ################################################################################

With ``feed`` and ``run`` you can store several *h0* condtions and to compare them with several *h>0* condition.
For each condition, you can run multiple experiments.

**Note:** the *h0* and *h>0* that you compare, must have equal targets. Otherwise an error will be raised.

Resuming outcomes
-----------------

Lastly, you could have run bootstrap sampling and stored the experiments' outcomes in your ``outcomes.json`` file.
After that, you want to add new experiment and to compare them with the previous ones.
Or, simply, you want re-run bootstrap sampling with different parameters.
You can load and keep on feeding the json file with the ``loadjson`` function, that takes as input the path to the ``outcomes.json`` file:

.. code-block:: python

    next_boot = Bootstrap()
    next_boot.loadjson('outcomes.json')
    next_boot.feed(h0='h0', h1='h2', exp_idx='h2.0', preds='test_boot/h2.0/preds.txt', targs='test_boot/h2.0/targs.txt', idxs='test_boot/h2.0/idxs.txt')
    next_boot.feed(h0='h0', h1='h2', exp_idx='h2.1', preds='test_boot/h2.1/preds.txt', targs='test_boot/h2.1/targs.txt', idxs='test_boot/h2.1/idxs.txt')
    next_boot.run(n_loops=1000, sample_size=.2, verbose=True)

That will produce:

.. sourcecode::

    ################################################################################
    start: 2021/02/09 16:21:34
    ################################################################################
    h0.0                                                         acc 69.0    F 67.76
    h0.1                                                         acc 72.6    F 72.59
    ################################################################################
    h0   vs   h1
    h1.0                                                         acc 74.1    F 74.07
    h1.1                                                         acc 73.0    F 72.99
    total size............... 2000
    sample size.............. 400
    targs count:              ['class 0 freq 930 perc 46.50%', 'class 1 freq 1070 perc 53.50%']
    h0 preds count:           ['class 0 freq 892 perc 44.60%', 'class 1 freq 1108 perc 55.40%']
    h1 preds count:           ['class 0 freq 1051 perc 52.55%', 'class 1 freq 949 perc 47.45%']
    h0 F-measure............. 70.57   h1 F-measure............. 73.55   diff... 2.98
    h0 accuracy.............. 70.8    h1 accuracy.............. 73.55   diff... 2.75
    h0 precision............. 70.66   h1 precision............. 73.79   diff... 3.13
    h0 recall................ 70.52   h1 recall................ 73.85   diff... 3.33
    bootstrap: 100%|███████████████████████████| 1000/1000 [00:08<00:00, 123.39it/s]
    count sample diff f1   is twice tot diff f1....... 73   / 1000    p < 0.073
    count sample diff acc  is twice tot diff acc...... 80   / 1000    p < 0.08
    count sample diff prec is twice tot diff prec..... 56   / 1000    p < 0.056
    count sample diff rec  is twice tot diff rec ..... 47   / 1000    p < 0.047  *
    ################################################################################
    h0   vs   h2
    h2.0                                                         acc 71.7    F 71.32
    h2.1                                                         acc 71.4    F 71.2
    total size............... 2000
    sample size.............. 400
    targs count:              ['class 0 freq 930 perc 46.50%', 'class 1 freq 1070 perc 53.50%']
    h0 preds count:           ['class 0 freq 892 perc 44.60%', 'class 1 freq 1108 perc 55.40%']
    h2 preds count:           ['class 0 freq 871 perc 43.55%', 'class 1 freq 1129 perc 56.45%']
    h0 F-measure............. 70.57   h2 F-measure............. 71.27   diff... 0.7
    h0 accuracy.............. 70.8    h2 accuracy.............. 71.55   diff... 0.75
    h0 precision............. 70.66   h2 precision............. 71.46   diff... 0.8
    h0 recall................ 70.52   h2 recall................ 71.2    diff... 0.68
    bootstrap: 100%|████████████████████████████| 1000/1000 [00:12<00:00, 81.14it/s]
    count sample diff f1   is twice tot diff f1....... 367  / 1000    p < 0.367
    count sample diff acc  is twice tot diff acc...... 326  / 1000    p < 0.326
    count sample diff prec is twice tot diff prec..... 334  / 1000    p < 0.334
    count sample diff rec  is twice tot diff rec ..... 369  / 1000    p < 0.369
       mean_epochs    acc diff_acc sign_acc   prec diff_prec sign_prec    rec diff_rec sign_rec     f1 diff_f1 sign_f1
    h0        None  70.80                    70.66                      70.52                    70.57
    h1        None  73.55     2.75           73.79      3.13            73.85     3.33        *  73.55    2.98
    h2        None  71.55     0.75           71.46       0.8            71.20     0.68           71.27     0.7
    ################################################################################
    end: 2021/02/09 16:21:55  - time elapsed: 00:00:20
    ################################################################################

.. _tethics:

Technical and ethic considerations
----------------------------------

The significance test is a critical metric. It makes the difference between the experiments' success or failure. Also, significance is *not* a gray-scaled measure: the *p*-value is significant or not.

However, the parameters' choice strongly affects the bootstrap sampling test's outcome.

Tuning the iterations' number is easy: the more the better.
For fast evaluatiions, the ``boostsa`` default iterations' number is set to 100, but my advice is to rely only on results based on at least **1000 iterations**.

The sample size, in terms of total amount of cases' percentage, is a more opinable parameter.
In literature, I only found the (not surprising) advice to not use a too small sample, because "*with small sample sizes, there is a risk that the calculated p-value will be artificially low—simply because the bootstrap samples are too similar*" (Søgaard et al., 2014).

However, this is actually a case that occurs only with tiny samples.

In fact, the opposite is also true: the *p*-value can be artificially low even for big samples, when their distribution becomes too similar to that of the whole data points.

To limit these possible test's misuses, ``boostsa`` only allows a **sample size ranging from 0.05 (5%) to 0.5 (50%)**.
However, this could be not sufficient to prevent incorrect results.

Therefore I invite you to:

- tune the parameters responsibly;
- always report both the *p*-values and the relative parameters.

If you are aware of any better indication that can be given, please let me know!

