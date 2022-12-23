# GridSearch Redesign

## A potential bug to fix
Adding to `_tried_so_far` should be done again when the trial is updated or
ended since there may be new hyperparameters.

## Grid Search

Things to consider:
* Conditional space.
* Lazy discovery of hps.
* Concurrent calls.

With these things in mind, even a simple grid search algorithm can be hard.


## Compare two sets of values

We should have a function to compare two set of values, which one is larger in
their combination order. To sort them in ascending order.

When comparing, only the active
values should be considered. We compare from the left most to the right most.
The first different value decides who is larger. 

If the they have a difference set of values due to different
conditional scope activation, it still works since the parent hp should be
different, which is on the left of the first different activated hp.

We should also make it work when comparing a finished trial and a ongoing trial
(new hps not reported back yet).  The above logic should resolve most of the
cases, but the case one combination is the prefix of another combination needs a
special casing here.  In this "prefix" case, we should judge the longer one as
larger.

This decision is for the following use case:
Trial a1 < a2 and they are next to each other before a1 got some new hps when
finished. a2 keeps running for a long time. a1 start to produce the
combinations whose order is between a1 and a2 by changing the values of the
newly appeared hps.  The new value sets are produce using get next combination
mechanism.  When a new set of values are produced we need to judge whether all
the combinations between a1 and a2 are exhausted, which is decided by if the
newly produced values is larger than a2. This is when a2 is the prefix of a
newly produced set of values, we have exhausted the values.

## Overall process

With the comparison function above, we achieved the following.  Given 2 finished
trials, we can tell if there are not tried value sets between them by
`next_combination(a1) < a2` (If true, there are sets between a1 & a2) even when
a2 is not finished.

Do not use `_tried_so_far`, which did not count the new hps of a1 in a2. Even
when it is exhausted, the new set will not equal to a2 due to the new hps.

So even when a trial is finished with new hps, we can start to produce more trials
between it and its original next trial. This is good for parallel.

We populate all the value sets at the begining and put them in a queue.  This
queue is for new `populate_space()` requests to fetch from it.  When a trial is
finished with new hps (if we can check it finished with new hps, we check. If
cannot check, we do the following for all finished trials.), we check if there
are more combinations between this finished one and its original next trial.  If
so, we put all of them into the queue.  To check if anything in between two
trials, we also need to maintain a linked list of trials sorted in ascending
order to get the trial next to it in the combination order.  It use linked list
because we will keep inserting the new combinations in between trials to
maintain the ascending order.