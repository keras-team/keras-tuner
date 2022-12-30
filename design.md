# GridSearch Redesign

## Things to consider:

* Conditional space.
* Lazy discovery of hps.
* Concurrent calls.

With these things in mind, even a simple grid search algorithm can be hard.

## Overall process

We populate all the value sets (only for the discovered hps, not including the
not discovered ones) at the begining and put them in a queue. This queue is for
new `populate_space()` requests to fetch from it. When a trial is finished, we
check if there are more combinations between this finished one and its original
next trial. If so, we put all of them into the queue. To check if anything in
between two trials, we also need to maintain a linked list of trials sorted in
ascending order to get the trial next to it in the combination order. It use
linked list because we will keep inserting the new combinations in between
trials to maintain the ascending order.

Pseudo code:
```py
# Try to exhaust all combinations between a1 & a2:
while next_combination(a1) < a2:
    new_a1 = next_combination(a1)
    queue.append(new_a1)
    link_list.insert_after(item=new_a1, pos=a1)
    a1 = new_a1
```

## Compare two sets of values

To achieve the aboe, we should have a function to compare two set of values,
which one is larger in their combination order to sort them in ascending order.

When comparing, only the active values should be considered. We compare from the
left most to the right most. The first different value decides who is larger. 

If the they have a difference set of values due to different
conditional scope activation, it still works since the parent hp should be
different, which is on the left of the first different activated hp.

## A corner case example

We should also make it work when comparing a finished trial and a ongoing trial
(new hps not reported back yet). The above logic should resolve most of the
cases, but the case one combination is the prefix of another combination needs a
special casing here. In this "prefix" case, we should judge the longer one as
larger.

This decision is for the following use case:
```py
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        hp.Int("hp1", 0, 5)
        ...
    
    def fit(self, hp, model, **kwargs):
        hp.Int("hp2", 0, 5, default=0)
        ...
```

In the first round of parallel `Oracle.create_trial()`, the oracle never know
`hp2`. It populates `hp1` from 0 to 5. If the trial with `hp1=0` (`hp2=0` was
discovered during the trial) finished first. It starts to populate `hp1=0,
hp2=1 to 5` by calling `next_combination()`, then it would get `hp1=1,hp2=0`,
which is actually equal to `hp1=1` (the second trial) as the `hp2=0` will be
discovered during the trial. So, these 2 trials are not equal in values does
not mean they are not equal. In this case, we need to see if `hp1=1,hp=0` is
greater equal than `hp1=1`, and we judge that it is.

Here is the general description. Trial a1 < a2 and they are next to each other
before a1 got some new hps when finished. a2 keeps running for a long time. a1
start to produce the combinations whose order is between a1 and a2 by changing
the values of the newly appeared hps. The new value sets are produce using get
next combination mechanism. When a new set of values are produced we need to
judge whether all the combinations between a1 and a2 are exhausted, which is
decided by if the newly produced values is larger than a2. This is when a2 is
the prefix of a newly produced set of values, we have exhausted the values.

With the comparison function above, we achieved the following. Given 2 finished
trials, we can tell if there are not tried value sets between them by
`next_combination(a1) < a2` (If true, there are sets between a1 & a2) even when
a2 is not finished.

So even when a trial is finished with new hps, we can start to produce more trials
between it and its original next trial. This is good for parallel.

## Caveat

Do not use `Oracle._tried_so_far`, which did not count the new hps of a1 in a2. Even
when it is exhausted, the new set will not equal to a2 due to the new hps.
