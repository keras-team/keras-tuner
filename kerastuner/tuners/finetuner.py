from tensorflow import keras
from tensorflow.keras import layers
from ..engine import hyperparameters as hp_module
from ..engine import multi_execution_tuner
from ..engine import oracle as oracle_module
from ..engine import trial as trial_lib

class FineTuner(multi_execution_tuner.MultiExecutionTuner):
    """FineTuner automatically transfer-learn and fine-tunes a hypermodel that
       start with pretrained weights.

    # Arguments:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class (or callable that takes
            hyperparameters and returns a Model instance).
            This hypermodel should: 
            1. load pretrained weight for feature extractor, and
            2. freeze all layers with pretrained weights.
            If those conditions are not met, FineTuner will simply be
            the ordinary tuner but fit model twice for each execution.
        **kwargs: Keyword arguments relevant to all `MultiExecutionTuner`
            subclasses. Please see the docstring for `MultiExecutionTuner`.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 unfreeze_factor_range=None,
                 **kwargs):
        super(FineTuner, self).__init__(
            oracle, hypermodel, **kwargs)
        if not unfreeze_factor_range:
            self.unfreeze_min = 0
            self.unfreeze_max = 1
        elif len(unfreeze_factor_range) == 1:
            self.unfreeze_min = 0
            self.unfreeze_max = unfreeze_factor_range
        elif len(unfreeze_factor_range) == 2:
            self.unfreeze_min = min(unfreeze_factor_range)
            self.unfreeze_max = max(unfreeze_factor_range)
            

    def _fit(self, model, hp, *fit_args, **fit_kwargs):
        model.fit(*fit_args, **fit_kwargs)
        unfreeze(model, 
            factor=hp.Float('unfreeze_factor',
                self.unfreeze_min,
                self.unfreeze_max,
                step=0.1,
                default=self.unfreeze_max))
        return model.fit(*fit_args, **fit_kwargs)


def unfreeze(model, factor=1, verbose=0, unfreeze_bn=False, biconnected=True):
    """Unfreeze a fraction of layers in a model.

    # Arguments:
        model: The model to unfreeze.
        factor: The fraction of layers to be unfrozen. 1 is unfreeze all
            (entire model is trainable), 0 is freeze all.
        biconnected: Whether to require each biconnected component is either
            fully frozen or unfrozen. When set to True, the `factor` is rounded
            down to the nearest fraction such that the boundary between frozen
            and unfrozen is always on an articulation point.
        verbose: Verbosity level. 0: quiet. 1: print number of layers unfrozen
            (always including batch normalization layers). 2: print name and
            info of all layers frozen or unfrozen.
        unfreeze_bn: Whether to unfreeze batch normalization layers.
    """
    if factor > 1:
        raise ValueError('Unfreeze factor should be between 0 and 1.'
                         'Received {}.'.format(factor))
    if verbose >= 1:
        print('Unfreezing {} of all layers.'.format(factor))

    # Currently only implemented for respecting biconnected components,
    # because this is considered a better practice than arbitrarily
    # freezing.
    if not biconnected:
        raise NotImplemented

    model.trainable = True

    num_layers = len(model.layers)
    target_unfreeze = num_layers * factor

    ps_model = PseudoSequential(model)

    num_unfrozen = 0
    num_bn = 0

    for (i, block) in enumerate(ps_model.blocks):
        if num_unfrozen + len(block) > target_unfreeze:
            break
    
        for l in block:
            if isinstance(l, layers.BatchNormalization):
                num_bn += 1
                l.trainable = unfreeze_bn
                if verbose == 2:
                    print('Layer {} is batch normalization. Unfreezing: {}'
                          .format(l.name, unfreeze_bn))
            else:
                num_unfrozen += 1
                l.trainable = True
                if verbose == 2:
                    print('Unfreezing layer {}.'.format(l.name))
    else:
        i += 1

    for block in ps_model.blocks[i:]:
        for l in block:
            if isinstance(l, layers.BatchNormalization):
                num_bn += 1
                l.trainable = unfreeze_bn
                if verbose == 2:
                    print('Layer {} is batch normalization. Unfreezing: {}'
                          .format(l.name, unfreeze_bn))
            else:
                l.trainable = False
                if verbose == 2:
                    print('Layer {} is left frozen.'.format(l.name))

    if verbose >= 1:
        print('Total layers: {}; Batch normalization layers: {}; '
              'Unfrozen (not including batch normalization: {}.'
              .format(num_layers, num_bn, num_unfrozen))

class PseudoSequential():
    """Partitions a Keras model into blocks that are sequential.

    # Description:
       A block is a list of layers, such that each block is only
       connected to two other blocks, except for the first (output)
       and last (input) block that are only connected to one
       neighboring block.
       Attribute `blocks` is the list of blocks, starting
       from the output block to the input block. Each layer appears
       exactly one time in `blocks`. As a convention, articulation
       points belong to the block that is closer to the input layer.

    # Illustration:
        #    InputLayer         #
        #     /    \            #
        #    |    layer2        #
        #    |     /    \       #
        #    |  layer3  layer4  #
        #    \    /    /        #
        #     \   |   /         #
        #      layer5           #
        #       /  \            #
        #  layer6  layer7       #
        #      \   /            #
        #     layer8            #
        #        |              #
        #     PredLayer         #
        Then the partition `self.blocks` will be:
        [[PredLayer],
         [layer8, layer7, layer6],
         [layer5, layer3, layer4, layer2, InputLayer]]
        Note that layer5, an articulation point, belongs to the
        InputLayer's block.

    # Arguments:
        model: A Keras model.

    # Methods:
        find_blocks: partition the model into pseudo sequential blocks.
            It updates and returns `self.blocks`.

    # A note on the algorithm:
        The partitioning algorithm is based on the standard algorithm
        for finding articulation points. However, determining the partition
        along with the DFS search relies on the fact that the graph, although
        being undirected, has a definite starting and end point and the DFS
        traversal visits outputs of each layer before the inputs.
    """

    def __init__(self, model):
        self.model = model
        self.blocks = []
        self.find_blocks()


    def find_blocks(self):
        """Partition the model and update `self.blocks`."""
        self.blocks = []

        if not self.model.layers:
            return self.blocks

        if isinstance(self.model, keras.Sequential):
            # For sequential model, every layer is a block.
            # We reverse order it so that the root (input)
            # is the last of the list of blocks.
            self.blocks = [[l] for l in self.model.layers[::-1]]
            return self.blocks

        # initiate variables for DFS-tree traversal
        # dfs_index represents the order each layer is traversed in DFS-tree
        self.dfs_index_counter = 0
        self.dfs_index = {}

        # dfs_stack: pushes a tuple `(layer, [])` into the stack when visited,
        # where the list is for keeping bubbles attached to this layer.
        # A bubble is a list of layers that are only connect to one layer.
        # A layer and the bubbles attached to it are treated as one
        # irreducible element in partitioning.
        # Recursion invariance: layers above current layer in the stack
        # are members of the DFS-subtree rooted at the current layer.
        self.dfs_stack = []
  
        # low-function: 
        # low[u] is the smallest dfs_index of all immediate neighbors
        # of all DFS-tree descendent of u. It represents the lowest
        # dfs_index that any back-edge in the DFS-subtree rooted at u
        # connects to.
        self.low = {}
  
        self.dfs_tree_parent = {}
        root_layer = self.model.layers[0]
        self.dfs_tree_parent[root_layer] = None

        self.dfs_tree_top = self.model.layers[-1]

        self.dfs(root_layer)
        return self.blocks
  
    def dfs(self, curr):
        """Recursion for DFS_tree traversal for `find_blocks`.

        # Arguments:
            curr: A Keras layer that is currently visiting.
        """
        self.dfs_index[curr] = self.dfs_index_counter
        self.low[curr] = self.dfs_index_counter
        self.dfs_index_counter += 1
        self.dfs_stack.append((curr, []))

        child_count = 0
        for child in iterate_neighbor_layers(curr):
            self.dfs_tree_parent[child] = curr
            if child == self.dfs_tree_parent[curr]:
                continue

            child_count += 1
            if child in self.dfs_index:
                # Child is visited, so the edge is a backedge.
                # Update low-function.
                self.low[curr] = min(self.low[curr], self.dfs_index[child])
            else:
                # Child has not been visited, so it is in the DFS-subtree
                self.dfs(child)
                self.low[curr] = min(self.low[curr], self.low[child])
                if self.low[child] >= self.dfs_index[curr]:
                    # The subtree is a block or a bubble.

                    subtree = []
                    while self.dfs_stack and self.dfs_stack[-1][0] != curr:
                        l, bubble = self.dfs_stack.pop()
                        subtree.append(l)
                        subtree.extend(bubble)
                    if self.dfs_tree_top in subtree:
                        # The subtree is a block if it includes the tree-top
                        self.blocks.append(subtree)
                        # Update tree top as the subtree is trimmed.
                        self.dfs_tree_top = curr
                    else:
                        # The subtree is a bubble attached to `curr` if it does
                        # not include the tree-top
                        self.dfs_stack[-1][1].extend(subtree)

        if not child_count:
            # No neighbor other than DFS-tree parent: [curr] is bubble
            # unless it is root or top.
            if self.dfs_tree_top != curr and self.model.layers[0] != curr:
                self.dfs_stack.pop()
                self.dfs_stack[-1][1].append(curr)


def iterate_neighbor_layers(layer):
    """Create a generator of all neighboring layers of a given layer."""
    visited = {layer}

    for n in layer.outbound_nodes:
        l = n.outbound_layer
        if l not in visited:
            visited.add(l)
            yield l

    for n in layer.inbound_nodes:
        for l, _, _, _ in n.iterate_inbound():
            if l not in visited:
                visited.add(l)
                yield l
