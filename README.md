# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.1 & 3.2

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (163) 
---------------------------------------------------------------------------|loop #ID
    def _map(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        in_storage: Storage,                                               | 
        in_shape: Shape,                                                   | 
        in_strides: Strides,                                               | 
    ) -> None:                                                             | 
        # Check if tensors are stride-aligned                              | 
        if (                                                               | 
            len(out_strides) == len(in_strides)                            | 
            and np.array_equal(out_strides, in_strides)                    | 
            and np.array_equal(out_shape, in_shape)                        | 
        ):                                                                 | 
            # Fast path - apply function directly using aligned indices    | 
            for i in prange(len(out)):-------------------------------------| #0
                out[i] = fn(in_storage[i])                                 | 
            return                                                         | 
                                                                           | 
        # Slow path - handle non-aligned strides                           | 
        # Calculate size of output                                         | 
        size = len(out_shape)                                              | 
        out_size = 1                                                       | 
        for i in range(size):                                              | 
            out_size *= out_shape[i]                                       | 
                                                                           | 
        # Main parallel loop                                               | 
        for i in prange(out_size):-----------------------------------------| #1
            # Create index buffers per thread                              | 
            out_index = np.empty(len(out_shape), np.int32)                 | 
            in_index = np.empty(len(in_shape), np.int32)                   | 
                                                                           | 
            # Convert position to indices                                  | 
            to_index(i, out_shape, out_index)                              | 
                                                                           | 
            # Calculate output position                                    | 
            o_pos = index_to_position(out_index, out_strides)              | 
            # Map output index to input index                              | 
            broadcast_index(out_index, out_shape, in_shape, in_index)      | 
                                                                           | 
            # Calculate input position                                     | 
            i_pos = index_to_position(in_index, in_strides)                | 
                                                                           | 
            # Apply function                                               | 
            out[o_pos] = fn(in_storage[i_pos])                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (192) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (193) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (235)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (235) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # Check if tensors are stride-aligned                              | 
        if (                                                               | 
            len(out_strides) == len(a_strides) == len(b_strides)           | 
            and np.array_equal(out_strides, a_strides)                     | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        ):                                                                 | 
            # Fast path - apply function directly using aligned indices    | 
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                    | 
            return                                                         | 
                                                                           | 
        # Slow path - handle non-aligned strides                           | 
        # Calculate size of output                                         | 
        size = len(out_shape)                                              | 
        out_size = 1                                                       | 
        for i in range(size):                                              | 
            out_size *= out_shape[i]                                       | 
                                                                           | 
        # Main parallel loop                                               | 
        for i in prange(out_size):-----------------------------------------| #3
            # Create index buffers per thread                              | 
            out_index = np.empty(len(out_shape), np.int32)                 | 
            a_index = np.empty(len(out_shape), np.int32)                   | 
            b_index = np.empty(len(out_shape), np.int32)                   | 
                                                                           | 
            # Convert position to indices and calculate positions          | 
            to_index(i, out_shape, out_index)                              | 
            o_pos = index_to_position(out_index, out_strides)              | 
            broadcast_index(out_index, out_shape, a_shape, a_index)        | 
            a_pos = index_to_position(a_index, a_strides)                  | 
            broadcast_index(out_index, out_shape, b_shape, b_index)        | 
            b_pos = index_to_position(b_index, b_strides)                  | 
            out[o_pos] = fn(a_storage[a_pos], b_storage[b_pos])            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (269) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (270) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (271) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (306)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (306) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # Calculate output size                                    | 
        size = len(out_shape)                                      | 
        out_size = 1                                               | 
        for i in range(size):                                      | 
            out_size *= out_shape[i]                               | 
                                                                   | 
        # Main parallel loop over output positions                 | 
        for i in prange(out_size):---------------------------------| #4
            # Create thread-local index buffers                    | 
            out_index = np.empty(size, np.int32)                   | 
            a_index = np.empty(size, np.int32)                     | 
                                                                   | 
            # Convert position to indices                          | 
            to_index(i, out_shape, out_index)                      | 
                                                                   | 
            # Calculate output position                            | 
            o_pos = index_to_position(out_index, out_strides)      | 
                                                                   | 
            # Copy output index to a_index                         | 
            for j in range(size):                                  | 
                a_index[j] = out_index[j]                          | 
                                                                   | 
            # Initialize reduction with first element              | 
            a_index[reduce_dim] = 0                                | 
            pos = index_to_position(a_index, a_strides)            | 
            reduced = a_storage[pos]                               | 
                                                                   | 
            # Inner reduction loop starting from second element    | 
            for j in range(1, a_shape[reduce_dim]):                | 
                a_index[reduce_dim] = j                            | 
                pos = index_to_position(a_index, a_strides)        | 
                # Apply reduction function                         | 
                reduced = fn(reduced, a_storage[pos])              | 
                                                                   | 
            # Store result                                         | 
            out[o_pos] = reduced                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (324) is 
hoisted out of the parallel loop labelled #4 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(size, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (325) is 
hoisted out of the parallel loop labelled #4 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(size, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (354)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/calebkoresh/Code/MLE/mod3-calebkoresh/minitorch/fast_ops.py (354) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    # Parallelize over batches and rows                                                     | 
    for batch in prange(out_shape[0]):------------------------------------------------------| #5
        for i in range(out_shape[1]):                                                       | 
            for j in range(out_shape[2]):                                                   | 
                # Calculate output position                                                 | 
                out_pos = (                                                                 | 
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]        | 
                )                                                                           | 
                                                                                            | 
                # Initialize accumulator                                                    | 
                acc = 0.0                                                                   | 
                                                                                            | 
                # Inner reduction loop - single multiply per iteration                      | 
                for k in range(a_shape[2]):                                                 | 
                    a_pos = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]    | 
                    b_pos = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]    | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                              | 
                                                                                            | 
                # Single write to output                                                    | 
                out[out_pos] = acc                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

