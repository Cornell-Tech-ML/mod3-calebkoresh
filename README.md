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

# 3.5

## !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

Execution time: 0.8050292519899994 seconds per epoch

Epoch  0  loss  6.493199070818681 correct 29
Epoch  10  loss  2.7580027237714626 correct 46
Epoch  20  loss  2.4828321729146503 correct 46
Epoch  30  loss  1.5920295028918885 correct 47
Epoch  40  loss  1.6197743451798527 correct 47
Epoch  50  loss  1.9466563935373031 correct 49
Epoch  60  loss  0.39866623801649714 correct 47
Epoch  70  loss  0.8853667933571048 correct 49
Epoch  80  loss  1.7211268414917367 correct 48
Epoch  90  loss  2.183933857905331 correct 48
Epoch  100  loss  0.931094372909125 correct 49
Epoch  110  loss  0.6701259784296106 correct 49
Epoch  120  loss  1.0452088095134564 correct 49
Epoch  130  loss  0.28999990902411815 correct 49
Epoch  140  loss  0.9721495901773777 correct 49
Epoch  150  loss  0.4567065266442798 correct 49
Epoch  160  loss  1.2805459043528233 correct 49
Epoch  170  loss  1.1702533334890757 correct 49
Epoch  180  loss  0.7865190778470524 correct 49
Epoch  190  loss  0.1758888856081132 correct 50
Epoch  200  loss  1.3255878981644842 correct 49
Epoch  210  loss  0.20083245434888478 correct 50
Epoch  220  loss  1.153371023762902 correct 49
Epoch  230  loss  0.2925343362299334 correct 49
Epoch  240  loss  0.3470266909056697 correct 50
Epoch  250  loss  1.2030936462035375 correct 49
Epoch  260  loss  1.1295748438935613 correct 49
Epoch  270  loss  0.17303362965018412 correct 49
Epoch  280  loss  0.14094373031928498 correct 49
Epoch  290  loss  1.095773906407053 correct 49
Epoch  300  loss  0.19999860785878065 correct 49
Epoch  310  loss  0.16104980500079744 correct 50
Epoch  320  loss  0.05998188346944402 correct 49
Epoch  330  loss  0.07329287188722461 correct 50
Epoch  340  loss  0.16906839280605915 correct 49
Epoch  350  loss  1.037448194419018 correct 49
Epoch  360  loss  1.1719503349792533 correct 50
Epoch  370  loss  0.06617582009842318 correct 50
Epoch  380  loss  0.8875721791185137 correct 49
Epoch  390  loss  0.06151631639574145 correct 49
Epoch  400  loss  0.06468910998681314 correct 49
Epoch  410  loss  0.04144270300013556 correct 50
Epoch  420  loss  0.13195023968817304 correct 49
Epoch  430  loss  0.1463525527557017 correct 49
Epoch  440  loss  1.1403345701198915 correct 49
Epoch  450  loss  0.08719855818243942 correct 50
Epoch  460  loss  0.8365523804664334 correct 49
Epoch  470  loss  0.07419720141536326 correct 49
Epoch  480  loss  1.0094449828076495 correct 49
Epoch  490  loss  1.0594685487069622 correct 49

## !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Execution time: 0.8047013162100006 seconds per epoch

Epoch  0  loss  5.66220585990756 correct 39
Epoch  10  loss  1.9445801856353837 correct 48
Epoch  20  loss  0.7656368245272299 correct 48
Epoch  30  loss  0.49932689930749663 correct 48
Epoch  40  loss  1.428676027606679 correct 48
Epoch  50  loss  1.2804779678397977 correct 48
Epoch  60  loss  1.0029768732440418 correct 48
Epoch  70  loss  1.3110966192086713 correct 49
Epoch  80  loss  0.848506420771861 correct 48
Epoch  90  loss  0.8705258705495974 correct 48
Epoch  100  loss  2.8859669331729467 correct 46
Epoch  110  loss  0.707384328765583 correct 49
Epoch  120  loss  0.044178466491135784 correct 49
Epoch  130  loss  1.1693907211313574 correct 48
Epoch  140  loss  1.222913842336754 correct 50
Epoch  150  loss  0.5502735344519482 correct 48
Epoch  160  loss  0.5103092061904246 correct 49
Epoch  170  loss  0.34567584130004836 correct 49
Epoch  180  loss  0.32813174538994666 correct 48
Epoch  190  loss  1.3670636624819463 correct 49
Epoch  200  loss  0.007496392507492306 correct 49
Epoch  210  loss  0.10502738275983844 correct 49
Epoch  220  loss  1.3653163978735536 correct 50
Epoch  230  loss  0.447595696795199 correct 50
Epoch  240  loss  0.40906895379634234 correct 49
Epoch  250  loss  0.6266076423692908 correct 48
Epoch  260  loss  0.15652917579383183 correct 50
Epoch  270  loss  0.4097629208515091 correct 49
Epoch  280  loss  0.7276187403615771 correct 49
Epoch  290  loss  0.8120409363606919 correct 49
Epoch  300  loss  2.2203463434430875 correct 49
Epoch  310  loss  0.5181097655465475 correct 48
Epoch  320  loss  0.35071576038199365 correct 49
Epoch  330  loss  0.03479027971341271 correct 49
Epoch  340  loss  1.1409790765685335 correct 50
Epoch  350  loss  2.096673140746442 correct 46
Epoch  360  loss  0.1351094958828389 correct 49
Epoch  370  loss  0.0003773956203015612 correct 50
Epoch  380  loss  1.1902339416820709 correct 50
Epoch  390  loss  0.9522925647187316 correct 48
Epoch  400  loss  0.0008432445143392846 correct 49
Epoch  410  loss  0.8076624658952364 correct 50
Epoch  420  loss  0.4094587400779073 correct 49
Epoch  430  loss  0.9174825349813267 correct 50
Epoch  440  loss  1.0484916309316767 correct 50
Epoch  450  loss  0.3738449409450156 correct 50
Epoch  460  loss  0.043768564148152174 correct 48
Epoch  470  loss  1.4124742356430051 correct 49
Epoch  480  loss  0.05132210559165108 correct 50
Epoch  490  loss  0.009154293404649235 correct 49

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Execution time: 0.79249581369 seconds per epoch

Epoch  0  loss  6.096246695043562 correct 35
Epoch  10  loss  6.731434331558648 correct 24
Epoch  20  loss  3.958361284541816 correct 37
Epoch  30  loss  5.549481081862607 correct 47
Epoch  40  loss  3.562820817664055 correct 48
Epoch  50  loss  3.031008764975347 correct 48
Epoch  60  loss  1.7468742685607888 correct 49
Epoch  70  loss  1.4785113214019874 correct 49
Epoch  80  loss  3.1222026951452135 correct 49
Epoch  90  loss  1.0947847651564289 correct 49
Epoch  100  loss  0.8296290706770652 correct 50
Epoch  110  loss  0.8337034050592373 correct 49
Epoch  120  loss  1.1702511098376787 correct 49
Epoch  130  loss  0.42893888864067825 correct 49
Epoch  140  loss  0.7825346774756781 correct 50
Epoch  150  loss  0.7442599746362768 correct 49
Epoch  160  loss  0.6954616173692377 correct 49
Epoch  170  loss  1.2217722281883125 correct 50
Epoch  180  loss  0.3828602090344991 correct 50
Epoch  190  loss  0.24376524329283475 correct 50
Epoch  200  loss  0.34560892783624625 correct 50
Epoch  210  loss  0.21243532656567574 correct 50
Epoch  220  loss  0.4018963568176308 correct 50
Epoch  230  loss  0.24628228295272384 correct 50
Epoch  240  loss  0.36577359100076257 correct 50
Epoch  250  loss  0.41127946187630504 correct 50
Epoch  260  loss  0.15029074865903652 correct 50
Epoch  270  loss  0.7884879450728498 correct 50
Epoch  280  loss  0.16416201372069636 correct 50
Epoch  290  loss  0.14909358573185313 correct 50
Epoch  300  loss  0.20651790900996897 correct 50
Epoch  310  loss  0.20623280664745267 correct 50
Epoch  320  loss  0.18381085872918243 correct 50
Epoch  330  loss  0.053579316855115265 correct 50
Epoch  340  loss  0.8058740307807477 correct 50
Epoch  350  loss  0.1641375225643702 correct 50
Epoch  360  loss  0.02912486931050478 correct 50
Epoch  370  loss  0.07001495694224134 correct 50
Epoch  380  loss  0.09892946540550224 correct 50
Epoch  390  loss  0.1409603244358746 correct 50
Epoch  400  loss  0.5317493423543882 correct 50
Epoch  410  loss  0.1545481825546182 correct 50
Epoch  420  loss  0.5313470115649092 correct 50
Epoch  430  loss  0.0901671807050033 correct 50
Epoch  440  loss  0.14569685607848265 correct 50
Epoch  450  loss  0.09153777087604398 correct 50
Epoch  460  loss  0.23857745216975004 correct 50
Epoch  470  loss  0.3374449864877952 correct 50
Epoch  480  loss  0.037991471814210954 correct 50
Epoch  490  loss  0.1425674790122402 correct 50

## !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  0  loss  11.936290332305337 correct 33
Epoch  10  loss  4.811263890565914 correct 43
Epoch  20  loss  4.596407538850281 correct 41
Epoch  30  loss  4.1970884213318245 correct 46
Epoch  40  loss  3.421232128626804 correct 47
Epoch  50  loss  3.075142229840897 correct 47
Epoch  60  loss  4.510708125882246 correct 42
Epoch  70  loss  2.2932305804004005 correct 46
Epoch  80  loss  3.8692861335487 correct 47
Epoch  90  loss  1.5002669072907304 correct 47
Epoch  100  loss  2.684153695155342 correct 47
Epoch  110  loss  1.2050818387658788 correct 49
Epoch  120  loss  1.6643745750222423 correct 48
Epoch  130  loss  2.0110278832510713 correct 48
Epoch  140  loss  2.150386493492419 correct 49
Epoch  150  loss  0.5309560353852444 correct 48
Epoch  160  loss  1.4996670245867816 correct 47
Epoch  170  loss  1.2155036569955675 correct 50
Epoch  180  loss  3.272161569512862 correct 48
Epoch  190  loss  1.201298777681166 correct 50
Epoch  200  loss  1.2403916620080786 correct 50
Epoch  210  loss  0.5018665562611422 correct 50
Epoch  220  loss  1.1165087658347073 correct 50
Epoch  230  loss  1.1112319349603863 correct 50
Epoch  240  loss  0.9049907785273675 correct 50
Epoch  250  loss  1.1986831162213891 correct 50
Epoch  260  loss  0.2773216427823427 correct 48
Epoch  270  loss  0.5560993419440172 correct 49
Epoch  280  loss  0.6420686400456341 correct 49
Epoch  290  loss  0.9492551695407033 correct 50
Epoch  300  loss  0.5802125774436099 correct 50
Epoch  310  loss  0.6396856088226834 correct 50
Epoch  320  loss  1.0250493026502216 correct 50
Epoch  330  loss  0.12159860278403695 correct 50
Epoch  340  loss  0.13624388388389166 correct 49
Epoch  350  loss  0.19492639594596411 correct 50
Epoch  360  loss  0.9658437898772982 correct 49
Epoch  370  loss  0.9578900952427208 correct 49
Epoch  380  loss  0.6144792055913281 correct 49
Epoch  390  loss  0.4727111792620923 correct 49
Epoch  400  loss  0.6330987612197554 correct 50
Epoch  410  loss  1.0069575863595865 correct 49
Epoch  420  loss  0.40371594142552825 correct 50
Epoch  430  loss  0.516135556088533 correct 49
Epoch  440  loss  0.9291627027946222 correct 50
Epoch  450  loss  0.1117431235600516 correct 49
Epoch  460  loss  0.3094036527247959 correct 50
Epoch  470  loss  0.2712980570668961 correct 50
Epoch  480  loss  0.6302078772918267 correct 50
Epoch  490  loss  1.1020161998275917 correct 49

## !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  0  loss  5.0126494230370815 correct 37
Epoch  10  loss  2.9166407065884363 correct 43
Epoch  20  loss  1.3348286220742438 correct 50
Epoch  30  loss  1.0095478519226735 correct 47
Epoch  40  loss  1.3255598834216435 correct 50
Epoch  50  loss  1.3986265559350417 correct 49
Epoch  60  loss  1.6077150512497314 correct 49
Epoch  70  loss  0.36676588807365335 correct 46
Epoch  80  loss  2.5267140610158663 correct 47
Epoch  90  loss  0.36317059179074046 correct 47
Epoch  100  loss  1.115171160877765 correct 49
Epoch  110  loss  1.65102074392709 correct 47
Epoch  120  loss  0.9369505684213869 correct 49
Epoch  130  loss  1.2769941267345117 correct 50
Epoch  140  loss  0.4720018724555447 correct 50
Epoch  150  loss  1.4422576520695891 correct 49
Epoch  160  loss  0.23571165102968547 correct 50
Epoch  170  loss  1.8073724776940399 correct 50
Epoch  180  loss  0.4551777878170803 correct 49
Epoch  190  loss  0.8651079017957704 correct 50
Epoch  200  loss  0.327106413442181 correct 49
Epoch  210  loss  1.2415138922690974 correct 50
Epoch  220  loss  0.197221318328968 correct 50
Epoch  230  loss  0.32504862723488 correct 50
Epoch  240  loss  0.22639051287102885 correct 49
Epoch  250  loss  0.1859906985029062 correct 50
Epoch  260  loss  0.7506191528920123 correct 50
Epoch  270  loss  0.0006875666478189518 correct 47
Epoch  280  loss  0.7924183747550889 correct 50
Epoch  290  loss  0.19759951414553456 correct 50
Epoch  300  loss  0.45902461347187495 correct 50
Epoch  310  loss  0.5907947242195744 correct 50
Epoch  320  loss  0.07332874980828026 correct 50
Epoch  330  loss  0.07692853410948255 correct 50
Epoch  340  loss  0.29720904006382587 correct 50
Epoch  350  loss  0.5207319528488299 correct 50
Epoch  360  loss  0.2394992620019682 correct 49
Epoch  370  loss  0.4469306820109283 correct 50
Epoch  380  loss  0.5314549216660728 correct 50
Epoch  390  loss  0.3703450317721038 correct 50
Epoch  400  loss  0.7090161032700347 correct 49
Epoch  410  loss  0.5940623565362964 correct 50
Epoch  420  loss  0.34125761697962265 correct 50
Epoch  430  loss  0.09115023025108165 correct 50
Epoch  440  loss  0.07668875350773928 correct 50
Epoch  450  loss  0.00010087363395523136 correct 50
Epoch  460  loss  0.17875593782510726 correct 50
Epoch  470  loss  0.49124325818223263 correct 50
Epoch  480  loss  0.20533286966298725 correct 50
Epoch  490  loss  0.07365456741785262 correct 50

## !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

