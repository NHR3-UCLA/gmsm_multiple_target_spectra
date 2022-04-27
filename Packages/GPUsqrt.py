from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

def GPUsqrt(InputMatrix,my_device):
    
    InputMatrix = np.float32(InputMatrix)

    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
            
    #Define the kernel
    prg = cl.Program(ctx, """
    __kernel void GPUsqrt(
        __global const float *a, __global float *c)
    {
          int gid = get_global_id(0);
          c[gid] = sqrt(a[gid]);
          
    }
    """).build()
    
    mf = cl.mem_flags                          
    InputMatrix1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=InputMatrix)
    OutputMatrix1 = cl.Buffer(ctx, mf.WRITE_ONLY, InputMatrix.nbytes)
    prg.GPUsqrt(queue, InputMatrix.shape, None, InputMatrix1, OutputMatrix1)
    OutputMatrix = np.empty_like(InputMatrix)
    cl.enqueue_copy(queue, OutputMatrix, OutputMatrix1)  
    queue.finish()
    
    return OutputMatrix