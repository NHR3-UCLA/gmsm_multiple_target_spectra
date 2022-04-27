from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

def MatrixSubtraction(a,b,my_device):
    
    a = np.float32(a)
    b = np.float32(b)
    
    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
    
    c = np.zeros(a.shape).astype(np.float32)
    
    #Define the kernel
    prg = cl.Program(ctx, """
    __kernel void MatrixSubtraction(
        __global const float *a,__global const float *b, __global float *c)
    {
          int gid = get_global_id(0);
          c[gid] = a[gid] - b[gid];
    }
    """).build()
    
    mf = cl.mem_flags                          
    a1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c1 = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
    prg.MatrixSubtraction(queue, a.shape, None, a1, b1, c1);
    cl.enqueue_copy(queue, c, c1) 
    queue.finish()
        
    return c