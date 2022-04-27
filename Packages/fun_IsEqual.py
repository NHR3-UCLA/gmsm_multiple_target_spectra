from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

def IsEqualFun(a,b,my_device):
    
    a = np.float32(a)
    b = np.float32(b)

    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
    
    #Define the kernel
    prg = cl.Program(ctx, """
    __kernel void ismatrixequal(
        __global const float *a,__global const float *b, __global float *c)
    {
          int gid = get_global_id(0);
          if (a[gid] == b[gid]) {
              c[gid] = 0;
          } else {
              c[gid] = 1;
          }
    }
    """).build()
    
    mf = cl.mem_flags                          
    a1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c1 = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    prg.ismatrixequal(queue, a.shape, None, a1, b1, c1)
    c = np.empty_like(a)
    cl.enqueue_copy(queue, c, c1)  
    
    return c