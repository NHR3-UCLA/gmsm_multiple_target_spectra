from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

def MatrixElementRep_Scalar(a,BenchmarkValue,ReplValue,my_device):
    
    a = np.float32(a);
   
    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
    
    #Define the kernel
    prg = cl.Program(ctx, """
    __kernel void MatrixElementRepScalar(
        __global const float *a,
        const float BenchmarkValue,
        const float ReplValue,
        __global float *c)
    {
          int gid = get_global_id(0);
          if (a[gid] >= BenchmarkValue) {
              c[gid] = ReplValue;
          } else {
              c[gid] = a[gid];
          }
    }
    """).build()
    
    
    mf = cl.mem_flags                          
    a1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c1 = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    prg.MatrixElementRepScalar(queue, a.shape, None, a1,np.float32(BenchmarkValue),np.float32(ReplValue), c1)
    c = np.empty_like(a)
    cl.enqueue_copy(queue, c, c1)  
    
    return c