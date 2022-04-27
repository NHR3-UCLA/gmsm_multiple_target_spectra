from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

def TripleMatrixComp(a,b,d,ReplValue,my_device):
        
    a = np.float32(a)
    b = np.float32(b)
    d = np.float32(d)
    
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(ctx) 
    
    #Define the kernel
    prg = cl.Program(ctx, """
    __kernel void TripleMatrixComp(
        __global const float *a,
        __global const float *b,
        __global const float *d,
        const float ReplValue,
        __global float *c)
    {
          int gid = get_global_id(0);
          if (b[gid] >= d[gid]) {
              c[gid] += ReplValue;
          } else {
              c[gid] = a[gid];
          }
    }
    """).build()
    
    
    mf = cl.mem_flags                          
    a1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    d1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
    c1 = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    prg.TripleMatrixComp(queue, a.shape, None, a1, b1,d1,np.float32(ReplValue), c1)
    c = np.empty_like(a)
    cl.enqueue_copy(queue, c, c1)  
    queue.finish()
    
    return c