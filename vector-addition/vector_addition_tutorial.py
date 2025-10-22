import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, # pointer to first input vector
               y_ptr, # pointer to second input vector
               output_ptr, # pointer to output vector
               n_elements, # size of vector
               BLOCK_SIZE: tl.constexpr, # number of elements each program (block) should process
               # note: `constexpr` so it can used as a shape value                         
               ):
    # there are multiple `programs` processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0) # we use a 1D launch grid so axis is 0
    # this program will process inputs that are offset from the initial data.
    # for instance, if you had a vector of lenght 256 and block_size of 64, the programs
    # would each acess elements [0:64, 64:128, 128:192, 192:256].
    # note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

# helper function to allocate the z tensor and enqueue the above kernel
# with appropriate grid/block sizess 
def add(x: torch.Tensor, y: torch.Tensor):
    # we need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and y.device == DEVICE
    n_elements = output.numel()
    # the SPMD launch grid denotes the number of kernel instances that run in parallel .
    # it is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # in this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # note: 
    # - each torch.tensor object is implicitly converted into a pointer to its first element.
    # - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    # - don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # we return a handle to z but, since `torch.cuda.synchronize() hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

# We can now use the above function to compute the element-wise sum of two torch.tensor objects 
# and test its correctness

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# benchmark

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot.
        x_vals=[2**i for i  in range (12, 28, 1)], # different possible values for `x_name`.
        x_log=True, # x axis is logarithmic.
        line_arg='provider', # argument name whose value corresponds to a different line in the plot.
        line_vals=['triton',  'torch'], # possible values to `line_arg`.
        line_names=['Triton', 'Torch'], # label name for the lines.
        styles=[('blue', '-'), ('green', '-')], # line styles.
        ylabel='GB/s', # label name for the y-axis.
        plot_name='vector-add-performance', # name for the plot. Used also as a file name for saving the plot.
        args={}, # values for function arguments not in `x_names` and `y_names`.
    )
)

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=False, save_path='/home/lukas/triton/vector-addition/')