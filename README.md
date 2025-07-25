## Test

* install vllm
* start server: make serve
  * serve LLama4 with --enforce-eager and swizzle tensor enabled
* send a test request: make request

Error:

    File "/data/users/ilx/gitrepos/vllm/vllm/attention/layer.py", line 254, in forward
      query = query.view(-1, self.num_heads, self.head_size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/ilx/uv_env/vllm/lib/python3.12/site-packages/torchao/swizzle/swizzle_tensor.py", line 147, in __torch_dispatch__
      return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/ilx/uv_env/vllm/lib/python3.12/site-packages/torch/utils/_pytree.py", line 1145, in tree_map
      return treespec.unflatten(map(func, *flat_args))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/ilx/uv_env/vllm/lib/python3.12/site-packages/torch/utils/_pytree.py", line 982, in unflatten
      leaves = list(leaves)
               ^^^^^^^^^^^^
    File "/home/ilx/uv_env/vllm/lib/python3.12/site-packages/torchao/swizzle/swizzle_tensor.py", line 145, in wrap
      return SwizzleTensor(e) if isinstance(e, torch.Tensor) else e
    File "/home/ilx/uv_env/vllm/lib/python3.12/site-packages/torchao/swizzle/swizzle_tensor.py", line 39, in __init__
      assert original.ndim == 2, "SwizzleTensor only supports ndim 2"

## Benchmark

* make serve
* make benchmark

serve without swizzle tensor:

* VLLM_SWIZZLE_TENSOR=1 make serve
