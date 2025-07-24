## Test

* install vllm
* start server: make serve
* send a test request: make request

when  --enforce-eager is used, will get the following error:

    _C::rotary_embedding: attempted to run this operator with Meta tensors, but there was no fake impl or Meta kernel registered. You may have run into this message while using an operator with PT2 compilation APIs (torch.compile/torch.export); in order to use this operator with those APIs you'll need to add a fake impl. Please see the following for next steps:  https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

## Benchmark

### Baseline (swizzle tensor not enabled):

* make serve
* make benchmark

| qps  | mean TTFT (ms) | Median TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | mean ITL (ms) | Median ITL (ms) | p99 ITL (ms) |
| ---- | -------------- | ---------------- | ------------- | -------------- | ---------------- | ------------- | ------------- | --------------- | ------------ |
| 1.49 | 2403.47        | 743.52           | 17902.17      | 40.31          | 41.10            | 41.79         | 41.00         | 29.04           | 191.56       |
| 1.49 | 2570.94        | 742.21           | 17642.33      | 40.16          | 41.02            | 41.92         | 40.48         | 29.06           | 189.20       |
| 1.49 | 2866.32        | 744.35           | 17957.88      | 39.83          | 40.90            | 41.76         | 40.10         | 28.98           | 190.63       |

### Swizzle tensor enabled:

* VLLM_SWIZZLE_TENSOR=1 make serve
* make benchmark

| qps  | mean TTFT (ms) | Median TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | mean ITL (ms) | Median ITL (ms) | p99 ITL (ms) |
| ---- | -------------- | ---------------- | ------------- | -------------- | ---------------- | ------------- | ------------- | --------------- | ------------ |
| 1.56 | 3227.89        | 754.47           | 11802.83      | 37.20          | 40.89            | 41.93         | 40.62         | 28.70           | 192.76       |
| 1.50 | 2625.89        | 744.05           | 17578.28      | 39.79          | 40.96            | 41.64         | 40.13         | 28.73           | 191.51       |
| 1.50 | 2545.97        | 748.51           | 18200.78      | 39.84          | 40.59            | 41.32         | 40.21         | 28.67           | 191.50       |
| 1.50 | 2865.06        | 746.89           | 18510.94      | 39.59          | 40.52            | 41.24         | 40.08         | 28.69           | 191.78       |
