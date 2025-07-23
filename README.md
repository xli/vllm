Test
====

* install vllm
* start server: make serve
* send a test request: make request

when  --enforce-eager is used, will get the following error:

    _C::rotary_embedding: attempted to run this operator with Meta tensors, but there was no fake impl or Meta kernel registered. You may have run into this message while using an operator with PT2 compilation APIs (torch.compile/torch.export); in order to use this operator with those APIs you'll need to add a fake impl. Please see the following for next steps:  https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
