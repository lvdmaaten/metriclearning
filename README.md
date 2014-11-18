Metric learning
===============

This package contains Torch7 implementations of metric learning algorithms.

Install
-------

Installation of the package can be performed via:

```sh
luarocks install metriclearning
```

Use
---

Below is a simple example of the usage of the package:

```lua
-- package:
m = require 'metriclearning'

-- a dataset:
X = torch.randn(100,10) -- 100 samples, 10-dim each
Y = x:narrow(2, 1, 1)
Y:apply(function(x) if x < 0 then return -1 else return 1 end) -- corresponding labels

-- metric learners:
W = m.nca(X, Y, {num_dims=5, lambda=0})  -- perform NCA using linear mapping of rank 5 and no regularization
```

Demos
-----

The following demos are currently provided:

```sh
cd demos
th demo_nca.lua
```