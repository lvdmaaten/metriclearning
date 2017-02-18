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
X = torch.randn(100, 10) -- 100 samples, 10-dim each
Y = X:select(2, 1):gt(0):long()
Y[Y:eq(0)] = -1  -- corresponding labels

-- learn Mahalanobis metric using LMNN:
M = m.lmnn(X, Y)
```

Demos
-----

The following demos are currently provided:

(The demos require the ```unsup``` package: ```luarocks install unsup```)

```sh
cd demos
th demo_nca.lua
th demo_lmnn.lua
th demo_itml.lua
```
