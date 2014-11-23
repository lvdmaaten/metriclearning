

-- function that implements NCA gradient:
local function nca_grad(W, X, Y, Y_tab, num_dims, lambda)
  
  -- dependencies:
  local pkg = require 'metriclearning'
  
  -- process input:
  local N = X:size(1)
  local D = X:size(2)
  W:resize(D, num_dims)
  
  -- compute projected data:
  local Z = torch.mm(X, W)
  
  -- compute pairwise square Euclidean distance matrix:
  local P = pkg.mahalanobis_distance(Z)
  
  -- compute similarities:
  local eps = 1e-14
  P = -P      -- is negation allocating new memory?
  P:exp()
  for n = 1,N do
    P[n][n] = 0
  end
  P:cdiv(P:sum(2):expand(N, N))
  P:apply(function(x) if x < eps then return eps else return x end end)
  
  -- compute log-probabilities:
  local log_P = torch.log(P)
  for n = 1,N do
    log_P[n][n] = 0
  end
  
  -- compute NCA cost function:
  local C = 0
  for n = 1,N do
    C = C - log_P[n]:index(1, Y_tab[Y[n]]):sum()
  end
  C = C / N + lambda * torch.norm(W)
  
  -- allocate some memory:
  local dC = torch.zeros(W:size())
  local dX = torch.DoubleTensor(X:size())
  local dZ = torch.DoubleTensor(Z:size())
  local weights = torch.DoubleTensor(N)
  
  -- compute gradient:
  for n = 1,N do
    
    -- compute differences in data and embedding:
    torch.add(dX, X:narrow(1, n, 1):expand(X:size()), -X)     -- is negation allocating new memory?
    torch.add(dZ, Z:narrow(1, n, 1):expand(Z:size()), -Z)     -- is negation allocating new memory?
    
    -- compute "weights" for final multiplication
    local inds = Y_tab[Y[n]]
    torch.mul(weights, P[n], -(inds:nElement()) + 1)
    weights:indexCopy(1, inds, weights:index(1, inds):add(1)) -- can this be done without memcopy?
    weights[n] = weights[n] - 1
    weights:resize(N, 1)
    
    -- sum final gradient:
    dZ:cmul(torch.expand(weights, dZ:size()))
    local tmp = torch.mm(dX:t(), dZ)
    dC:addmm(dX:t(), dZ)
  end
  dC:mul(2 / N)
  dC:add(2 * lambda, W)
  
  -- return cost function and gradient:
  dC:resize(dC:nElement())
  return C, dC
end


-- function that numerically checks gradient of NCA loss:
local function checkgrad(W, X, Y, Y_tab, num_dims, lambda)
    
    -- compute true gradient
    local _,dC = nca_grad(W, X, Y, Y_tab, num_dims, lambda)
    
    -- compute numeric approximations to gradient
    local eps = 1e-7
    local dC_est = torch.DoubleTensor(dC:size())
    for i = 1,dC:size(1) do
        for j = 1,dC:size(2) do
            W[i][j] = W[i][j] + eps
            local C1 = nca_grad(W, X, Y, Y_tab, num_dims, lambda)
            W[i][j] = W[i][j] - 2 * eps
            local C2 = nca_grad(W, X, Y, Y_tab, num_dims, lambda)
            W[i][j] = W[i][j] + eps
            dC_est[i][j] = (C1 - C2) / (2 * eps)
        end
    end

    -- compute errors of final estimate
    local diff = torch.norm(dC - dC_est) / torch.norm(dC + dC_est)
    print('Error in NCA gradient: ' .. diff)
end


-- function that performs NCA:
local function nca(X, Y, opts)
  
  -- retrieve hyperparameters:
  local num_dims = opts.num_dims
  local lambda   = opts.lambda
  
  -- initialize solution:
  local W = torch.randn(X:size(2), num_dims) * 0.001
  
  -- count how often each label appears:
  local label_counts = {}
  for n = 1,Y:nElement() do
     if label_counts[Y[n]] == nil then
       label_counts[Y[n]] = 1
     else
       label_counts[Y[n]] = label_counts[Y[n]] + 1
     end
  end
  
  -- build a table with indices per label:
  local Y_tab = {}
  local num_classes = 0
  for key,val in pairs(label_counts) do
    Y_tab[key] = torch.LongTensor(label_counts[key])
    num_classes = num_classes + 1
  end
  local cur_counts = torch.ones(num_classes)
  for n = 1,Y:nElement() do
    Y_tab[Y[n]][cur_counts[Y[n]]] = n
    cur_counts[Y[n]] = cur_counts[Y[n]] + 1
  end
  
  -- perform numerical check of the gradient:
  -- checkgrad(W, X, Y, Y_tab, num_dims, lambda)
  
  -- perform minimization of NCA loss:
  local state = {lineSearch = optim.fista, maxIter = 500, maxEval = 1000, tolFun = 1e-5, tolX = 1e-5, verbose = true}
  local func = function(x)
    local C,dC = nca_grad(x, X, Y, Y_tab, num_dims, lambda)
    return C,dC
  end
  W = optim.lbfgs(func, W, state)
  
  -- return linear mapping
  return W
end

-- return NCA function:
return nca
