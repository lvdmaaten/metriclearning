

-- function that computes a pairwise squared Euclidean distance matrix:
function sq_eucl_distance(Z)
  local N = Z:size(1)
  local buff = torch.DoubleTensor(Z:size())
  torch.cmul(buff, Z, Z)
  local sum_Z = -buff:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z_expand = sum_Z:expand(N, N)
  local D = torch.mm(Z, Z:t())
  D:mul(2)
  D:add(sum_Z_expand):add(sum_Z_expand:t())
  return D
end


-- function that implements NCA gradient:
local function nca_grad(W, X, Y, Y_tab, num_dims, lambda)
  
  -- compute projected data:
  local N = X:size(1)
  local Z = torch.mm(W, X)
  
  -- compute pairwise square Euclidean distance matrix:
  local P = sq_eucl_distance(Z)
  
  -- compute similarities:
  P:exp()
  P:cdiv(P:sum(2):expand(N, N))
  
  -- compute NCA cost function:
  local C = 0
  for n = 1,N do
    C = C - P[n]:index(2, Y_tab[Y[n]]):sum() 
    C = C + P[n][n]       -- exclude self-similarities
  end
  
  -- compute gradient:
  local dC = torch.zeros(W:size())
  local dX = torch.DoubleTensor(X:size())
  local dZ = torch.DoubleTensor(Z:size())
  local weights = torch.DoubleTensor(N, 1)
  for n=1:N do
    
    -- compute differences in data and embedding:
    torch.add(dX, X:narrow(1, n, 1):expand(N, N), -X)     -- is the negation allocating new memory?
    torch.add(dZ, Z:narrow(1, n, 1):expand(N, N), -Z)
    
    -- construct "weights" for final multiplication
    torch.mul(weights, P[n], -Y_tab[Y[n]]:nElement() + 1)
    weights:indexCopy(1, Y_tab[Y[n]], weights:index(1, Y_tab[Y[n]]):add(1))
    weights[n] = weights[n] - 1
    
    -- sum final gradient:
    torch.addmm(dC, dX:t(), dZ:cmul(weights:expand(dZ:size())))
  end
  
  -- return cost function and gradient:
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


-- function that computes nearest neighbor error under metric:
local function nearest_neighbor_error(W, X, Y)
  
  -- compute projected data:
  local N = X:size(1)
  local Z = torch.mm(W, X)
  
  -- compute pairwise square Euclidean distance matrix:
  local D = sq_eucl_distance(Z)
  
  -- compute nearest neighbor error:
  local err = 0
  for n = 1,N do
    D[n][n] = math.huge
    _,ind = min(D[n])
    if Y[n] ~= Y[ind] then
      err = err + 1
  end
  err = err / N  
  
  -- return result:
  return err
end


-- function that performs NCA:
local function nca(X, Y, num_dims, lambda)
  
  -- retrieve hyperparameters:
  local num_dims = opts.num_dims
  local lambda   = opts.lambda
  
  -- initialize solution:
  local W = torch.randn(X:size(2), num_dims) * 0.0001
  
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
  local cur_counts = torch.ones(label_counts)
  for n = 1,Y:nElement() do
    Y_tab[Y[n]][cur_counts[Y[n]]] = n
    cur_counts[Y[n]] = cur_counts[Y[n]] + 1
  end
  
  -- perform numerical check of the gradient:
  checkgrad(W, X, Y, Y_tab, num_dims, lambda)
  
  -- perform minimization of NCA loss:
  local state = {learningRate = 1e-3, momentum = 0.5 }
  local func = function(x)
    local C,dC = nca_grad(x, X, Y, Y_tab, num_dims, lambda)
    return C,dC
  end
  optim.lbfgs(func, W, state)
  
  -- measure nearest neighbor error on training data under metric:
  local err = nearest_neighbor_error(W, X, Y)
  
  -- return linear mapping
  return W
end

-- return NCA function:
return nca
