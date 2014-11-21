
-- dependencies:
require 'torch'
require 'optim'

-- function that computes a pairwise squared Euclidean distance matrix:
local function sq_eucl_distance(Z)
  local N = Z:size(1)
  local buff = torch.DoubleTensor(Z:size())
  torch.cmul(buff, Z, Z)
  local sum_Z = buff:sum(2)
  local sum_Z_expand = sum_Z:expand(N, N)
  local D = torch.mm(Z, Z:t())
  D:mul(-2)
  D:add(sum_Z_expand):add(sum_Z_expand:t())
  return D
end

-- function that performs nearest neighbor classification:
local function nn_classification(train_Z, train_Y, test_Z)
  
  -- compute squared Euclidean distance matrix between train and test data:
  local N = train_Z:size(1)
  local M =  test_Z:size(1)
  local buff1 = torch.DoubleTensor(train_Z:size())
  local buff2 = torch.DoubleTensor( test_Z:size())
  torch.cmul(buff1, train_Z, train_Z)
  torch.cmul(buff2,  test_Z,  test_Z)
  local sum_Z1 = buff1:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z2 = buff2:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z1_expand = sum_Z1:t():expand(M, N)
  local sum_Z2_expand = sum_Z2:expand(M, N)
  local D = torch.mm(test_Z, train_Z:t())
  D:mul(-2)
  D:add(sum_Z1_expand):add(sum_Z2_expand)
  
  -- perform 1-nearest neighbor classification:
  test_Y = torch.DoubleTensor(M)
  for m = 1,M do
    _,ind = torch.min(D[m], 1)
    test_Y[m] = train_Y[ind[1]]
  end
  
  -- return classification
  return test_Y
end


-- function that computes training nearest neighbor error:
local function train_nn_error(X, Y)
  
  -- compute projected data:
  local N = X:size(1)
  
  -- compute pairwise square Euclidean distance matrix:
  local D = sq_eucl_distance(X)
  
  -- compute nearest neighbor error:
  local err = 0
  for n = 1,N do
    _,ind = torch.min(D[n], 1)
    if Y[n] ~= Y[ind] then
      err = err + 1
    end
  end
  err = err / N  
  
  -- return result:
  return err
end


-- return package:
return {
   nca = require 'metriclearning.nca',
   sq_eucl_distance = sq_eucl_distance,
   nn_classification = nn_classification,
   train_nn_error = train_nn_error
}