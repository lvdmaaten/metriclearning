
-- dependencies:
require 'torch'
require 'optim'

-- function that computes a Mahalanobis distance matrix:
local function mahalanobis_distance(X, metric)
  
  -- default to squared Euclidean metric:
  local N = X:size(1)
  local M = metric or torch.eye(X:size(2), X:size(2))
  
  -- compute Mahalanobis distance:
  local XM = torch.mm(X, M)
  local buff = torch.DoubleTensor(X:size())
  torch.cmul(buff, XM, X)
  local sum_X = buff:sum(2)
  local D = torch.mm(X, X:t())
  D:mul(-2)
  D:add(sum_X:expand(N, N)):add(sum_X:expand(N, N):t())
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
  local test_Y = torch.DoubleTensor(M)
  for m = 1,M do
    local _,ind = torch.min(D[m], 1)
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
  local D = mahalanobis_distance(X)
  for n = 1,N do
    D[n][n] = math.huge
  end
  
  -- compute nearest neighbor error:
  local err = 0
  for n = 1,N do
    local _,ind = torch.min(D[n], 1)
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
   --lmnn = require 'metriclearning.lmnn',
   mahalanobis_distance = mahalanobis_distance,
   nn_classification = nn_classification,
   train_nn_error = train_nn_error
}
