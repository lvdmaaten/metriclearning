
-- dependencies:
require 'torch'
require 'optim'

-- function that performs nearest neighbor classification:
local function nn_classification(train_Z, train_Y, test_Z)
  
  -- compute squared Euclidean distance matrix:
  local N = train_Z:size(1)
  local M =  test_Z:size(1)
  local buff1 = torch.DoubleTensor(train_Z:size())
  local buff2 = torch.DoubleTensor( test_Z:size())
  torch.cmul(buff1, train_Z, train_Z)
  torch.cmul(buff2,  test_Z,  test_Z)
  local sum_Z1 = -buff1:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z2 = -buff2:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z1_expand = sum_Z1:t():expand(M, N)
  local sum_Z2_expand = sum_Z2:expand(M, N)
  local D = torch.mm(test_Z, train_Z:t())
  D:mul(2)
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


-- return package:
return {
   nca = require 'metriclearning.nca',
   nn_classification = nn_classification
}