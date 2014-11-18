metriclearning = require 'metriclearning'


-- function that performs nearest neighbor classification:
local function nn_classification(train_Z, train_Y, test_Z)
  
  -- compute squared Euclidean distance matrix:
  local N = train_Z:size(1)
  local M =  test_Z:size(1)
  local buff1 = torch.DoubleTensor(train_Z:size())
  local buff2 = torch.DoubleTensor( test_X:size())
  torch.cmul(buff1, train_Z, train_Z)
  torch.cmul(buff2,  test_X,  test_Z)
  local sum_Z1 = -buff1:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z2 = -buff2:sum(2)               -- right direction to sum? or is sum(1) faster?
  local sum_Z1_expand = sum_Z1:t():expand(M, N)
  local sum_Z2_expand = sum_Z2:expand(M, N)
  local D = torch.mm(test_Z, train_Z:t())
  D:mul(2)
  D:add(sum_Z1_expand):add(sum_Z2_expand)
  
  -- perform 1-nearest neighbor classification:
  test_Y = torch.LongTensor(M)
  for m = 1,M do
    _,ind = min(D[m])
    test_Y[m] = train_Y[ind]
  end
  
  -- return classification
  return test_Y
end

-- function that performs demo of metric learning code on MNIST:
local function demo_nca()

  -- amount of data to use for test:
  local N = 5000

  -- load subset of MNIST test data:
  local mnist = require 'mnist'
  local trainset = mnist.traindataset()
  local testset = mnist.testdataset()
  trainset.data  = trainset.data:narrow(1, 1, N)
  trainset.label = trainset.label:narrow(1, 1, N)
  testset.data   =  testset.data:narrow(1, 1, N)
  testset.label  =  testset.label:narrow(1, 1, N)
  local train_X = torch.Tensor(trainset.data:size())
  local  test_X = torch.Tensor( testset.data:size())
  train_X:map(trainset.data, function(xx, yy) return yy end)
   test_X:map( testset.data, function(xx, yy) return yy end)
  train_X:resize(train_X:size(1), train_X:size(2) * train_X:size(3))
   test_X:resize( test_X:size(1),  test_X:size(2) *  test_X:size(3))
  train_Y = trainset.label
   test_Y =  testset.label

  -- run NCA:
  opts = {num_dims = 30, lambda = 0}
  local timer = torch.Timer()
  local W = metriclearning.nca(train_X, train_Y, opts)
  print('Successfully performed NCA in ' .. timer:time().real .. ' seconds.')
  
  -- perform classification of test data:
  local train_Z = torch.mm(W, train_X)
  local  test_Z = torch.mm(W,  test_X)
  local pred_Y = nn_classification(train_Z, train_Y, test_Z)
  
  -- compute classification error
  local err = 0
  for n = 1,predY.nElement() do
    if pred_Y[n] ~= test_Y[n] then
      err = err + 1
    end
  end
  err = err / predY.nElement()
  print('Nearest-neighbor error after NCA: ' .. err)
end

-- run the demo:
demo_nca()


