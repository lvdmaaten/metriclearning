metriclearning = require 'metriclearning'
require 'unsup'


-- function that performs demo of metric learning code on MNIST:
local function demo_nca()

  -- amount of data to use for test:
  local N = 1000

  -- load subset of MNIST test data:
  local mnist = require 'mnist'
  local trainset = mnist.traindataset()
  local testset = mnist.testdataset()
  trainset.data  = trainset.data:narrow(1, 1, N)
  trainset.label = trainset.label:narrow(1, 1, N)
  testset.data   =  testset.data:narrow(1, 1, N)
  testset.label  =  testset.label:narrow(1, 1, N)
  local train_X = torch.DoubleTensor(trainset.data:size())
  local  test_X = torch.DoubleTensor( testset.data:size())
  train_X:map(trainset.data, function(xx, yy) return yy end)
   test_X:map( testset.data, function(xx, yy) return yy end)
  train_X:resize(train_X:size(1), train_X:size(2) * train_X:size(3))
   test_X:resize( test_X:size(1),  test_X:size(2) *  test_X:size(3))
  local train_Y = trainset.label + 1
  local  test_Y =  testset.label + 1
  
  -- perform PCA:
  local pca_dims = 50
  local mean = -torch.mean(train_X, 1)
  train_X:add(mean:expand(train_X:size()))
   test_X:add(mean:expand( test_X:size()))
  _, V = unsup.pca(train_X)  
  V = V:narrow(2, 1, pca_dims)
  train_X = torch.mm(train_X, V)
   test_X = torch.mm( test_X, V)
   
  -- compute classification error before NCA:
  local pred_Y = metriclearning.nn_classification(train_X, train_Y, test_X)
  local err = 0
  for n = 1,pred_Y:nElement() do
    if pred_Y[n] ~= test_Y[n] then
      err = err + 1
    end
  end
  err = err / pred_Y:nElement()
  print('Test nearest neighbor error before NCA: ' .. err)

  -- run NCA:
  opts = {num_dims = 50, lambda = 0}
  local timer = torch.Timer()
  local W = metriclearning.nca(train_X, train_Y, opts)
  print('Performed NCA in ' .. timer:time().real .. ' seconds.')
  
  -- perform NCA mapping:
  local train_Z = torch.mm(train_X, W)
  local  test_Z = torch.mm( test_X, W)
  
  -- compute classification error after NCA:
  local pred_Y = metriclearning.nn_classification(train_Z, train_Y, test_Z)
  local err = 0
  for n = 1,pred_Y:nElement() do
    if pred_Y[n] ~= test_Y[n] then
      err = err + 1
    end
  end
  err = err / pred_Y:nElement()
  print('Test nearest neighbor error after NCA: ' .. err)
end

-- run the demo:
demo_nca()


