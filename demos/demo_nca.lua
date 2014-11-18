metriclearning = require 'metriclearning'


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


