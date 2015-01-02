
-- dependencies:
require 'nn'

-- function that trains DrLim: 
local function drlim(X, Y)
  
  -- process inputs:
  local D = X:size(2)
  
  -- generate data set of similarly and dissimilarly labeled examples:
  local dataset = {X, Y}
  
  -- specify network architecture:
  local num_hiddens = {1024, 1024, 512, 256}
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(D, num_hiddens[1]))
  mlp:add(nn.Tanh())
  for n = 2,#num_hiddens do
    mlp:add(nn.Linear(num_hiddens[n - 1], num_hiddens[n])) 
    mlp:add(nn.Tanh())
  end
  mlp:add(nn.Linear(num_hiddens[n], 1))
  
  -- set criterion and train model:
  local criterion = nn.ClassNLLCriterion()
  local trainer = nn.StochasticGradient(mlp, criterion)
  trainer:train(dataset)
  
  -- return trained network:
  return mlp
end


-- return function:
return drlim