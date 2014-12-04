

-- function that performs LMNN:
local function lmnn(X, Y)
  
  -- dependencies:
  local pkg = require 'metriclearning'
  
  -- initialize metric
  local N = X:size(1)
  local num_dims = X:size(2)
  local M = torch.eye(num_dims, num_dims)
  
  -- set learning parameters:
  local min_iter = 50          -- minimum number of iterations
  local max_iter = 1000        -- maximum number of iterations
  local eta = .1               -- learning rate
  local mu = .5                -- weighting of pull and push terms
  local tol = 1e-3             -- tolerance for convergence
  local num_targets = 3        -- number of target neighbors
  local best_C = math.huge     -- best error obtained so far
  local best_M = M:clone()     -- best metric found so far
  
  -- make same-label mask matrix:
  local same_label = torch.ByteTensor(N, N)
  for n = 1,N do
    for m = 1,N do
      if Y[n] == Y[m] then
        same_label[n][m] = 1
      else
        same_label[n][m] = 0
      end
    end
  end
  
  -- find target neighbors:
  local targets = torch.LongTensor(N, num_targets)
  local D = pkg.mahalanobis_distance(X)
  for n = 1,N do
    D[n][n] = math.huge
  end
  D:map(same_label, function(xx, yy) if yy == 0 then return math.huge else return xx end end)
  for t = 1,num_targets do
    local _,ind = D:min(2)
    targets:select(2, t):copy(ind)
    for n = 1,N do
      D[n][ind[n][1]] = math.huge
    end
  end
  
  -- initialize gradient:
  local G = torch.zeros(num_dims, num_dims)
  for t = 1,num_targets do
    local diff_X = -X:index(1, targets:select(2, t))
    diff_X:add(X)
    G:addmm(1 - mu, diff_X:t(), diff_X)
  end
  
  -- allocate some memory for learning:
  local slack = torch.zeros(N, N, num_targets)
  local old_slack  = torch.DoubleTensor(N, N, num_targets)
  local violations = torch.ByteTensor(N, N)
  local D_targets  = torch.DoubleTensor(N)
  local rows = torch.range(1, N):long():resize(N, 1):expand(N, N)
  local cols = torch.range(1, N):long():resize(1, N):expand(N, N)
  
  -- perform learning iterations:
  local iter, C, prev_C = 0, math.huge, math.huge
  while (C - prev_C > tol or iter < min_iter) and iter < max_iter do
    
    -- compute distance under current metric:
    D = pkg.mahalanobis_distance(X, M)
    
    -- compute slack variables and sum cost function:
    prev_C = C
    C = 0
    old_slack:copy(slack)
    for t = 1,num_targets do
      
      -- get slack for current targets:
      local targets_t = targets:select(2, t)
      local   slack_t =   slack:select(3, t)
      
      -- compute slack for current targets:
      slack_t:copy(-D)
      for n = 1,N do
        D_targets[n] = D[n][targets_t[n]]
      end
      slack_t:add(D_targets:resize(N, 1):expand(N, N)):add(1)
      slack_t[same_label] = 0
      slack_t[torch.lt(slack_t, 0)] = 0
      
      -- sum cost function (distance to targets):
      C = C + (1 - mu) * D_targets:sum()
    end
    
    -- compute final cost:
    C = C + mu * slack:sum()
    
    -- maintain best solution found so far (subgradient method):
    if C < best_C then
      best_C = C
      best_M:copy(M)
    end
    
    -- update the current gradient:
    for t = 1,num_targets do
      
      -- get current targets and slack (old and new):
      local   targets_t =   targets:select(2, t)
      local     slack_t =     slack:select(3, t)
      local old_slack_t = old_slack:select(3, t)
      
      -- construct binary slack variables:
      local     slack_tb = torch.gt(slack_t, 0)
      local old_slack_tb = torch.gt(old_slack_t, 0)
      
      -- add new violations to the gradient:
      violations:map2(slack_tb, old_slack_tb, function(xx, yy, zz) if yy == 1 and zz == 0 then return 1 else return 0 end end)
      if violations:sum() > 0 then
        local diff_X1 = X:index(1, rows[violations]) -
                        X:index(1, targets_t:index(1, rows[violations]))
        local diff_X2 = X:index(1, rows[violations]) -
                        X:index(1, cols[violations])
        G:addmm( mu, diff_X1:t(), diff_X1)
        G:addmm(-mu, diff_X2:t(), diff_X2)
      end
      
      -- remove resolved violations from the gradient:
      violations:map2(slack_tb, old_slack_tb, function(xx, yy, zz) if yy == 0 and zz == 1 then return 1 else return 0 end end)
      if violations:sum() > 0 then
        local diff_X1 = X:index(1, rows[violations]) -
                        X:index(1, targets_t:index(1, rows[violations]))
        local diff_X2 = X:index(1, rows[violations]) -
                        X:index(1, cols[violations])
        G:addmm(-mu, diff_X1:t(), diff_X1)
        G:addmm( mu, diff_X2:t(), diff_X2)
      end
    end  
    
    -- perform gradient update:
    M:addcmul(-eta / N, G, torch.ones(num_dims, num_dims))
    
    -- project metric back onto the PSD cone:
    local L, V = torch.eig(M, 'V')
    local L_real = L:select(2, 1) 
    L_real[torch.lt(L_real, 0)] = 0
    local L_diag = torch.eye(num_dims)  -- should pre-allocate
    L_diag:cmul(L_real:reshape(1, num_dims):expand(num_dims, num_dims))
    M:copy(torch.mm(torch.mm(V, L_diag), V:t()))
    
    -- update learning rate:
    if prev_C > C then
      eta = eta * 1.01
    else
      eta = eta * 0.5
    end
    
    -- print out progress:
    iter = iter + 1
    if iter == 1 or iter % 10 == 0 then
      print('Iteration ' .. iter .. ': loss function is ' .. C / N .. ' and number of constraint violations is ' .. slack:gt(0):sum())
    end
  end  
  
  -- return metric:
  return best_M
end

-- return LMNN function:
return lmnn
  