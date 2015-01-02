

-- function that performs ITML:
local function itml(X, Y, inp_opts)
  
  -- input parameters:
  local opts = inp_opts or {}
  local u = opts.u or 1
  local l = opts.l or 1
  local gamma = opts.gamma or 1
  
  -- initialize some variables:
  local N = X:size(1)
  local D = X:size(2)
  local M0     = torch.eye(D)
  local inv_M0 = torch.eye(D)
  local M      = torch.eye(D)
  local C, old_C = math.huge, math.huge
  local delta, eps = 0, 1e-9
  
  -- learning parameters:
  local max_iter = 1e6
  local tol = 1e-5
  
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
  
  -- initialize slack variables:
  local lambda = torch.zeros(N, N)
  local slack  = torch.zeros(N, N)
  slack[same_label] = u
  slack[torch.add(-same_label, 1)] = l
  for n = 1,N do
    slack[n][n] = 0
  end
  local slack0 = slack:clone()
  
  -- performing learning iterations until convergence:
  local iter = 0
  while iter < max_iter do --and (C == math.huge or old_C - C > tol) do
    
    -- perform Bragman projection:
    iter = iter + 1
    local i = 1 + math.floor(math.random() * N)
    local j = 1 + math.floor(math.random() * N)
    if i ~= j then
      local diff = X[i]:clone()
      diff:add(-X[j]):resize(D, 1)
      local P = torch.mm(torch.mm(diff:t(), M), diff)[1][1]
      if P < eps then P = eps end
      if same_label[i][j] then
        delta = 1
      else
        delta = -1
      end
      local alpha = math.min(lambda[i][j], (delta / 2) * ((1 / P) - (gamma / slack[i][j])))
      local beta = (delta * alpha) / (1 - (delta * alpha * P))
      slack[i][j] = (gamma * slack[i][j]) / (gamma + delta * alpha * slack[i][j]);
      lambda[i][j] = lambda[i][j] - alpha;
      M:addmm(beta, torch.mm(M, torch.mm(diff, diff:t())), M);
    end
    
    -- compute value of cost function
    if iter % 50000 == 0 then
      old_C = C
      local tmp1 = torch.mm(M, inv_M0)
      local tmp2 = torch.cdiv(slack, slack0)
      local tmp3 = tmp2:clone()
      for n = 1,N do
        tmp2[n][n] = 0          -- needed for trace computation
        tmp3[n][n] = 1          -- needed for determinant computation
      end      
      C = 0
      for d = 1,D do
        C = C + tmp1[d][d]      -- trace of M * inv(M0)
      end
      local eig_vals = torch.eig(tmp1, 'N')
      local determinant = eig_vals:select(2, 1):prod()
      C = C - math.log(determinant) - D                                        -- Stein's loss
      C = C + gamma * (tmp2:sum() - tmp3:log():sum() - slack:nElement() + N)   -- slack term
      print('After ' .. iter .. ' updates: objective is ' .. C)
    end
  end
  
  -- return final mapping:
  return M
end

-- return ITML function:
return itml
  