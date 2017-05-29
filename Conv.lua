local Conv = torch.class('similarityMeasure.Conv')

function Conv:__init(config)
  self.learning_rate = config.learning_rate or 0.01
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.shared        = config.shared        or 1
  self.criteria      = config.criteria      or 1
  self.reg           = config.reg           or 1e-4
  self.flag          = config.flag          or 1
  self.ngram         = config.win           or 3
  self.div           = config.rdiv          or 1

	
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  self.num_classes = 5
	
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- optimization objective
  if self.criteria == 1 then
    self.criterion = nn.DistKLDivCriterion()
  else
    self.criterion = nn.MSECriterion()
  end
    
  
  ---------------------------------Combination of ConvNets.
  variants1 = {
  [1] = function (x) dofile 'models.lua' end,
  [2] = function (x) dofile 'models_rdim.lua' end,
  [3] = function (x) dofile 'models_edif.lua' end,
  [4] = function (x) dofile 'models_euc.lua' end,
  [5] = function (x) dofile 'models_cos.lua' end,
  [6] = function (x) dofile 'models_max.lua' end,
  [7] = function (x) dofile 'models_mean.lua'end,
  [8] = function (x) dofile 'models_win.lua' end,
  } 

variants1[self.flag]()
  print('<model> creating a fresh model')
  
  -- Size of vocabulary; Number of output classes
  
  self.length = self.emb_dim
  self.convModel = createModel(self.length, self.num_classes, self.ngram,self.shared,self.div)  
  self.softMaxC = self:ClassifierOOne()

  ----------------------------------------
  local modules = nn.Parallel()
   :add(self.convModel) 
   :add(self.softMaxC) 
  self.params, self.grad_params = modules:getParameters()
end

function Conv:ClassifierOOne()
  local modelQ1 = nn.Sequential()	
  local ngram = self.ngram
  local NumFilter = self.length

  variants2 = {
  [1] = function (x) inputNum = 2*2*NumFilter + 2*NumFilter*(ngram+1) end,
  [2] = function (x) inputNum = 2*2*NumFilter/self.div + 2*NumFilter*(ngram)/self.div  end,
  [3] = function (x) inputNum = 2*NumFilter*(ngram)/self.div end,
  [4] = function (x) inputNum = 2*NumFilter/self.div end,
  [5] = function (x) inputNum = 2*NumFilter/self.div end,
  [6] = function (x) inputNum = 1*2*NumFilter/self.div + 1*NumFilter*(ngram)/self.div end,
  [7] = function (x) inputNum = 1*2*NumFilter/self.div + 1*NumFilter*(ngram)/self.div end,
  [8] = function (x) inputNum = 2*2*NumFilter/self.div + 2*NumFilter*(ngram)/self.div end,
  } 
variants2[self.flag]()
 
  print(inputNum)
  modelQ1:add(nn.Linear(inputNum, self.sim_nhidden))
  modelQ1:add(nn.Tanh())	
  modelQ1:add(nn.Linear(self.sim_nhidden, self.num_classes))
  modelQ1:add(nn.LogSoftMax())	
  return modelQ1
end

function Conv:trainCombineOnly(dataset)
  train_looss = 0.0
   
  for i = 1, dataset.size  do
    local targets = torch.zeros(1, self.num_classes)
      local sim  = -0.1
      sim = dataset.labels[i]
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{1, floor}] = 1
      else
        targets[{1, floor}] = ceil - sim
        targets[{1, ceil}] = sim - floor
      end
    
    local feval = function(x)
      self.grad_params:zero()
      local loss = 0

      local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
      local linputs = self.emb_vecs:index(1, lsent:long()):double()
      local rinputs = self.emb_vecs:index(1, rsent:long()):double()
    		
   	  local part2 = self.convModel:forward({linputs, rinputs})
     -- for i,module in ipairs(self.convModel:listModules()) do
     --     print(module.modules)
    --end 
   	--print(self.convModel.modules)
   	--print(part2:size())
   	--print(part2:dim())
   	  local output = self.softMaxC:forward(part2)

      loss = self.criterion:forward(output, targets[1])
      train_looss = loss + train_looss
      local sim_grad = self.criterion:backward(output, targets[1])
      local gErrorFromClassifier = self.softMaxC:backward(part2, sim_grad)
	  self.convModel:backward({linputs, rinputs}, gErrorFromClassifier)
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
  end
  return train_looss
end

-- Predict the similarity of a sentence pair.
function Conv:predictCombination(lsent, rsent)
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()

  local part2 = self.convModel:forward({linputs, rinputs})
  local output = self.softMaxC:forward(part2)
  local val = -1.0
  val = torch.range(1, 5, 1):dot(output:exp())
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Conv:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
  end
  return predictions
end

function Conv:print_config()
  local num_params = self.params:nElement()

  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('learning rate: ' .. self.learning_rate)
  print('regularization strength: ' .. self.reg)
  print('sim module hidden dim: ' .. self.sim_nhidden)
  print('shared weights: ' .. self.shared)
  print('loss function:' .. self.criteria)
  print('Variant:' .. self.flag)
  print('Reduced Dimension Factor:' .. self.div)
  print('Window Types:' .. self.ngram)
  
end
