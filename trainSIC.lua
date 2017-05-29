------ Adapted from Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks
------ Hua He, Kevin Gimpel, and Jimmy Lin
------ Department of Computer Science, University of Maryland, College Park
------ Toyota Technological Institute at Chicago
------ David R. Cheriton School of Computer Science, University of Waterloo

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('gnuplot')

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv.lua')
include('CsDis.lua')

printf = utils.printf

--global paths
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

--Mean Square Error
function mse(x, y)
  z = x - y
  z = z:pow(2)
  return z:mean()
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -e,--edim (default 50)        Nr Dimensions in the Embed Vector
  -d,--dim  (default 50)        Nr Neurons in the Hidden Layer
  -s,--shared (default 1)       Shared Weights in the Convolution Layer 
  -l,--learn (default 0.01)     Learning Rate
  -n, --epoch (default 25)      Nr Epoch
  -c, --cost (default 1)        0-MSE , 1-KL Div
  -f, --flag (default 1)        Variants
  -w, --win (default 3)         n-gram windows
  -r, --rdiv (default 1)        reduced dimensions
]]


--torch.seed()
torch.manualSeed(-3.0753778015266e+18)
print('<torch> using the Manual seed: ' .. torch.initialSeed())

-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.' .. args.edim .. 'd.th')

local emb_dim = emb_vecs:size(2)

-- Discard vectors not in Vocab
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
local taskD = 'sic'
-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)
printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = similarityMeasure.Conv{
  emb_vecs   = vecs,
  sim_nhidden = args.dim,
  learning_rate = args.learn,
  shared = args.shared , 
  criteria = args.cost,
  flag = args.flag,
  win = args.win,
  rdiv = args.rdiv
}

-- number of epochs to train
local num_epochs = args.epoch

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score_p = -1.0
local best_dev_score_mse = 100.0
--local best_dev_model = model

header('Training model')
print('Epoch\tLossF\tDevp\tTestp\tDevMSE\tTestMSE')

for i = 1, num_epochs do
  local start = sys.clock()

  local train_loss = model:trainCombineOnly(train_dataset)
  
  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score_p = pearson(dev_predictions, dev_dataset.labels)
  local dev_score_mse = mse(dev_predictions, dev_dataset.labels)

  if (dev_score_p >= best_dev_score_p) then
    best_dev_score_p = dev_score_p
    best_dev_score_mse = dev_score_mse
    local test_predictions = model:predict_dataset(test_dataset)
    local test_sco_p = pearson(test_predictions, test_dataset.labels)
    local test_sco_mse = mse(test_predictions, test_dataset.labels)
    printf('%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\n', i,train_loss,dev_score_p,test_sco_p,dev_score_mse,test_sco_mse)
    local predictions_save_path = string.format(
	similarityMeasure.predictions_dir .. '/results-.edim%d.dim%d.shared%d.cost%d.var%d.win%d.rdiv%d.epoch-%d.%.5f.pred', args.edim, args.dim, args.shared,args.cost,args.flag,args.win,args.rdiv,i, test_sco_p)
--    local predictions_file = torch.DiskFile(predictions_save_path, 'w')
--    for i = 1, test_predictions:size(1) do
--      local temp = torch.FloatStorage({test_dataset.labels[i],test_predictions[i]})
--      predictions_file:writeFloat(temp)
--    end
--    predictions_file:close()
    gnuplot.pngfigure(predictions_save_path ..'.png')
    gnuplot.axis{1,5,1,5}
    gnuplot.plot(
        {'',  test_dataset.labels,  test_predictions,  '.'})
    gnuplot.xlabel('Dataset Labels')
    gnuplot.ylabel('Predicted Similarity')
    gnuplot.plotflush()
  else 
    printf('%d\t%d\t%.5f\t-\t%.5f\t-\n', i,train_loss,dev_score_p, dev_score_mse)
  end
end
print('finished training in ' .. (sys.clock() - train_start))

