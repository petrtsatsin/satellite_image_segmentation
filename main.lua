require 'pl'
require 'trepl'
require 'torch'   
require 'nn'

opt = lapp[[
   -l,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in #
samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 10)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
for testing'
]]

print(opt)

torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
end

print('==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'valid'

print('==> training!')

print(data)

while true do
   train(data.trainData)
   test(data.validData)
end
