require 'torch'
require 'paths'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')
local test_data_path = '../results/train.t7'
local testData = torch.load(test_data_path)
print (testData.data:size())
testData.size = function() return testData.data:size(1) end

opt = {batchSize = 10, 
       types = 'float', 
       debug = true,
       model_path = '../results/model.net',
       thd = 0.5,
       save = '../results'}

local batch_size = 10

paths.dofile('meters.lua')

local model = torch.load(opt.model_path)
print(model)

local inputs = torch.Tensor(opt.batchSize, testData.data:size(2), 
         testData.data:size(3), testData.data:size(4))
local targets = torch.Tensor(opt.batchSize, 
         testData.labels:size(2), testData.labels:size(3))

if opt.types == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

local numberTestBatches = math.floor(testData:size() / opt.batchSize)
local maskmeter  = IouMeter(opt.thd, numberTestBatches * opt.batchSize)

model:evaluate()
maskmeter:reset()
local time = sys.clock()
print('testing on test set:')
for t = 1, testData:size(), opt.batchSize do
      xlua.progress(t, testData:size())
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end 
      local idx = 1 
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = testData.data[i]
         targets[idx] = testData.labels[i]
         idx = idx + 1 
      end 
      local y = model:forward(inputs)
      maskmeter:add(y:view(targets:size()),targets)
      if (opt.debug and t == 1) then 
           print("output first batch: ")
           torch.save(paths.concat(opt.save, 'dbg_predictions.t7'),
                y:view(targets:size()))
           torch.save(paths.concat(opt.save, 'dbg_inputs.t7'),
                inputs)
           torch.save(paths.concat(opt.save, 'dbg_targets.t7'),
                targets)
      end
end
local logepoch = string.format('[test]'..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f suc@.7 %06.2f ',
      maskmeter:value('mean'), maskmeter:value('median'),
      maskmeter:value('0.5'), maskmeter:value('0.7'))
print(logepoch)
