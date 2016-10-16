require 'torch'
require 'xlua'
require 'optim'

paths.dofile('meters.lua')

local t = require 'model'
local model = t.model
local loss = t.loss

local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local inputs = torch.Tensor(opt.batchSize, validData.data:size(2), 
         validData.data:size(3), validData.data:size(4))
local targets = torch.Tensor(opt.batchSize, 
         validData.labels:size(2), validData.labels:size(3))
if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end
local numberTestBatches = math.floor(validData:size() / opt.batchSize)
local maskmeter  = IouMeter(0.5, numberTestBatches * opt.batchSize)
local epoch = 0

function test(testData)
   model:evaluate()
   maskmeter:reset()
   local time = sys.clock()
   print('testing on validation set:')
   local loss_err = 0
   epoch = epoch + 1
   for t = 1, validData:size(), opt.batchSize do
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
      local E = loss:forward(y, targets:viewAs(y))
     -- local outputs = self.maskNet:forward(self.inputs)
      maskmeter:add(y:view(targets:size()),targets)
      loss_err = loss_err + E
   end
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   testLogger:add{['% loss (test set)'] = loss_err}
   -- write log
   local logepoch = string.format('[test]  | epoch %05d '..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f suc@.7 %06.2f ',
      epoch,
      maskmeter:value('mean'), maskmeter:value('median'),
      maskmeter:value('0.5'), maskmeter:value('0.7'))
    print(logepoch)  
end

return test
