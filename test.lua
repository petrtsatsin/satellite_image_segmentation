require 'torch'
require 'xlua'
require 'optim'

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
function test(testData)
   model:evaluate()
   local time = sys.clock()
   print('testing on validation set:')
   local loss_err = 0
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
      loss_err = loss_err + E
   end
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   testLogger:add{['% loss (test set)'] = loss_err}
end

return test
