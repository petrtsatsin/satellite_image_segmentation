require 'torch'
require 'xlua'
require 'optim'

-- Model + Loss:
local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss

function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   module.fgradInput = nil
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local w,dE_dw = model:getParameters()

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}

local x = torch.Tensor(opt.batchSize,trainData.data:size(2), 
         trainData.data:size(3), trainData.data:size(4))
local yt = torch.Tensor(opt.batchSize, trainData.labels:size(2),
         trainData.labels:size(3))
if opt.type == 'cuda' then 
   x = x:cuda()
   yt = yt:cuda()
end

local epoch

local function train(trainData)

   model:training()
   print(trainData)
   epoch = epoch or 1
   local time = sys.clock()
   print('-------')
   print(trainData.data:size(1))
   print('-------')
   local shuffle = torch.randperm(trainData.size(1))

   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize ..
']')
   local loss_err = 0 
   for t = 1, trainData.size(),opt.batchSize do
      xlua.progress(t, trainData.size())
      collectgarbage()
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         yt[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end
      local eval_E = function(w)
         dE_dw:zero()
         local y = model:forward(x)
         local E = loss:forward(y,yt:viewAs(y))   
         local dE_dy = loss:backward(y,yt:viewAs(y))   
         dE_dy:mul(x:size(1))
         model:backward(x,dE_dy)
         loss_err = loss_err + E
         return E,dE_dw
      end
      optim.sgd(eval_E, w, optimState)
   end
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   trainLogger:add{['% Loss (train set)'] = loss_err}
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '.. filename)
   model1 = model:clone()
   netLighter(model1)
   torch.save(filename, model1)
   epoch = epoch + 1
end

return train

