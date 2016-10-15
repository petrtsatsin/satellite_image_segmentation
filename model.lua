require 'torch'
require 'image'
require 'nn'

local nfeats = 3
local width = 46
local height = 46
local nstates = {32,64,128,128}
local filtsize = {7,7,7}
local poolsize = 2

print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2],
filtsize[2]))
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
CNN:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize[3],
filtsize[3]))

local classifier = nn.Sequential()

classifier:add(nn.Reshape(nstates[3]))
classifier:add(nn.Linear(nstates[3], nstates[4]))
classifier:add(nn.Linear(nstates[4], width*height))

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end

model = nn.Sequential()
model:add(CNN)
model:add(classifier)

loss = nn.SoftMarginCriterion()

print('==> model: ')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

