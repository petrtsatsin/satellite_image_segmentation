require 'torch'
require 'torchx'
require 'image'
require 'xlua'
local gm = require 'graphicsmagick'

data_path = "../data"
image_size = 46
validRatio = 0.1
testRatio  = 0.1
data_cache = true

local function preprocessImage(img)
    local size = math.min(img:size(2), img:size(3))
    return image.scale(image.crop(img, "c", size, size), image_size,
        image_size)
end

local function imageLoad(img_path)
    return gm.Image(img_path):toTensor('float', 'RGB', 'DHW')
end

local function getData(data_path) 
    local tiles = paths.indexdir(paths.concat(data_path, "tiles"), {"tif"}) 
    local masks = paths.indexdir(paths.concat(data_path, "masks"), {"tif"})
    local size = tiles:size()
    local shuffle = torch.randperm(size) -- shuffle the data
    local input  = torch.FloatTensor(size, 3, image_size, image_size)
    local target = torch.FloatTensor(size, image_size, image_size)

    for i=1,tiles:size() do
        local img = preprocessImage(imageLoad(tiles:filename(i)))
        local idx = shuffle[i]
        input[idx]:copy(img)
        xlua.progress(i, size)
        collectgarbage()
    end

    for i=1,masks:size() do
        local img = preprocessImage(imageLoad(masks:filename(i)))
        local idx = shuffle[i]
        target[idx]:copy(img[3])
        xlua.progress(i, size)
        collectgarbage()
    end
-- train, validation, test split
    local nValid = math.floor(size * validRatio)
    local nTest  = math.floor(size * testRatio)
    local nTrain = size - nValid - nTest
    print("Train size: " .. nTrain)
    print("Validaton size: " .. nValid)
    print("Test size: " .. nTest)

    local trainInput  = input:narrow (1, 1, nTrain)
    local trainTarget = target:narrow(1, 1, nTrain)
    local validInput  = input:narrow (1, nTrain+1, nValid)
    local validTarget = target:narrow(1, nTrain+1, nValid)
    local testInput   = input:narrow (1, nTrain+nValid+1, nTest)
    local testInput   = target:narrow(1, nTrain+nValid+1, nTest)

    trainData = {data = trainInput, labels = trainTarget}
    testData = {data = testInput, labels = testTarget}
    validData = {data = validInput, labels = validTarget}
    torch.save(paths.concat(opt.save,'train.t7'), trainData)
    torch.save(paths.concat(opt.save,'test.t7'), testData)
    torch.save(paths.concat(opt.save,'valid.t7'), validData)
    return  trainData, validData
end

if data_cache then
   trainData = torch.load(paths.concat(opt.save,'train.t7'))
   validData = torch.load(paths.concat(opt.save,'valid.t7'))
else 
   trainData, validData = getData(data_path)
end

trainData.size = function() return trainData.data:size(1) end
validData.size = function() return validData.data:size(1) end

print(trainData)
print(validData)
print("Done")

return {
   trainData = trainData,
   validData = validData
}
