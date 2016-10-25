function global_normalization(trainData, testData) 
    print(sys.COLORS.red ..  '==> preprocessing data')
    local channels = {'r','g','b'}
    print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
    local mean = {}
    local std = {}
    for i,channel in ipairs(channels) do
       mean[i] = trainData.data[{ {},i,{},{} }]:mean()
       std[i] = trainData.data[{ {},i,{},{} }]:std()
       trainData.data[{ {},i,{},{} }]:add(-mean[i])
       trainData.data[{ {},i,{},{} }]:div(std[i])
    end

    for i,channel in ipairs(channels) do
   -- normalize each channel globally:
        testData.data[{ {},i,{},{} }]:add(-mean[i])
        testData.data[{ {},i,{},{} }]:div(std[i])
    end

    for i,channel in ipairs(channels) do
        local trainMean = trainData.data[{ {},i }]:mean()
        local trainStd = trainData.data[{ {},i }]:std()

        local testMean = testData.data[{ {},i }]:mean()
        local testStd = testData.data[{ {},i }]:std()

        print('training data, '..channel..'-channel, mean: ' .. trainMean)
        print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
        print('validation data, '..channel..'-channel, mean: ' ..testMean)
        print('validation data, '..channel..'-channel, standard deviation: ' .. testStd)
    end
end
return global_normalization
