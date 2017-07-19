--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--


local M = {}

local function convertToTensor(files)

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)
   print("=> Downloading Rsigns dataset from " .. '/media/D/DIZ/Signs/SignsR/R201707-val.t7a')
   val = torch.load('/media/D/DIZ/Signs/SignsR/R201707-val.t7a','ascii')
   print("=> Downloading Rsigns dataset from " .. '/media/D/DIZ/Signs/SignsR/R201707-trn.t7a')
   train = torch.load('/media/D/DIZ/Signs/SignsR/R201707-trn.t7a','ascii')
   
   print(" | saving Rsigns dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = train,
      val = val,
   })
end

return M
