require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'paths'
local ffi = require 'ffi'
image = require 'image'
local models = require 'models/init_test'
local opts = require 'opts'
local DataLoader = require 'dataloader'
local checkpoints = require 'checkpoints'

opt = opts.parse(arg)
show = false    -- Set show to true if you want to visualize. In addition, you need to use qlua instead of th.

checkpoint, optimState = checkpoints.latest(opt)
model = models.setup(opt, checkpoint)
offset = 0
if opt.smooth then
   offset = 1
end

print(model)
local valLoader = DataLoader.create(opt)
print('data loaded')
input = torch.CudaTensor()
function copyInputs(sample)
   input = input or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   input:resize(sample.input:size()):copy(sample.input)
   return input
end

function sleep(n)
   os.execute("sleep " .. tonumber(n))
end

function process( scoremap )
   local avg = nn.Sequential()
   avg:add(nn.SpatialSoftMax())
   avg:add(nn.SplitTable(1, 3))
   avg:add(nn.NarrowTable(2, 4))
   local paral = nn.ParallelTable()
   local seq = nn.Sequential()
   seq:add(nn.Contiguous())
   seq:add(nn.View(1, 288, 960):setNumInputDims(2))
   local conv = nn.SpatialConvolution(1, 1, 9, 9, 1, 1, 4, 4)
   conv.weight:fill(1/81)
   conv.bias:fill(0)
   seq:add(conv)
   paral:add(seq)
   for i=1, 3 do
      paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
   end
   avg:add(paral)
   avg:add(nn.JoinTable(1, 3))
   avg:cuda()
   return avg:forward(scoremap)
end

model:evaluate()
T = 0
N = 0
for n, sample in valLoader:run() do
   print(n)
   input = copyInputs(sample)
   local imgpath = sample.imgpath
   local timer = torch.Timer()
   output = model:forward(input)
   local t = timer:time().real
   print('time: ' .. t)
   local scoremap = output[1]  --:double()
   if opt.smooth then
      scoremap = process(scoremap):float()
   else
      local softmax = nn.SpatialSoftMax():cuda()
      scoremap = softmax(scoremap):float()
   end
   if n > 1 then
      T = T + t
      N = N + 1
      print('avgtime: ' .. T/N)
   end
   timer:reset()
   local exist = output[2]:float()
   local outputn
   for b = 1, input:size(1) do
      print('img: ' .. ffi.string(imgpath[b]:data()))
      local img = image.load(opt.data .. ffi.string(imgpath[b]:data()), 3, 'float')
      outputn = scoremap[{b,{},{},{}}]
      --print(outputn:size())
      local _, maxMap = torch.max(outputn, 1)
      local save_path = string.sub(ffi.string(imgpath[b]:data()), 15, -16) .. '.json_'
      for cnt = 1, 5 do
         out_img = maxMap:eq(cnt) -- outputn[{cnt, {}, {}}]--maxMap:eq(cnt) * 1
         out_img = image.scale(out_img, 1276, 384, 'simple')
         if cnt == 1 then
            out_img = torch.cat(torch.ones(1, 333, 1276):byte(), out_img, 2)
         else
            out_img = torch.cat(torch.zeros(1, 333, 1276):byte(), out_img, 2)
         end
         image.save(opt.save .. save_path .. (cnt -1) .. '.png', out_img:cuda())
      end
   end
end
