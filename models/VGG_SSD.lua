require 'nn'
require 'nngraph'
require 'detectutil'
require 'cunn'
require 'cudnn'

cudnn.benchmark = true
cudnn.fastest = true
nnlib = cudnn


function buildModel(nClasses, inputRes)
	local nClasses = nClasses or 21
	local img_inp = nn.Identity()()

	local vgg = nn.Sequential()
	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane)
	  local block = nn.Sequential()
	  block:add(nnlib.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
	  block:add(nnlib.SpatialBatchNormalization(nOutputPlane,1e-3))
	  block:add(nnlib.ReLU(true))
	  return block
	end
	local function PredictLoc(nInputPlane, nAnchors)
		local block = nn.Sequential()
		block:add(nnlib.SpatialConvolution(nInputPlane, nAnchors * 4, 3,3,1,1,1,1))
		block:add(nn.Transpose({2,4},{2,3}))
		block:add(nn.View(-1):setNumInputDims(3))
		return block

	end
	-- nClasses , include the background
	local function PredictConf(nInputPlane, nAnchors)
		local block = nn.Sequential()
		block:add(nnlib.SpatialConvolution(nInputPlane, nAnchors * nClasses, 3,3,1,1,1,1))
		block:add(nn.Transpose({2,4},{2,3}))
		block:add(nn.View(-1):setNumInputDims(3))
		return block
	end

	local function ConvReLU(nInputPlane, nOutputPlane, kernel_size, stride, padding)
	  kernel_size = kernel_size or 3
	  stride = stride or 1
	  padding = padding or 1
	  local block = nn.Sequential()
	  block:add(nnlib.SpatialConvolution(nInputPlane, nOutputPlane, kernel_size,kernel_size, stride,stride, padding,padding))
	  block:add(nnlib.ReLU(true))
	  return block
	end
	local function BuildPriorLayers(mbox_layers, data_inp)
		
		local min_scale = 20 -- default box scale
		local max_scale = 95 -- defaut box scale
		local min_dim = inputRes
		local step = math.floor((max_scale - min_scale) / (#mbox_layers -2) )
		local min_sizes = {min_dim * 10 / 100.0}
		local max_sizes = {nil}

		for k = 1,  math.floor((max_scale - min_scale)/step + 1) do
			min_sizes[k+1] = min_dim * ((min_scale + step *(k-1)) / 100.0) 
			max_sizes[k+1] = min_dim * ((min_scale + step * k) / 100.0) 
		end
		aspect_ratios = {{2}, {2,3}, {2,3}, {2,3}, {2,3},{2,3}}
		local priorbox_layers = {}
		for i, layer in ipairs(mbox_layers) do
			priorbox_layers[i] = nn.PriorBox(min_sizes[i], max_sizes[i], aspect_ratios[i], true, true){layer,data_inp}
		end
		return priorbox_layers
	end
	-- Will use "ceil" MaxPooling because we want to save as much
	-- space as we can
	local MaxPooling = nnlib.SpatialMaxPooling

	vgg:add(ConvBNReLU(3,64))
	vgg:add(ConvBNReLU(64,64))
	vgg:add(MaxPooling(2,2,2,2):ceil())
	vgg:add(ConvBNReLU(64,128))
	vgg:add(ConvBNReLU(128,128))
	vgg:add(MaxPooling(2,2,2,2):ceil())

	vgg:add(ConvBNReLU(128,256))
	vgg:add(ConvBNReLU(256,256))
	vgg:add(ConvBNReLU(256,256))
	vgg:add(MaxPooling(2,2,2,2):ceil())

	vgg:add(ConvBNReLU(256,512))
	vgg:add(ConvBNReLU(512,512))
	vgg:add(ConvBNReLU(512,512))
	-- conv4_3 end
	conv4_3 = vgg(img_inp)
	conv4_3_conf = PredictConf(512, 3)(conv4_3)
	conv4_3_loc =  PredictLoc(512, 3)(conv4_3)

	local fc7 = nn.Sequential()
	fc7:add(MaxPooling(2,2,2,2):ceil())
	fc7:add(ConvBNReLU(512,512))
	fc7:add(ConvBNReLU(512,512))
	fc7:add(ConvBNReLU(512,512))
	fc7:add(MaxPooling(3,3,1,1,1,1):ceil())
	fc7:add(nn.SpatialDilatedConvolution(512,1024,3,3,1,1,6,6,6,6))
	fc7:add(nnlib.ReLU(true))
	fc7:add(ConvReLU(1024,1024,1,1,0))
	fc7 = fc7(conv4_3)
	fc7_conf = PredictConf(1024, 6)(fc7)
	fc7_loc =  PredictLoc(1024, 6)(fc7)

	local conv6_2 = nn.Sequential()
	conv6_2:add(ConvReLU(1024,256,1,1,0))
	conv6_2:add(ConvReLU(256,512,3,2,1))
	conv6_2 = conv6_2(fc7)
	conv6_2_conf = PredictConf(512, 6)(conv6_2)
	conv6_2_loc =  PredictLoc(512, 6)(conv6_2)

	local conv7_2 = nn.Sequential()
	conv7_2:add(ConvReLU(512,128,1,1,0))
	conv7_2:add(ConvReLU(128,256,3,2,1))
	conv7_2 = conv7_2(conv6_2)
	conv7_2_conf = PredictConf(256, 6)(conv7_2)
	conv7_2_loc =  PredictLoc(256, 6)(conv7_2)

	local conv8_2 = nn.Sequential()
	conv8_2:add(ConvReLU(256,128,1,1,0))
	conv8_2:add(ConvReLU(128,256,3,2,1))
	conv8_2 = conv8_2(conv7_2)
	conv8_2_conf = PredictConf(256, 6)(conv8_2)
	conv8_2_loc =  PredictLoc(256, 6)(conv8_2)

	local pool6 = nn.Sequential()
	pool6:add(nn.SpatialAveragePooling(3,3,1,1):ceil())
	pool6 = pool6(conv8_2)
	pool6_conf = PredictConf(256, 6)(pool6)
	pool6_loc =  PredictLoc(256, 6)(pool6)

	local confs = {conv4_3_conf,fc7_conf, conv6_2_conf,conv7_2_conf, conv8_2_conf, pool6_conf}
	local locs = {conv4_3_loc,fc7_loc, conv6_2_loc,conv7_2_loc, conv8_2_loc, pool6_loc}
	local mbox_conf = nn.JoinTable(2,2)(confs)
	local mbox_loc = nn.JoinTable(2,2)(locs)

	mbox_conf = nn.View(-1,nClasses):setNumInputDims(1)(mbox_conf)  -- mbox_conf_reshape
	mbox_conf = nn.Transpose({2,3})(mbox_conf)       -- convert to n,c,h
	mbox_softmax = cudnn.SpatialLogSoftMax()(mbox_conf)
	mbox_softmax = nn.Transpose({2,3})(mbox_softmax) -- convert to n,h,c
	mbox_conf_flatten = nn.View(-1):setNumInputDims(2)(mbox_softmax) --flatten

	-- get prior box
	local mbox_layers = {conv4_3,fc7, conv6_2,conv7_2, conv8_2, pool6}

	local prior_layers = BuildPriorLayers(mbox_layers, img_inp)

	local mbox_prior = nn.JoinTable(3,3)(prior_layers)
	
	return nn.gModule({img_inp},{mbox_loc, mbox_conf_flatten, mbox_prior})
end

local model = buildModel(21,300):cuda()
img_inp = torch.Tensor(3,3,300,300):cuda()
print(img_inp:size())
out = model:forward(img_inp)
graph.dot(model.fg, 'VGG_SSD','./model')
print(out[1]:size())
print(out[2]:size())
print(out[3]:size())


