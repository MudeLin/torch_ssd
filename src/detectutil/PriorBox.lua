require 'nn'
require 'math'

--[[
Generate the prior boxes of designated sizes and aspect ratios across
  all dimensions @f$ (H \times W) @f$.
  Intended for use with MultiBox detection method to generate prior (template).

]]--

local PriorBox, Parent = torch.class('nn.PriorBox', 'nn.Module')

function PriorBox:__init(min_size, max_size, aspect_ratios, flip, clip, variances)
	Parent.__init(self)
	assert(min_size ~= nil, 'min_size must be set')
	self.min_size = min_size
	self.max_size = max_size
	assert(type(aspect_ratios) == 'number' or type(aspect_ratios) == 'table', 
			'param spect_ratios must be table or number')
	if type(aspect_ratios) == 'number' then 
		self.aspect_ratios = {aspect_ratios}
	else 
		self.aspect_ratios = aspect_ratios
	end
	self.flip = flip 
	self.clip = clip
	if self.flip then
		local num_ratios = #self.aspect_ratios

		for i = 1, num_ratios do 
			self.aspect_ratios[#self.aspect_ratios+1] = 1.0 / self.aspect_ratios[i]
		end
	end
	if variances then
		assert(type(variances) == 'table' and #variances == 4)
	end
	self.variances = variances or {0.1,0.1,0.2,0.2}

end
-- input[1] is feature maps
-- input[2] is origin img data
-- output is a tensor in shape of 1*2*(hxwxKx4), 1 for all inp image, 2 means prior box and variance, 4 means x_min, y_min, x_max, y_max,
--																		K anchors
function PriorBox:updateOutput(input)
	assert(#input == 2, 'input node must be two')
	local layer_height = input[1]:size(3)
	local layer_width = input[1]:size(4)
	local img_height = input[2]:size(3)
	local img_width = input[2]:size(4)
	local step_x = img_width / layer_width
	local step_y = img_height / layer_height
	local num_ratios = #self.aspect_ratios + 1
	if self.max_size then
		num_ratios = num_ratios + 1
	end
	self.output:resize(2, layer_height, layer_width, num_ratios * 4)
	for h = 1, layer_height do 
		for w= 1, layer_width do 
			local center_x = (w - 0.5) * step_x
			local center_y = (h - 0.5) * step_y
			local box_width = self.min_size
			local box_height = box_width
			-- first prior: aspect_ratio = 1, size = min_size
			-- xmin
			self.output[{1,h,w,0 * 4 + 1}] = (center_x - box_width / 2.0) / img_width
			-- ymin 
			self.output[{1,h,w,0 * 4 + 2}] = (center_y - box_height / 2.0) / img_height
			-- xmax
			self.output[{1,h,w,0 * 4 + 3}] = (center_x + box_width / 2.0) / img_width
			-- ymax
			self.output[{1,h,w,0 * 4 + 4}] = (center_y + box_height / 2.0) / img_height
			local ratio_offset = 0
			if self.max_size then 
				local box_width = math.sqrt(self.min_size * self.max_size)
				local box_height = box_width
				-- second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
				-- xmin
				self.output[{1,h,w,1 * 4 + 1}] = (center_x - box_width / 2.0) / img_width
				-- ymin 
				self.output[{1,h,w,1 * 4 + 2}] = (center_y - box_height / 2.0) / img_height
				-- xmax
				self.output[{1,h,w,1 * 4 + 3}] = (center_x + box_width / 2.0) / img_width
				-- ymax
				self.output[{1,h,w,1 * 4 + 4}] = (center_y + box_height / 2.0) / img_height
				ratio_offset = ratio_offset + 1
			end

			-- rest of priors
			for i,ar in ipairs(self.aspect_ratios) do 
				
				if math.abs(ar - 1.) > 1e-6 then
			        box_width = self.min_size * math.sqrt(ar)
					box_height = self.min_size / math.sqrt(ar)
					  -- xmin
					self.output[{1,h,w,(ratio_offset + i) * 4 + 1}] = (center_x - box_width / 2.0) / img_width
					-- ymin 
					self.output[{1,h,w,(ratio_offset + i) * 4 + 2}] = (center_y - box_height / 2.0) / img_height
					-- xmax
					self.output[{1,h,w,(ratio_offset + i) * 4 + 3}] = (center_x + box_width / 2.0) / img_width
					-- ymax
					self.output[{1,h,w,(ratio_offset + i) * 4 + 4}] = (center_y + box_height / 2.0) / img_height
        		end
			end   
		end	
	end
	if self.clip then 
		local bbox = self.output:narrow(1,1,1)
		bbox[torch.gt(bbox,1.0)] = 1.0
		bbox[torch.lt(bbox,0.0)] = 0.0
	end
	for i = 1, 4 do 
		self.output[{{2},{},{},i}] = self.variances[i]
	end
	return self.output:view(1,2,-1)


end

function PriorBox:upadteGradInput(input, gradOutput)
end

function PriorBox:accGradParameters(input, gradOutput)
end

function PriorBox:reset()
end


-- Test
-- local model = nn.PriorBox(46, 96, {2,3}, true, true)
-- local feat_inp = torch.Tensor(5,12, 46,46)
-- local img_inp = torch.Tensor(5, 12, 300,300)
-- local out = model:forward({feat_inp,img_inp})
-- print(out:size())