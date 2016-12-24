require 'nn'
require 'math'
require('detectutil.BboxUtil')


--[[
@Brief Generate the detection output based on location regression value and class confidence and prior bounding box
predictions by doing non maximum suppression.
NOTE: does not implement Backwards operation.
]]--

local DetectionOutput, Parent = torch.class('nn.DetectionOutput', 'nn.Module')

function DetectionOutput:__init(num_classes, share_location, background_label_id, nms_threshold, nms_top_k, keep_top_k, confidence_threshold)
	Parent.__init(self)
	self.num_classes = num_classes or 21
	self.share_location = share_location or true
	self.background_label_id = share_location or self.num_classes
	self.nms_threshold = nms_threshold or 0.45
	self.nms_top_k = nms_top_k or 400
	self.keep_top_k = keep_top_k or self.nms_top_k * 0.5 
	self.confidence_threshold = confidence_threshold or 0.01
end
-- input[1] is mbox location regression score of shape N*(C1x4), C1 is total feature maps locations x #anchors
-- input[2] is mbox class confidence of shape N*(C1xK), K is class number.  
-- input[3] is prior bounding box    of shape 1*2*(C1x4)
-- input 
-- output will be of shape M*7, 7 stands for [image_id, label, confidence, xmin, ymin,xmax, ymax],
										-- note xmin, ymin, xmax,ymax are in normalized form
function DetectionOutput:updateOutput(input)
	local batch_size = input[1]:size(1)
	local prior_bboxes = input[3]:view(2,-1,4)
	local res_bboxes_table = {}
	for n = 1, batch_size do 
		timer = torch.Timer() -- the Timer starts to count now
		local bbox_regress_value = input[1]:narrow(1,n,1):view(-1,4)
		local bbox_conf_scores   = input[2]:narrow(1,n,1):view(-1,self.num_classes)
		local decoded_bboxes = DecodeBatchBBox(prior_bboxes:select(1,1), prior_bboxes:select(1,2),bbox_regress_value)
		print('Time elapsed: ' .. timer:time().real .. ' seconds')

		for k = 1, self.num_classes do 
			if k ~= self.background_label_id then
				local bbox_class_score = bbox_conf_scores:select(2,k)
				timer2 = torch.Timer() -- the Timer starts to count now
				local nmsed_bboxes_indices = ApplyNMSFast(decoded_bboxes, bbox_class_score, self.confidence_threshold, 
															self.nms_threshold, self.nms_top_k)
				local res_item = torch.Tensor(#nmsed_bboxes_indices,7)
				for i = 1, #nmsed_bboxes_indices do 
					bboxes_index = nmsed_bboxes_indices[i]
					res_item[{i,1}] = n
					res_item[{i,2}] = k
					res_item[{i,3}] = bbox_class_score[bboxes_index]
					res_item[{i,4}] = decoded_bboxes[{bboxes_index,1}]
					res_item[{i,5}] = decoded_bboxes[{bboxes_index,2}]
					res_item[{i,6}] = decoded_bboxes[{bboxes_index,3}]
					res_item[{i,7}] = decoded_bboxes[{bboxes_index,4}]
				end
				table.insert(res_bboxes_table, res_item)
			end
		end
	end
	res_bboxes_table = torch.cat(res_bboxes_table, 1)
	if res_bboxes_table:size(1) > self.keep_top_k then 
		local ys, inds = torch.topk(res_bboxes_table:select(2,3), self.keep_top_k, 1, true, true)
		self.output = torch.Tensor(self.keep_top_k, 7)
		for ind  = 1, self.keep_top_k do 
			self.output[{ind,{}}] = res_bboxes_table[{inds[ind],{}}]
		end
	else
		self.output = res_bboxes_table 
	end
	return self.output
end

function DetectionOutput:upadteGradInput(input, gradOutput)
	assert(true, "Backward not finished!")
end

function DetectionOutput:accGradParameters(input, gradOutput)
	assert(true, "Backward not finished!")
end

function DetectionOutput:reset()
end

-- Test 
-- local model = nn.DetectionOutput(21, true, 21, 0.45, 400, 200, 0.01)
-- local mbox_loc = torch.randn(3,120*4)
-- local mbox_conf = torch.randn(3,120*21)
-- local mbox_prior = torch.randn(1,2,120*4)

-- local out = model:forward({mbox_loc,mbox_conf,mbox_prior})
-- print(out:size())