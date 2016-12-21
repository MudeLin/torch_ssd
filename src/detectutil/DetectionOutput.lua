require 'nn'
require 'math'
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
-- input[1] is mbox location regression score
-- input[2] is mbox class confidence 
-- input[3] is prior bounding box
function DetectionOutput:updateOutput(input)

end

function DetectionOutput:upadteGradInput(input, gradOutput)
	assert(true, "Backward not finished!")
end

function DetectionOutput:accGradParameters(input, gradOutput)
	assert(true, "Backward not finished!")
end

function DetectionOutput:reset()
end
