
require 'math'
require 'torch'
function DecodeBBox(prior_bbox, prior_variance, loc_regress_value)
	local prior_width = prior_bbox[3] - prior_bbox[1]
	local prior_height = prior_bbox[4] - prior_bbox[2]
	local prior_center_x = prior_bbox[1] + prior_width / 2.0
	local prior_center_y = prior_bbox[2] + prior_height / 2.0

	local decode_center_x = loc_regress_value[1] * prior_variance[1] * prior_width + prior_center_x
	local decode_center_y = loc_regress_value[2] * prior_variance[2] * prior_height + prior_center_y
	local decode_width = math.exp(prior_variance[3] * loc_regress_value[3]) * prior_width
	local decode_height = math.exp(prior_variance[4] * loc_regress_value[4]) * prior_height
	local decoded_bbox = torch.FloatTensor(4)
	decoded_bbox[1] = decode_center_x - decode_width / 2.0
	decoded_bbox[2] = decode_center_y - decode_height / 2.0
	decoded_bbox[3] = decode_center_x + decode_width / 2.0
	decoded_bbox[4] = decode_center_y + decode_height / 2.0
	return decoded_bbox
end
-- return the local regression value
function EncodeBBox(prior_bbox, prior_variance, gt_bbox)
	local prior_width = prior_bbox[3] - prior_bbox[1]
	local prior_height = prior_bbox[4] - prior_bbox[2]
	local prior_center_x = prior_bbox[1] + prior_width / 2.0
	local prior_center_y = prior_bbox[2] + prior_height / 2.0

	local gt_width = gt_bbox[3] - gt_bbox[1]
	local gt_height = gt_bbox[4] - gt_bbox[2]
	local gt_center_x = gt_bbox[1] + gt_width / 2.0
	local gt_center_y = gt_bbox[2] + gt_height / 2.0

	local encode_bbox = torch.FloatTensor(4)
	encode_bbox[1] = (gt_center_x - prior_center_x) / prior_width / prior_variance[1]
	encode_bbox[2] = (gt_center_y - prior_center_y) / prior_height / prior_variance[2]
	encode_bbox[3] = math.log( gt_width / prior_width) / prior_variance[3]
	encode_bbox[4] = math.log( gt_height / prior_height) / prior_variance[4]
	return encode_bbox
end


function BBoxArea(bbox)
	if bbox[1] > bbox[3] or bbox[2] > bbox[4] then 
		return 0.0
	else
		return (bbox[4] - bbox[2]) * (bbox[3] - bbox[1])
	end
end
function JaccardOverlap(bbox1, bbox2)
	-- calculate intersect bbox
	local intersect_bbox = torch.FloatTensor(4)
	if bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1] or
	   bbox2[2] > bbox1[4] or bbox2[4] < bbox1[2]  then 
	   intersect_bbox:fill(0.0)
	else
		intersect_bbox[1] = math.max(bbox1[1],bbox2[1])
		intersect_bbox[2] = math.max(bbox1[2],bbox2[2])
		intersect_bbox[3] = math.min(bbox1[3],bbox2[3])
		intersect_bbox[4] = math.min(bbox1[4],bbox2[4])
	end
	local intersect_area = BBoxArea(intersect_bbox)
	local overlap =  intersect_area / (BBoxArea(bbox1) + BBoxArea(bbox2) - intersect_area)
	return overlap
end
-- bboxes is a tensor of shape Nx4
-- scores is a tensor of shape Nx1
-- top_k , at most k bounding box is considered
function ApplyNMSFast(bboxes, scores, score_threshold, nms_threshold, top_k)
	assert(bboxes:size(1) == scores:size(1), 'Scores and bbox number not matched' )
	top_k = top_k or scores:size(1)
	local top_scores, top_indices = torch.topk(scores, top_k, 1, true, true)
	-- Do nms.
	local bbox_indices = {}
	for top_ind = 1, top_indices:size(1) do
		local bbox_ind = top_indices[top_ind]
		if BBoxArea(bboxes[bbox_ind]) > 0 then 
			local keep = true
			for k = 1, #bbox_indices do 
				if keep then 
					local kept_ind = bbox_indices[k]
					local overlap = JaccardOverlap(bboxes[bbox_ind], bboxes[kept_ind])
					keep = overlap <= nms_threshold and scores[bbox_ind] > score_threshold
				else
					break
				end
			end
			if keep then 
				table.insert(bbox_indices, bbox_ind)
			end
		end

		if scores[bbox_ind] < score_threshold then 
			break
		end
	end
	return bbox_indices
end

require 'image'
function VisualizeBBox(bboxes, img)
	local width = img:size(2)
	local height = img:size(3)
	local res_img = img

	for k = 1, 5 do
		print(bboxes[k][1]*width)
		print(bboxes[k][3]*height)
		print(bboxes[k][2]*width)
		print(bboxes[k][4]*height)
		print('next')
		res_img = image.drawRect(res_img, bboxes[k][1]*width,bboxes[k][2]*height,bboxes[k][3]*width,bboxes[k][4]*height,  {lineWidth = 1, color = {255, 0, 255}})
	end
	return res_img
end
-- -- Test encode decode bbox
-- prior_bbox = torch.randn(4)
-- loc_regress_value = torch.randn(4)
-- prior_variance = torch.randn(4):fill(0.1)
-- decoded_bbox = DecodeBBox(prior_bbox, prior_variance, loc_regress_value)
-- print(loc_regress_value)
-- encoded_loc_regress_value = EncodeBBox(prior_bbox, prior_variance, decoded_bbox)
-- print(encoded_loc_regress_value)

-- gt_bbox = torch.randn(4)
-- encoded_loc_regress_value = EncodeBBox(prior_bbox, prior_variance, gt_bbox)
-- print(gt_bbox)
-- decoded_bbox = DecodeBBox(prior_bbox, prior_variance, encoded_loc_regress_value)
-- print(decoded_bbox)

-- Test nms
bboxes = torch.randn(100000,4)
bboxes[torch.gt(bboxes,1.0)] = 0.99
bboxes[torch.lt(bboxes,0.0)] = 0.01

scores = torch.randn(100000)
score_threshold = 0.2
nms_threshold = 0.45
top_k = 300
local bbox_indices = ApplyNMSFast(bboxes,scores, score_threshold, nms_threshold, top_k)

vis_bboxes = torch.FloatTensor(#bbox_indices, 4)

for k = 1, #bbox_indices do 
	vis_bboxes[{k,{}}] =  bboxes[{bbox_indices[k],{}}]
end
local img = torch.FloatTensor(3,368,368)
local res_img = VisualizeBBox(vis_bboxes,img)
image.display(res_img)