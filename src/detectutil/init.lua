require 'torch'
require 'nn'

detectutil = {}
detectutil.version = 1
require('detectutil.PriorBox')
require('detectutil.DetectionOutput')
require('detectutil.BboxUtil')

return detectutil