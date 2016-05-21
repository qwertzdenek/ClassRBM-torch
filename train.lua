local mnist = require 'mnist'
require 'classrbm'
require 'plot_stats'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a ClassRBM MNIST digit classificator.')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-prefix', '', 'prefix of this run')
cmd:option('-v', false, 'verbose mode')

-- model params
cmd:text('Model parameters')
cmd:option('-n_visible', 28*28, 'RBM hidden layer size.')
cmd:option('-n_hidden', 6000, 'RBM hidden layer size.')
cmd:option('-n_class', 10, 'Number of target classes')

cmd:text('Optimalization parameters')
-- optimization
cmd:option('-learning_rate',0.012,'learning rate')
cmd:option('-momentum',0.5,'momentum')
cmd:option('-L2',0.000022,'L2 decay')
cmd:option('-max_epochs', 4, 'number of full passes through the training data while RBM train')

cmd:option('-sparsity_decay_rate',0.9,'decay rate for sparsity')
cmd:option('-sparsity_target',0.08,'sparsity target')
cmd:option('-sparsity_cost',0.0002,'sparsity cost')

cmd:option('-batch_size',12,'number of sequences to train on in parallel')
cmd:option('-stat_interval',1024,'statistics interval')
cmd:option('-cuda', false,'use CUDA backend')

opt = cmd:parse(arg)
torch.seed()

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

--local y = torch.Tensor(opt.batch_size, opt.n_class)
--local v = torch.Tensor(opt.batch_size, opt.n_visible)
local oneY = torch.Tensor(opt.n_class)

testset['data'] = testset['data']:double():clamp(0,1)
trainset['data'] = trainset['data']:double():clamp(0,1)

testset['label'] = testset['label']:double()
trainset['label'] = trainset['label']:double()

if opt.cuda then
	require 'cutorch'

	testset['data'] = testset['data']:cuda()
	testset['label'] = testset['label']:cuda()
	
	trainset['data'] = trainset['data']:cuda()
	trainset['label'] = trainset['label']:cuda()
	
	y = y:cuda()
	v = v:cuda()
	oneY = oneY:cuda()
end

local validation_size = 256

function reconstruction_test()
	local err = 0

	for i=1, validation_size do
		local index = torch.random(testset.size)
		local v1 = testset[index].x:view(opt.n_visible)
		local y1 = testset[index].y
		oneY:zero()
		oneY[y1+1] = 1
		local v2, y2 = rbm:forward{v1, oneY}
		err = err + (torch.ne(oneY, y2):sum() == 0 and 0 or 1)
	end

	return (1-err/validation_size)*100
end

function reconstruction_train()
	local err = 0

	for i=1, validation_size do
		local index = torch.random(trainset.size)
		local v1 = trainset[index].x:view(opt.n_visible)
		local y1 = trainset[index].y
		oneY:zero()
		oneY[y1+1] = 1
		local v2, y2 = rbm:forward{v1, oneY}
		err = err + (torch.ne(oneY, y2):sum() == 0 and 0 or 1)
	end

	return (1-err/validation_size)*100
end

function free_energy_test()
	local err = 0

	for i=1, validation_size do
		local index = torch.random(testset.size)
		local v1 = testset[index].x:view(opt.n_visible)
		local y1 = testset[index].y
		oneY:zero()
		oneY[y1+1] = 1
		err = err + rbm:freeEnergy(v1, oneY)
	end

	return err/validation_size
end

function free_energy_train()
	local err = 0

	for i=1, validation_size do
		local index = torch.random(trainset.size)
		local v1 = trainset[index].x:view(opt.n_visible)
		local y1 = trainset[index].y
		oneY:zero()
		oneY[y1+1] = 1
		err = err + rbm:freeEnergy(v1, oneY)
	end

	return err/validation_size
end

-- 1) Run RBM pretrain
function pretrain_feval(t)
	--~ for i=1, opt.batch_size do
		--~ local visible = trainset[t*opt.batch_size + i].x
		--~ local class = trainset[t*opt.batch_size + i].y
		--~ v[i]:copy(visible)
		
		--~ y:zero()
		--~ y[i][class+1] = 1
	--~ end
	
	local index = torch.random(trainset.size)
	local v = trainset[index].x:view(opt.n_visible)
	local y = trainset[index].y
	oneY:zero()
	oneY[y+1] = 1

	local _, yt = rbm:forward{v, oneY}
	
	--~ local err = torch.ne(yt, y):sum(2):gt(0):sum() / opt.batch_size
	local err = torch.ne(oneY, yt):sum() == 0 and 0 or 1
	rbm:backward({v, oneY}, nil)

	dl_dx:mul(-opt.learning_rate)

	rbm.gradWeight:add(torch.mul(rbm.gradWeight, -opt.L2))
	rbm.gradUWeight:add(torch.mul(rbm.gradUWeight, -opt.L2))

	momentum_update(dl_dx, velocity, x, opt)
	--sparsity_update(rbm, qval, {v, oneY}, opt)

	return err
end

-- Create RBM
rbm = ClassRBM(opt.n_visible, opt.n_hidden, opt.n_class, opt.batch_size)

-- Training parameters
weightVelocity = rbm.gradWeight:clone()
vbiasVelocity = rbm.gradVbias:clone()
hbiasVelocity = rbm.gradHbias:clone()
uweightVelocity = rbm.gradUWeight:clone()
dbiasVelocity = rbm.gradDbias:clone()

qval = torch.zeros(opt.n_hidden, 1)

if opt.cuda then
	criterion = criterion:cuda()
	rbm = rbm:cuda()
	weightVelocity = weightVelocity:cuda()
	vbiasVelocity = vbiasVelocity:cuda()
	hbiasVelocity = hbiasVelocity:cuda()
	uweightVelocity = uweightVelocity:cuda()
	dbiasVelocity = dbiasVelocity:cuda()
	qval = qval:cuda()
end

velocity = nn.Module.flatten{weightVelocity, vbiasVelocity, hbiasVelocity, uweightVelocity, dbiasVelocity}
x,dl_dx = rbm:getParameters()

histogramValues = {
  weight = rbm.weight,
  vbias = rbm.vbias,
  hbias = rbm.hbias,
  uweight = rbm.uweight,
  dbias = rbm.dbias,

  weightVelocity = weightVelocity,
  vbiasVelocity = vbiasVelocity,
  hbiasVelocity = hbiasVelocity,
  uweightVelocity = uweightVelocity,
  dbiasVelocity = dbiasVelocity
}

training_time = trainset.size/2

err = 0; iter = 0
for epoch=1, opt.max_epochs do
	print('pretrain epoch '..epoch)

	velocity:zero()

	if epoch == math.floor(opt.max_epochs*0.5) then
		torch.save('models/'..opt.prefix..'pretrained_rbm_'..epoch..'.dat', rbm)
		config.momentum = 0.8
	end
	if epoch == math.floor(opt.max_epochs*0.72) then
		config.momentum = 0.9
	end
	if epoch == opt.max_epochs then
		torch.save('models/'..opt.prefix..'pretrained_rbm_'..epoch..'.dat', rbm)
	end

	for t = 0, training_time-1 do
		iter = iter + 1
		xlua.progress(t, training_time)

		err = err + pretrain_feval(t)

		if iter >= opt.stat_interval then
			local test = reconstruction_test(rbm)
			local train = reconstruction_train(rbm)
			local energy_test = free_energy_test(rbm)
			local energy_train = free_energy_train(rbm)
			
			print(string.format('%s t=%d loss=%.4f test=%.4f%% train=%.4f%% ftest=%.4f ftrain=%.4f', os.date("%d/%m %H:%M:%S"), t, err/opt.stat_interval, test, train, energy_test, energy_train))

			-- reset counters
			err = 0; iter = 0

			if opt.v then
				draw_hist(rbm.mu1, 'mean_hidden-'..epoch..'-'..t, 'pravděpodobnost')
				
				draw_stats(histogramValues, 'hist_'..epoch..'-'..t)

				local wm = image.toDisplayTensor{
					  input=rbm.weight:view(torch.LongStorage{rbm.n_hidden, 28, 28}),
					  padding=2, nrow=22}
				image.save('images/weight-map_'..epoch..'-'..t..'.png', wm)
			end
		end
	end
end
