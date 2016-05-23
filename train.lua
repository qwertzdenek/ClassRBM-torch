-- train.lua
-- Zdeněk Janeček, 2016 (ycdmdj@gmail.com)
--
-- University of West Bohemia

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
cmd:option('-learning_rate',0.005,'learning rate')
cmd:option('-momentum',0.5,'momentum')
cmd:option('-L2',0.000022,'L2 decay')
cmd:option('-max_epochs', 60, 'number of full passes through the training data while RBM train')

--cmd:option('-sparsity_decay_rate',0.9,'decay rate for sparsity')
--cmd:option('-sparsity_target',0.08,'sparsity target')
--cmd:option('-sparsity_cost',0.0002,'sparsity cost')

cmd:option('-batch_size',12,'number of sequences to train on in parallel')
cmd:option('-stat_interval',4092,'statistics interval')
cmd:option('-cuda', false,'use CUDA backend')
cmd:option('-opencl', false,'use OpenCL backend')

opt = cmd:parse(arg)
torch.seed()

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local oneY = torch.Tensor(opt.n_class)

testset['data'] = testset['data']:double():clamp(0,1)
trainset['data'] = trainset['data']:double():clamp(0,1)

testset['label'] = testset['label']:double()
trainset['label'] = trainset['label']:double()

if opt.cuda then
	require 'cutorch'
	require 'cunn'

	testset['data'] = testset['data']:cuda()
	testset['label'] = testset['label']:cuda()

	trainset['data'] = trainset['data']:cuda()
	trainset['label'] = trainset['label']:cuda()

	oneY = oneY:cuda()
elseif opt.opencl then
	require 'cltorch'
	require 'clnn'

	testset['data'] = testset['data']:cl()
	testset['label'] = testset['label']:cl()

	trainset['data'] = trainset['data']:cl()
	trainset['label'] = trainset['label']:cl()

	oneY = oneY:cl()
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
	local index = torch.random(trainset.size)
	local v = trainset[index].x:view(opt.n_visible)
	local y = trainset[index].y
	oneY:zero()
	oneY[y+1] = 1

	local _, yt = rbm:forward{v, oneY}

	local err = torch.ne(oneY, yt):sum() == 0 and 0 or 1
	rbm:backward({v, oneY}, nil)

	dl_dx:mul(-opt.learning_rate)

	rbm.gradWeight:add(torch.mul(rbm.gradWeight, -opt.L2))
	rbm.gradUWeight:add(torch.mul(rbm.gradUWeight, -opt.L2))

	momentum_update(dl_dx, velocity, x, opt)
	--sparsity_update(rbm, qval, {v, oneY}, opt)

	return err
end

function store_rbm(params, name)
	local target_rbm = ClassRBM(opt.n_visible, opt.n_hidden, opt.n_class, opt.batch_size)
	local p, _ = target_rbm:getParameters()
	p:copy(params)
	torch.save(name, target_rbm)
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
	rbm = rbm:cuda()
	qval = qval:cuda()
	weightVelocity = weightVelocity:cuda()
	vbiasVelocity = vbiasVelocity:cuda()
	hbiasVelocity = hbiasVelocity:cuda()
	uweightVelocity = uweightVelocity:cuda()
	dbiasVelocity = dbiasVelocity:cuda()
elseif opt.opencl then
	rbm = rbm:cl()
	qval = qval:cl()
	weightVelocity = weightVelocity:cl()
	vbiasVelocity = vbiasVelocity:cl()
	hbiasVelocity = hbiasVelocity:cl()
	uweightVelocity = uweightVelocity:cl()
	dbiasVelocity = dbiasVelocity:cl()
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

err = 0; iter = 0; patience = 15; best_val_err = 1/0
best_rbm = torch.Tensor()
best_rbm:resizeAs(x):copy(x)
for epoch=1, opt.max_epochs do
	print('pretrain epoch '..epoch)

	velocity:zero()

	if epoch == math.floor(opt.max_epochs*0.5) then
		config.momentum = 0.8
	elseif epoch == math.floor(opt.max_epochs*0.72) then
		config.momentum = 0.9
	end

	epoch_err = 0

	for t = 0, training_time-1 do
		iter = iter + 1
		--xlua.progress(t, training_time)

		curr_err = pretrain_feval(t)
		epoch_err = epoch_err + curr_err
		err = err + curr_err

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

	if epoch_err < best_val_err then
		best_val_err = epoch_err
		best_rbm:copy(x)

		patience = 15
	else
		patience = patience - 1
		print('-> patience=', patience)

		if patience < 0  then
			break
		end
	end

	if epoch == math.floor(opt.max_epochs*0.5) then
		store_rbm(best_rbm, 'models/'..opt.prefix..'pretrained_rbm_'..epoch..'.dat')
	end
end

store_rbm(best_rbm, 'models/'..opt.prefix..'pretrained_rbm_final.dat')
