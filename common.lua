-- common.lua
-- Zdeněk Janeček, 2016 (ycdmdj@gmail.com)
--
-- University of West Bohemia

function momentum_update(derivate, velocity, target, config)
	--~ derivate:mul(-config.learning_rate)

	velocity:mul(config.momentum):add(derivate)
	target:add(velocity)
end

function sample_ber(x)
	local r = torch.rand(x:size())
	if torch.type(x) == 'torch.ClTensor' then
		r = r:cl()
	elseif torch.type(x) == 'torch.CudaTensor' then
		r = r:cuda()
	end

	x:csub(r):sign():clamp(0, 1)
	return x
end

function sample_max(x)
	if x:dim() == 1 then
		local _, pos = torch.max(x, 1)
		x:zero()
		x[pos[1]] = 1
	elseif x:dim() == 2 then
		local _, pos = torch.max(x, 2)
		x:zero()
		for i=1, x:size(1) do
			x[i][pos[i][1]] = 1
		end
	end
end

function sparsity_update(rbm, qold, input, config)
	local v = input[1]
	local y = input[2]

	local target = torch.Tensor(1)

	if config.cuda then
		target = target:cuda()
	end

	-- get moving average of last value and current
	local qcurrent = rbm.mu1:mean(1)[1]
	qcurrent:mul(1-config.sparsity_decay_rate)
    qcurrent:add(config.sparsity_decay_rate, qold)
    qold:copy(qcurrent)

    target:resizeAs(qcurrent)
    target:fill(config.sparsity_target)
    local diffP = qcurrent:csub(target)
    local dP_dW = torch.ger(diffP, v:mean(1)[1])
    local dP_dU = torch.ger(diffP, y:mean(1)[1])

    rbm.weight:csub(dP_dW:mul(config.sparsity_cost))
    rbm.uweight:csub(dP_dU:mul(config.sparsity_cost))
    rbm.hbias:csub(diffP:mul(config.sparsity_cost))
end
