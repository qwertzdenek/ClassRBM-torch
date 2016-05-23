-- common.lua
-- Zdeněk Janeček, 2016 (ycdmdj@gmail.com)
--
-- University of West Bohemia

require 'common'
require 'nn'

local ClassRBM, parent = torch.class('ClassRBM', 'nn.Module')

function ClassRBM:__init(n_visible, n_hidden, n_class, batch_size)
    parent.__init(self)
    self.n_visible = n_visible
    self.n_hidden = n_hidden
    self.n_class = n_class

    self.input = torch.Tensor()
    self.y = torch.Tensor()
    self.r = torch.Tensor()
    
    self.vt = torch.Tensor()
    self.yt = torch.Tensor()
    self.mu1 = torch.Tensor() -- statistics reasons

    self.weight = torch.Tensor(self.n_hidden, self.n_visible)
    self.vbias = torch.Tensor(self.n_visible)
    self.hbias = torch.Tensor(self.n_hidden)
    self.uweight = torch.Tensor(self.n_hidden, self.n_class)
    self.dbias = torch.Tensor(self.n_class)

    self.gradWeight = torch.Tensor(self.n_hidden, self.n_visible)
    self.gradVbias = torch.Tensor(self.n_visible)
    self.gradHbias = torch.Tensor(self.n_hidden)
    self.gradUWeight = torch.Tensor(self.n_hidden, self.n_class)
    self.gradDbias = torch.Tensor(self.n_class)

    self.posGrad = torch.Tensor(self.n_hidden, self.n_visible)
    self.negGrad = torch.Tensor(self.n_hidden, self.n_visible)

    self.posGradClass = torch.Tensor(self.n_hidden, self.n_class)
    self.negGradClass = torch.Tensor(self.n_hidden, self.n_class)

    self.cdSteps = 1

    self:reset()
end

-- propup
function ClassRBM:updateOutputExpected(v_t, y_t)
    if v_t:dim() == 1 then
      self.output:resize(self.n_hidden)
      self.output:copy(self.hbias)
      self.output:addmv(1, self.weight, v_t)
      self.output:addmv(1, self.uweight, y_t)
   elseif v_t:dim() == 2 then
      local nframe = v_t:size(1)
      self.output:resize(nframe, self.n_hidden)
      self.output:copy(self.hbias:view(1, self.n_hidden)
	      :expand(nframe, self.n_hidden))
      self.output:addmm(1, v_t, self.weight:t())
      self.output:addmm(1, y_t, self.uweight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output:sigmoid()
end

-- propdown
function ClassRBM:updateInputExpected(h_t)
    if h_t:dim() == 1 then
      self.input:resize(self.n_visible)
      self.input:copy(self.vbias)
      self.input:addmv(1, self.weight:t(), h_t)
   elseif h_t:dim() == 2 then
      local nframe = h_t:size(1)
      self.input:resize(nframe, self.n_visible)
      self.input:copy(self.vbias:view(1, self.n_visible)
          :expand(nframe, self.n_visible))
      self.input:addmm(1, h_t, self.weight)
   else
      error('hidden must be vector or matrix')
   end

   return self.input:sigmoid()
end

function ClassRBM:updateInputClass(h_t)
	if h_t:dim() == 1 then
      self.y:resize(self.n_class)
      self.y:copy(self.dbias)
      self.y:addmv(1, self.uweight:t(), h_t)
      self.y:exp()
	  self.y:div(self.y:sum())
   elseif h_t:dim() == 2 then
      local nframe = h_t:size(1)
      self.y:resize(nframe, self.n_class)
      self.y:copy(self.dbias:view(1, self.n_class)
          :expand(nframe, self.n_class))
      self.y:addmm(1, h_t, self.uweight)
      self.y:exp()
      self.y:cdiv(self.y:sum(2):expand(nframe, self.n_class))
   else
      error('hidden must be vector or matrix')
   end

   return self.y
end

function ClassRBM:updateInput(h_t)
	return self:updateInputExpected(h_t), self:updateInputClass(h_t)
end

-- do Gibbs sampling
function ClassRBM:updateOutput(input)
    local vt = input[1]
    local yt = input[2]
    
    for t=1, self.cdSteps do
        local ht = sample_ber(self:updateOutputExpected(vt, yt), self.r)
        vt, yt = self:updateInput(ht)
        sample_ber(vt, self.r)
        sample_max(yt)
    end
    
    self.vt:resizeAs(vt)
    self.vt:copy(vt)
    
    self.yt:resizeAs(yt)
    self.yt:copy(yt)

    return vt, yt
end

-- we don't need gratOutput in unsupervised greedy training
function ClassRBM:updateGradInput(input)
    local v1 = input[1]
    local vt = self.vt
    local y = input[2]
    local yt = self.yt

    local mu1 = self:updateOutputExpected(v1, y)
    self.mu1:resizeAs(mu1)
    self.mu1:copy(mu1)

    local mut = self:updateOutputExpected(vt, yt)

	if v1:dim() == 1 then
		-- gradients
		torch.ger(self.posGrad, self.mu1, v1)
		torch.ger(self.negGrad, mut, vt)
		
		torch.csub(self.gradWeight, self.negGrad, self.posGrad)
		torch.csub(self.gradVbias, vt, v1)
		torch.csub(self.gradHbias, mut, self.mu1)
		
		-- class gradients
		torch.ger(self.posGradClass, self.mu1, y)
		torch.ger(self.negGradClass, mut, yt)
		
		torch.csub(self.gradUWeight, self.negGradClass, self.posGradClass)
		torch.csub(self.gradDbias, yt, y)
	elseif v1:dim() == 2 then
		local nframe = v1:size(1)

		-- gradients
		torch.mm(self.posGrad, self.mu1:t(), v1)
		torch.mm(self.negGrad, mut:t(), vt)

		torch.csub(self.gradWeight, self.negGrad, self.posGrad)
		self.gradWeight:div(nframe)

		torch.mean(self.gradVbias, vt:csub(v1), 1)
		torch.mean(self.gradHbias, mut-self.mu1, 1)
		
		-- class gradients
		torch.mm(self.posGradClass, self.mu1:t(), y)
		torch.mm(self.negGradClass, mut:t(), yt)

		torch.csub(self.gradUWeight, self.negGradClass, self.posGradClass)
		self.gradUWeight:div(nframe)

		torch.mean(self.gradDbias, yt:csub(y), 1)
	else
		error('input must be vector or matrix')
	end

    return self.gradInput
end

function ClassRBM:freeEnergy(visible, class)
	if visible:dim() == 1 then
		self.output:resize(self.n_hidden)
		self.output:copy(self.hbias)
		self.output:addmv(1, self.weight, visible)
		self.output:addmv(1, self.uweight, class):exp():add(1):log()

		local neg = self.output:sum()
		local pos = torch.dot(class, self.dbias)
		return -neg-pos
	elseif visible:dim() == 2 then
		local nframe = visible:size(1)
		self.output:resize(nframe, self.n_hidden)
		self.output:copy(self.hbias:view(1, self.n_hidden)
			:expand(nframe, self.n_hidden))
        self.output:addmm(1, visible, self.weight:t())
        self.output:addmm(1, class, self.uweight:t()):exp():add(1):log()

		local neg = self.output:sum(2)
		local pos = torch.mv(class, self.dbias)
		return (-neg-pos):sum() / nframe
	end
end

function ClassRBM:reset()
    self.weight:normal(0, 0.08)
    self.uweight:normal(0, 0.08)
	self.vbias:zero()
	self.hbias:zero()
	self.dbias:zero()

    self.gradWeight:zero()
    self.gradHbias:zero()
    self.gradVbias:zero()
    self.gradUWeight:zero()
    self.gradDbias:zero()
end

function ClassRBM:parameters()
   return {self.weight, self.vbias, self.hbias, self.uweight, self.dbias},
          {self.gradWeight, self.gradVbias, self.gradHbias, self.gradUWeight, self.gradDbias}
end
