using Plots
using Roots
using Distributions
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(6)

N = 10000


T = 10000

Ts = [1000:1000:9000;10000:10000:90000;100000:100000:900000;1000000:1000000:T]

# c = 1

c1 = sqrt(8)
c = c1*sqrt(log(N))

loss_hedge_adv = zeros(T)
loss_hedge_stoch = zeros(T)


Loss_hedge_adv = zeros(T)
Loss_hedge_stoch = zeros(T)

Loss_opt_adv = zeros(T)
Loss_opt_stoch = zeros(T)

w_hedge_adv = ones(N)/N
w_hedge_stoch = ones(N)/N



entropy = w -> -(w' * log.(w))
function expwt(lr,Loss)
  exp.(-lr*(Loss .-minimum(Loss)))/sum(exp.(-lr*(Loss .-minimum(Loss))))
end

# function implr(Loss,t)
#   expwtLoss = lr -> (lr-c1*sqrt((entropy(expwt(lr, Loss))+c2)/t)) 
#   lr_hat = find_zero(expwtLoss,(0,c1*sqrt((log(N)+c2)/t)))
#   lr_hat
# end
# 
# function impwt(Loss,t)
#   expwt(implr(Loss,t),Loss)
# end

# losses_adv = rand(Bernoulli(0.5),N)
# losses_stoch = [rand(Bernoulli(0.4),1); rand(Bernoulli(0.5),N-1)]

noise = rand(N)
losses_adv = noise .<= 0.5
losses_stoch = noise .<= [0.25; 0.5*ones(N-1)]



Losses_adv = losses_adv
Losses_stoch = losses_stoch

loss_hedge_adv[1] = losses_adv' * w_hedge_adv
Loss_hedge_adv[1] = loss_hedge_adv[1]

loss_hedge_stoch[1] = losses_stoch' * w_hedge_stoch
Loss_hedge_stoch[1] = loss_hedge_stoch[1]

for t in 2:T
  # losses_adv = rand(Bernoulli(0.5),N)
  # losses_stoch = [rand(Bernoulli(0.4),1); rand(Bernoulli(0.5),N-1)]
  noise = rand(N)
  losses_adv = noise .<= 0.5
  losses_stoch = noise .<= [0.25; 0.5*ones(N-1)]

  global w_hedge_adv = expwt(c/sqrt(t), Losses_adv)
  global loss_hedge_adv[t] = losses_adv' * w_hedge_adv
  global Loss_hedge_adv[t] = Loss_hedge_adv[t-1] + loss_hedge_adv[t]

  global w_hedge_stoch = expwt(c/sqrt(t), Losses_stoch)
  global loss_hedge_stoch[t] = losses_stoch' * w_hedge_stoch
  global Loss_hedge_stoch[t] = Loss_hedge_stoch[t-1] + loss_hedge_stoch[t]

  
  global Losses_adv = Losses_adv + losses_adv
  global Losses_stoch = Losses_stoch + losses_stoch

  global Loss_opt_adv[t] = minimum(Losses_adv)
  global Loss_opt_stoch[t] = minimum(Losses_stoch)

  # if t in Ts
  #   print("t=T"*string(t)*"\n")
  #   global Regret_hedge_adv = Loss_hedge_adv - Loss_opt_adv
  #   global Regret_hedge_stoch = Loss_hedge_stoch - Loss_opt_stoch
  # 
  # 
  #   p = plot(1:t,Regret_hedge_adv[1:t], label="Adversarial", ylabel = "Regret", xlabel="Time",legend=:bottomright)
  #   plot!(1:t,Regret_hedge_stoch[1:t],label = "Stochastic")
  # end
end

Regret_hedge_adv = Loss_hedge_adv - Loss_opt_adv
Regret_hedge_stoch = Loss_hedge_stoch - Loss_opt_stoch

Plots.scalefontsizes(1.5)

p = plot(1:T,Regret_hedge_adv[1:T], label="Adversarial", ylabel = "Regret", xlabel="Time",legend=:topleft,  linewidth = 3, color=:dodgerblue)
plot!(1:T,Regret_hedge_stoch[1:T],label = "Stochastic",  linewidth = 3, color=:darkgoldenrod3)
# vline!([log(N)/(0.1)^2], label="log N / Δ²")
# hline!([log(N)/(0.1)], label="log N / Δ")

p

savefig(p, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/stoch-adv-regret-T"*string(T)*"-N"*string(N)*".png")

Plots.scalefontsizes(1/1.5)
