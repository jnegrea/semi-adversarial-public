using Plots
using Roots
using Distributions
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(6)

N = 10000
N0= 10

T = 1000000

Ts = [1000:1000:9000;10000:10000:90000;100000:100000:900000;1000000:1000000:T]

# c = 1

c1 = sqrt(2/3)
c2 = 1 
# chedge = (128/9)^0.25 *sqrt(log(N))
cmeta=sqrt(8*log(2))

c = ((128/9)^0.25)*sqrt(log(N))


loss_hedge_adv = zeros(T)
loss_hedge_stoch = zeros(T)
loss_hedge_semi = zeros(T)

Loss_hedge_adv = zeros(T)
Loss_hedge_stoch = zeros(T)
Loss_hedge_semi = zeros(T)

w_hedge_adv = ones(N)/N
w_hedge_stoch = ones(N)/N
w_hedge_semi = ones(N)/N



loss_care_adv = zeros(T)
loss_care_stoch = zeros(T)
loss_care_semi = zeros(T)

Loss_care_adv = zeros(T)
Loss_care_stoch = zeros(T)
Loss_care_semi = zeros(T)

w_care_adv = ones(N)/N
w_care_stoch = ones(N)/N
w_care_semi = ones(N)/N



loss_meta_adv = zeros(T)
loss_meta_stoch = zeros(T)
loss_meta_semi = zeros(T)

Loss_meta_adv = zeros(T)
Loss_meta_stoch = zeros(T)
Loss_meta_semi = zeros(T)

w_meta_adv = ones(N)/N
w_meta_stoch = ones(N)/N
w_meta_semi = ones(N)/N



Loss_opt_adv = zeros(T)
Loss_opt_stoch = zeros(T)
Loss_opt_semi = zeros(T)



entropy = w -> -(w' * log.(w))
function expwt(lr,Loss)
  exp.(-lr*(Loss .-minimum(Loss)))/sum(exp.(-lr*(Loss .-minimum(Loss))))
end

function implr(Loss,t)
  expwtLoss = lr -> (lr-c1*sqrt((entropy(expwt(lr, Loss))+c2)/t)) 
  lr_hat = find_zero(expwtLoss,(0,c1*sqrt((log(N)+c2)/t)+1e-8))
  lr_hat
end

function carewt(Loss,t)
  expwt(implr(Loss,t),Loss)
end

function metawt(cmeta,Loss_hedge, Loss_care, w_hedge, w_care, t)
  w_meta_hedge = exp(-cmeta/sqrt(t)*Loss_hedge)/(exp(-cmeta/sqrt(t)*Loss_hedge)+exp(-cmeta/sqrt(t)*Loss_care))
  
  w_meta_hedge*w_hedge+(1-w_meta_hedge)*w_care
end

# losses_adv = rand(Bernoulli(0.5),N)
# losses_stoch = [rand(Bernoulli(0.4),1); rand(Bernoulli(0.5),N-1)]
# losses_semi = [rand(Bernoulli(0.4),N0); rand(Bernoulli(0.5),N-N0)]

noise = rand(N)
losses_adv = noise .<= 0.5
losses_stoch = noise .<= [0.45; 0.5*ones(N-1)]
losses_semi = noise .<= [0.45*ones(N0); 0.5*ones(N-N0)]

# scale = 0.2
# scale2= 0.1
# losses_adv = scale2.+scale*(((1:N).%2 .+ 1%2).%2)
# losses_stoch = [scale2*((1%2 .+ 1%2).%2); scale2.+(N*scale-scale2)/(N-1)*(((2:N).%2 .+ 1%2)).%2]
# losses_semi = [scale2*(((1:N0).%2 .+ 1%2).%2); scale2.+(N*scale-N0*scale2)/(N-N0)*((((N0+1):N).%2 .+ 1%2).%2)]

# scale = 0.1
# scale2= 0.1
# losses_adv = scale2.+scale*(((1:N).%2 .+ 1%2).%2)
# losses_stoch = [scale2*((1%2 .+ 1%2).%2); scale.+scale2*(((2:N).%2 .+ 1%2)).%2]
# losses_semi = [scale2*(((1:N0).%2 .+ 1%2).%2); scale.+scale2*((((N0+1):N).%2 .+ 1%2).%2)]



Losses_adv = losses_adv
Losses_stoch = losses_stoch
Losses_semi = losses_semi



loss_hedge_adv[1] = losses_adv' * w_hedge_adv
Loss_hedge_adv[1] = loss_hedge_adv[1]

loss_care_stoch[1] = losses_stoch' * w_care_stoch
Loss_care_stoch[1] = loss_care_stoch[1]



loss_meta_adv[1] = losses_adv' * w_meta_adv
Loss_meta_adv[1] = loss_meta_adv[1]

loss_meta_stoch[1] = losses_stoch' * w_meta_stoch
Loss_meta_stoch[1] = loss_meta_stoch[1]



loss_meta_adv[1] = losses_adv' * w_meta_adv
Loss_meta_adv[1] = loss_meta_adv[1]

loss_meta_stoch[1] = losses_stoch' * w_meta_stoch
Loss_meta_stoch[1] = loss_meta_stoch[1]

for t in 2:T
  # losses_adv = rand(Bernoulli(0.5),N)
  # losses_stoch = [rand(Bernoulli(0.45),1); rand(Bernoulli(0.5),N-1)]
  # losses_semi = [rand(Bernoulli(0.45),N0); rand(Bernoulli(0.5),N-N0)]
  
  noise = rand(N)
  losses_adv = noise .<= 0.5
  losses_stoch = noise .<= [0.45; 0.5*ones(N-1)]
  losses_semi = noise .<= [0.45*ones(N0); 0.5*ones(N-N0)]
  
  # losses_adv = scale2.+scale*(((1:N).%2 .+ t%2).%2)
  # losses_stoch = [scale2*((1%2 .+ t%2).%2); scale2.+(N*scale-1*scale2)/(N-1)*(((2:N).%2 .+ t%2).%2)]
  # losses_semi = [scale2*(((1:N0).%2 .+ t%2).%2); scale2.+(N*scale-N0*scale2)/(N-N0)*((((N0+1):N).%2 .+ t%2).%2)]
  
  # losses_adv = scale2.+scale*(((1:N).%2 .+ t%2).%2)
  # losses_stoch = [scale2*((1%2 .+ t%2).%2); scale.+scale2*(((2:N).%2 .+ t%2).%2)]
  # losses_semi = [scale2*(((1:N0).%2 .+ t%2).%2); scale.+scale2*((((N0+1):N).%2 .+ t%2).%2)]

  global w_hedge_adv = expwt(c/sqrt(t), Losses_adv)
  global loss_hedge_adv[t] = losses_adv' * w_hedge_adv
  global Loss_hedge_adv[t] = Loss_hedge_adv[t-1] + loss_hedge_adv[t]
  
  global w_hedge_stoch = expwt(c/sqrt(t), Losses_stoch)
  global loss_hedge_stoch[t] = losses_stoch' * w_hedge_stoch
  global Loss_hedge_stoch[t] = Loss_hedge_stoch[t-1] + loss_hedge_stoch[t]
  
  global w_hedge_semi = expwt(c/sqrt(t), Losses_semi)
  global loss_hedge_semi[t] = losses_semi' * w_hedge_semi
  global Loss_hedge_semi[t] = Loss_hedge_semi[t-1] + loss_hedge_semi[t]



  global w_care_adv = carewt(Losses_adv,2)
  global loss_care_adv[t] = losses_adv' * w_care_adv
  global Loss_care_adv[t] = Loss_care_adv[t-1] + loss_care_adv[t]
  
  global w_care_stoch = carewt(Losses_stoch,2)
  global loss_care_stoch[t] = losses_stoch' * w_care_stoch
  global Loss_care_stoch[t] = Loss_care_stoch[t-1] + loss_care_stoch[t]
  
  global w_care_semi = carewt(Losses_semi,2)
  global loss_care_semi[t] = losses_semi' * w_care_semi
  global Loss_care_semi[t] = Loss_care_semi[t-1] + loss_care_semi[t]

  
  
  global w_meta_adv = metawt(cmeta,Loss_hedge_adv[t-1], Loss_care_adv[t-1], w_hedge_adv, w_care_adv, t)
  global loss_meta_adv[t] = losses_adv' * w_meta_adv
  global Loss_meta_adv[t] = Loss_meta_adv[t-1] + loss_meta_adv[t]
  
  global w_meta_stoch = metawt(cmeta,Loss_hedge_stoch[t-1], Loss_care_stoch[t-1], w_hedge_stoch, w_care_stoch, t)
  global loss_meta_stoch[t] = losses_stoch' * w_meta_stoch
  global Loss_meta_stoch[t] = Loss_meta_stoch[t-1] + loss_meta_stoch[t]
  
  global w_meta_semi = metawt(cmeta,Loss_hedge_semi[t-1], Loss_care_semi[t-1], w_hedge_semi, w_care_semi, t)
  global loss_meta_semi[t] = losses_semi' * w_meta_semi
  global Loss_meta_semi[t] = Loss_meta_semi[t-1] + loss_meta_semi[t]
  
  
    
  global Losses_adv = Losses_adv + losses_adv
  global Losses_stoch = Losses_stoch + losses_stoch
  global Losses_semi = Losses_semi + losses_semi


  global Loss_opt_adv[t] = minimum(Losses_adv)
  global Loss_opt_stoch[t] = minimum(Losses_stoch)
  global Loss_opt_semi[t] = minimum(Losses_semi)


  if t in Ts
    print("t=T"*string(t)*"\n")
    # global Regret_meta_adv = Loss_meta_adv - Loss_opt_adv
    # global Regret_meta_stoch = Loss_meta_stoch - Loss_opt_stoch
    # 
    # 
    # p = plot(1:t,Regret_meta_adv[1:t], label="Adversarial", ylabel = "Regret", xlabel="Time",legend=:bottomright)
    # plot!(1:t,Regret_meta_stoch[1:t],label = "Stochastic")
  end
end

Regret_meta_adv = Loss_meta_adv - Loss_opt_adv
Regret_meta_stoch = Loss_meta_stoch - Loss_opt_stoch
Regret_meta_semi = Loss_meta_semi - Loss_opt_semi

Plots.scalefontsizes(1.5)

p = plot(1:T,Regret_meta_adv[1:T], label="Adversarial", ylabel = "Regret", xlabel="Time",legend=:topleft,  linewidth = 3, color=:dodgerblue)
plot!(1:T,Regret_meta_semi[1:T],label = "In Between",  linewidth = 3, color=:darkorchid3)
plot!(1:T,Regret_meta_stoch[1:T],label = "Stochastic",  linewidth = 3, color=:darkgoldenrod3)
# vline!([log(N)/(0.1)^2], label="log N / Δ²")
# hline!([log(N)/(0.1)], label="log N / Δ")

p

savefig(p, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/stoch-adv-semi-regret-T"*string(T)*"-N"*string(N)*".png")

Plots.scalefontsizes(1/1.5)


Regret_hedge_adv = Loss_hedge_adv - Loss_opt_adv
Regret_hedge_stoch = Loss_hedge_stoch - Loss_opt_stoch
Regret_hedge_semi = Loss_hedge_semi - Loss_opt_semi

Plots.scalefontsizes(1.5)

p2 = plot(1:T,Regret_hedge_adv[1:T], label="Adversarial", ylabel = "Regret", xlabel="Time",legend=:topleft,  linewidth = 3, color=:dodgerblue)
plot!(1:T,Regret_hedge_semi[1:T],label = "In Between",  linewidth = 3, color=:darkorchid3)
plot!(1:T,Regret_hedge_stoch[1:T],label = "Stochastic",  linewidth = 3, color=:darkgoldenrod3)
# vline!([log(N)/(0.1)^2], label="log N / Δ²")
# hline!([log(N)/(0.1)], label="log N / Δ")

p2

savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/bad-stoch-adv-semi-regret-T"*string(T)*"-N"*string(N)*".png")

Plots.scalefontsizes(1/1.5)
