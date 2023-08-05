using Plots
using Roots
using Distributions

N = 100
N0 = 10
N02 = 4

T = 10000
Ts = 2 .^(1:log2(T))

c = sqrt(8*(log(N)+1/8))

c1 = sqrt(8)
c2 = 1/8

w_hedge = 1/N * ones(N)

w_care =   1/N * ones(N)

loss_hedge = zeros(T)
# loss_hedge2 = zeros(T)
loss_care = zeros(T)

Loss_hedge = zeros(T)
# Loss_hedge2 = zeros(T)
Loss_care = zeros(T)

Loss_opt = zeros(T)

entropy = w -> -(w' * log.(w))

function expwt(lr,Loss)
  exp.(-lr*(Loss .-minimum(Loss)))/sum(exp.(-lr*(Loss .-minimum(Loss))))
end

function implr(Loss,t)
  expwtLoss = lr -> (lr-c1*sqrt((entropy(expwt(lr, Loss))+c2)/t)) 
  lr_hat = find_zero(expwtLoss,(0,c1*sqrt((log(N)+c2)/t)))
  lr_hat
end

function impwt(Loss,t)
  expwt(implr(Loss,t),Loss)
end

# Adversarial Case
# gain_dist = Multinomial(1, 1/N * ones(N))
# 
# losses = 1.0 .-rand(gain_dist)
# Losses = losses
# 
# loss_hedge[1] = losses' * w_hedge
# Loss_hedge[1] = loss_hedge[1]
# 
# loss_care[1] = losses' * w_care
# Loss_care[1] = loss_care[1]
# 
# for t in 2:T
#   losses = 1.0 .-rand(gain_dist)
# 
#   w_hedge = expwt(c/sqrt(t), Losses)
# 
#   loss_hedge[t] = losses' * w_hedge
#   Loss_hedge[t] = Loss_hedge[t-1] + loss_hedge[t]
# 
#   w_care = impwt(Losses,t)
#   loss_care[t] = losses' * w_care
#   Loss_care[t] = Loss_care[t-1] + loss_care[t]
# 
#   global Losses = Losses + losses
#   Loss_opt[t] = minimum(Losses)
# end
# 
# Regret_hedge = Loss_hedge - Loss_opt
# Regret_care = Loss_care - Loss_opt
# 
# p1 = plot(1:T,Regret_hedge)
# plot!(1:T,Regret_care)

# Semi-Adversarial Case
# gain_dist = Multinomial(1, 1/N0 * [ones(N0); zeros(N-N0)])
gain_dist = Multinomial(N02, 1/(N0+0.0*(N-N0)) * [ones(N0); 0.0*ones(N-N0)])
# loss_dist = Multinomial(1, 1/(N0*0.0+(N-N0)) * [0.0*ones(N0); ones(N-N0)])
losses = 1.0 .- 1/1*rand(gain_dist)
# losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.9),N-N0)]

# losses = 1/1*rand(loss_dist)
# losses[(N0+1):(N0+10)] .= 0.6
Losses = losses

loss_hedge[1] = losses' * w_hedge
Loss_hedge[1] = loss_hedge[1]

# loss_hedge2[1] = losses' * w_hedge2
# Loss_hedge2[1] = loss_hedge2[1]

loss_care[1] = losses' * w_care
Loss_care[1] = loss_care[1]

lr_hedge = zeros(T)
lr_care = zeros(T)

for t in 2:(T)
  # if t in Ts
  #   which_idx=1-which_idx
  #   # global losses = 1*rand(loss_dist)
  #   global losses == setindex!(zeros(N), ones(N02), (t-1) % N0 + 1)
  # end
  losses = 1.0 .- 1/1*rand(gain_dist)

  # losses[(N0+1):(N0+10)] .= 0.6

  # losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.75),N-N0)]
  w_hedge = expwt(c/sqrt(t), Losses)
  lr_hedge[t]  =c/sqrt(t)
  
  
  # w_hedge2 = expwt(1/sqrt(t), Losses)

  
  loss_hedge[t] = losses' * w_hedge
  # loss_hedge2[t] = losses' * w_hedge2

  Loss_hedge[t] = Loss_hedge[t-1] + loss_hedge[t]
  # Loss_hedge2[t] = Loss_hedge2[t-1] + loss_hedge2[t]

  w_care = impwt(Losses,t)
  lr_care[t] = implr(Losses,t)
  
  loss_care[t] = losses' * w_care
  Loss_care[t] = Loss_care[t-1] + loss_care[t]
  
  global Losses = Losses + losses
  Loss_opt[t] = minimum(Losses)
end


Regret_hedge = Loss_hedge - Loss_opt
# Regret_hedge2 = Loss_hedge2 - Loss_opt
Regret_care = Loss_care - Loss_opt

p2 = plot(1:T,Regret_hedge, label="Hedge", ylabel = "Learning Rate", xlabel="Time",legend=:bottomright)
# plot!(1:T,Regret_hedge2)
plot!(1:T,Regret_care,label = "CARE")

# p1
p2


# p3 = plot(2:T,lr_hedge[2:T] .* sqrt.(2:T), label="Hedge", ylabel = "Learning Rate * sqrt(time)", xlabel="Time",legend=:bottomright)
# plot!(2:T,lr_care[2:T] .* sqrt.(2:T) ,label = "CARE")
# p3
