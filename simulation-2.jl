using Plots
using Roots
using Distributions
# LinearAlgebra.BLAS.set_num_threads(6)

N = 1000
N0 = 2
N02= 2

T = 1000000
Ts = 2 .^(1:Int64(floor(log2(T))))

# c = 1

c1 = sqrt(8)
c2 = 1/8

c = c1*sqrt(log(N)+c2)
c0 = c1*sqrt(c2)

w_hedge = 1/N * ones(N)
w_hedge0 = 1/N * ones(N)
w_care =   1/N * ones(N)

loss_hedge = zeros(T)
loss_hedge0 = zeros(T)

# loss_hedge2 = zeros(T)
loss_care = zeros(T)

Loss_hedge = zeros(T)
Loss_hedge0 = zeros(T)

# Loss_hedge2 = zeros(T)
Loss_care = zeros(T)

Loss_opt = zeros(T)

function entropy(w) 
  -(w' * ifelse.(w.==0,0,log.(w)))
end

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
# gain_dist = Multinomial(1, 1/N0 * [ones(N0); zeros(N-N0)])
# which_dist = 1
# gain_dist = Multinomial(1, 1/N0 * [ones(N0); zeros(N-N0)])

# losses = 1.0 .-rand(gain_dist)
# losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.75),N-N0)]

# losses = [rand(Bernoulli(0.5),1);rand(Bernoulli(0.75),N-1)] #easy stochastic
# losses = [rand(Bernoulli(0.5),2);rand(Bernoulli(0.75),N-2)] #medium stochastic
losses = [rand(Bernoulli(0.1),N0);rand(Bernoulli(0.9),N-N0)] #hard stochastic

# which_idx = 1

# losses = setindex!(ones(N), zeros(N02), which_idx*N02 .+ (1:N02) )
# losses[(N0+1):(N0+10)] .= 0.6
Losses = losses

loss_hedge[1] = losses' * w_hedge
Loss_hedge[1] = loss_hedge[1]

loss_hedge0[1] = losses' * w_hedge0
Loss_hedge0[1] = loss_hedge0[1]
# loss_hedge2[1] = losses' * w_hedge2
# Loss_hedge2[1] = loss_hedge2[1]

loss_care[1] = losses' * w_care
Loss_care[1] = loss_care[1]

lr_hedge = zeros(T)
lr_hedge0 = zeros(T)
lr_care = zeros(T)

entropy_hedge = zeros(T)
entropy_hedge0 = zeros(T)
entropy_care = zeros(T)

entropy_hedge = zeros(T)
entropy_hedge0 = zeros(T)
entropy_care = zeros(T)


for t in 2:T
  # losses = 1.0 .-rand(gain_dist)
  # losses = [rand(Bernoulli(0.5),N0);1]

  losses = [rand(Bernoulli(0.1),N0);rand(Bernoulli(0.99),N-N0)]

  # if t in Ts
  #   global which_idx=1-which_idx
  #   # global losses = 1*rand(loss_dist)
  #   global losses = setindex!(ones(N0+1), zeros(N02), which_idx*N02 .+ (1:N02) )
  # end
  # losses = setindex!(ones(N0+1), zeros(N02), (t%2)*N02 .+ (1:N02) )

  global w_hedge = expwt(c/sqrt(t), Losses)
  global loss_hedge[t] = losses' * w_hedge
  global Loss_hedge[t] = Loss_hedge[t-1] + loss_hedge[t]
  global lr_hedge[t]  = c/sqrt(t)
  global entropy_hedge[t]  = entropy(w_hedge)


  global w_hedge0 = expwt(c0/sqrt(t), Losses)
  global loss_hedge0[t] = losses' * w_hedge0
  global Loss_hedge0[t] = Loss_hedge0[t-1] + loss_hedge0[t]
  global lr_hedge0[t]  = c0/sqrt(t)
  global entropy_hedge0[t]  = entropy(w_hedge0)


  global w_care = impwt(Losses,t)
  global loss_care[t] = losses' * w_care
  global Loss_care[t] = Loss_care[t-1] + loss_care[t]
  global lr_care[t] = implr(Losses,t)
  global entropy_care[t]  = entropy(w_care)


  global Losses = Losses + losses
  global Loss_opt[t] = minimum(Losses)
  if t % 10000==0
    print("t=T"*string(t)*"\n")
    global Regret_hedge = Loss_hedge - Loss_opt
    global Regret_hedge0 = Loss_hedge0 - Loss_opt
    global Regret_care = Loss_care - Loss_opt

    p2 = plot(1:t,Regret_hedge[1:t], label="Hedge sqrtlog N", ylabel = "Regret", xlabel="Time",legend=:bottomright)#,yaxis=:log, xaxis=:log)
    plot!(1:t,Regret_hedge0[1:t],label = "Hedge 1")
    plot!(1:t,Regret_care[1:t],label = "CARE")
    # plot!(1:T,Regret_hedge2)

    # p1
    p2
    savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/regret-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

    # p3 = plot(2:t,lr_hedge[2:t] .* sqrt.(2:t), label="Hedge sqrtlog N", ylabel = "Learning Rate * sqrt(time)", xlabel="Time",legend=:right,fmt=:pdf)
    # plot!(2:t,lr_hedge0[2:t] .* sqrt.(2:t) ,label = "Hedge 1")
    # plot!(2:t,lr_care[2:t] .* sqrt.(2:t) ,label = "CARE")
    # p3
    # savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/lr-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

    p3 = plot(2:t,entropy_hedge[2:t], label="Hedge sqrtlog N", ylabel = "Entropy", xlabel="Time",legend=:right,fmt=:pdf,yaxis=:log)
    plot!(2:t,entropy_hedge0[2:t],label = "Hedge 1")
    plot!(2:t,entropy_care[2:t] ,label = "CARE")
    p3
    savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

  end
end

Regret_hedge = Loss_hedge - Loss_opt
Regret_hedge0 = Loss_hedge0 - Loss_opt
Regret_care = Loss_care - Loss_opt

p2 = plot(1:T,Regret_hedge[1:T], label="Hedge sqrtlog N", ylabel = "Regret", xlabel="Time",legend=:bottomright,yaxis=:log, xaxis=:log)
plot!(1:T,Regret_hedge0[1:T],label = "Hedge 1")
plot!(1:T,Regret_care[1:T],label = "CARE")
# plot!(1:T,Regret_hedge2)

# p1
p2
savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/regret-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.png")

p3 = plot(2:T,entropy_hedge[2:T], label="Hedge sqrtlog N", ylabel = "Entropy ", xlabel="Time",legend=:right,yaxis=:log)
plot!(2:T,entropy_hedge0[2:T]  ,label = "Hedge 1")
plot!(2:T,entropy_care[2:T] ,label = "CARE")
plot!([log(N0)], seriestype="hline", label="log N0")
plot!([log(N)], seriestype="hline", label="log N")

p3
savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.png")
