using Plots
using Roots
using Distributions
# LinearAlgebra.BLAS.set_num_threads(6)

N = 100
N0 = 1
N02= 1

T = 1000000
Ts = 2 .^(1:Int64(floor(log2(T))))

# c = 1

c1 = sqrt(8)
c2 = 0.001

c = c1*sqrt(log(N)+c2)
c0 = c1*sqrt(c2)

w_hedge = 1/N * ones(N)
w_hedge0 = 1/N * ones(N)
w_care =   1/N * ones(N)
w_normal =   1/N * ones(N)


loss_hedge = zeros(T)
loss_hedge0 = zeros(T)

# loss_hedge2 = zeros(T)
loss_care = zeros(T)
loss_normal = zeros(T)

Loss_hedge = zeros(T)
Loss_hedge0 = zeros(T)

# Loss_hedge2 = zeros(T)
Loss_care = zeros(T)
Loss_normal = zeros(T)

Loss_opt = zeros(T)

entropy = w -> -(w' * ifelse.(w.==0,0,log.(w)))

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

function normalwt(Loss,Loss_normal,t,N)
  R = max.(Loss_normal .- Loss,0)
  R2 = R.^2 ./2
  R2max = maximum(R2)
  R2cen = R2 .- R2max

  foo = c -> 1+log(N) - log(sum(exp.(R2cen./c))) - R2max/c
  c = find_zero(foo,1)

  wt = R ./ c .* exp.(R2 ./c)

  wt ./sum(wt)
end

# Adversarial Case
# gain_dist = Multinomial(1, 1/N0 * [ones(N0); zeros(N-N0)])
# which_dist = 1
# gain_dist = Multinomial(1, 1/N0 * [ones(N0); zeros(N-N0)])

# losses = 1.0 .-rand(gain_dist)
# losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.75),N-N0)]

# losses = [rand(Bernoulli(0.5),1);rand(Bernoulli(0.75),N-1)] #easy stochastic
# losses = [rand(Bernoulli(0.5),2);rand(Bernoulli(0.75),N-2)] #medium stochastic
# losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.55),N-N0)] #hard stochastic
losses = [(0.5*(1%2*ones(N02))); (0.5*((1+1)%2*ones(N02))) ;rand(Bernoulli(0.55),N-2*N02)]
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

loss_normal[1] = losses' * w_normal
Loss_normal[1] = loss_normal[1]

lr_hedge = zeros(T)
lr_hedge0 = zeros(T)
lr_care = zeros(T)

entropy_hedge = zeros(T)
entropy_hedge0 = zeros(T)
entropy_care = zeros(T)
entropy_normal = zeros(T)



for t in 2:T
  # losses = 1.0 .-rand(gain_dist)
  # losses = [rand(Bernoulli(0.5),N0);1]

  # losses = [rand(Bernoulli(0.5),N0);rand(Bernoulli(0.55),N-N0)] #hard stochastic
  losses = [(0.5*(t%2*ones(N02))); (0.5*((t+1)%2*ones(N02))) ;rand(Bernoulli(0.9),N-2*N02)]
  # if t in Ts
  #   which_idx=1-which_idx
  #   # losses = 1*rand(loss_dist)
  #   losses = setindex!(ones(N0+1), zeros(N02), which_idx*N02 .+ (1:N02) )
  # end
  # losses = setindex!(ones(N0+1), zeros(N02), (t%2)*N02 .+ (1:N02) )

  w_hedge = expwt(c/sqrt(t), Losses)
  loss_hedge[t] = losses' * w_hedge
  Loss_hedge[t] = Loss_hedge[t-1] + loss_hedge[t]
  lr_hedge[t]  = c/sqrt(t)
  entropy_hedge[t]  = entropy(w_hedge)


  w_hedge0 = expwt(c0/sqrt(t), Losses)
  loss_hedge0[t] = losses' * w_hedge0
  Loss_hedge0[t] = Loss_hedge0[t-1] + loss_hedge0[t]
  lr_hedge0[t]  = c0/sqrt(t)
  entropy_hedge0[t]  = entropy(w_hedge0)


  w_care = impwt(Losses,t)
  loss_care[t] = losses' * w_care
  Loss_care[t] = Loss_care[t-1] + loss_care[t]
  lr_care[t] = implr(Losses,t)
  entropy_care[t]  = entropy(w_care)

  w_normal = normalwt(Losses,Loss_normal[t-1],t,N)
  loss_normal[t] = losses' * w_normal
  Loss_normal[t] = Loss_normal[t-1] + loss_normal[t]
  entropy_normal[t]  = entropy(w_normal)


  Losses = Losses + losses
  Loss_opt[t] = minimum(Losses)
  if t % 10000==0
    print("t=T"*string(t)*"\n")
    Regret_hedge = Loss_hedge - Loss_opt
    Regret_hedge0 = Loss_hedge0 - Loss_opt
    Regret_care = Loss_care - Loss_opt
    Regret_normal = Loss_normal - Loss_opt

    p2 = plot(1:t,Regret_hedge[1:t], label="Hedge", ylabel = "Regret", xlabel="Time",legend=:bottomright)#,yaxis=:log, xaxis=:log)
    # plot!(1:t,Regret_hedge0[1:t],label = "Hedge 1")
    plot!(1:t,Regret_care[1:t],label = "CARE")
    plot!(1:t,Regret_normal[1:t],label = "NormalHedge")
    # plot!(1:T,Regret_hedge2)

    # p1
    p2
    savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/regret-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

    # p3 = plot(2:t,lr_hedge[2:t] .* sqrt.(2:t), label="Hedge sqrtlog N", ylabel = "Learning Rate * sqrt(time)", xlabel="Time",legend=:right,fmt=:pdf)
    # plot!(2:t,lr_hedge0[2:t] .* sqrt.(2:t) ,label = "Hedge 1")
    # plot!(2:t,lr_care[2:t] .* sqrt.(2:t) ,label = "CARE")
    # p3
    # savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/lr-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

    p3 = plot(2:t,entropy_hedge[2:t], label="Hedge", ylabel = "Entropy", xlabel="Time",legend=:right,fmt=:pdf)
    # plot!(2:t,entropy_hedge0[2:t],label = "Hedge 1")
    plot!(2:t,entropy_care[2:t] ,label = "CARE")
    plot!(2:t,entropy_normal[2:t] ,label = "NormalHedge")
    p3
    savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

  end
end

Regret_hedge = Loss_hedge - Loss_opt
Regret_hedge0 = Loss_hedge0 - Loss_opt
Regret_care = Loss_care - Loss_opt
Regret_normal = Loss_normal - Loss_opt


p2 = plot(1:T,Regret_hedge[1:T], label="Hedge", ylabel = "Regret", xlabel="Time",legend=:bottomright)
# plot!(1:T,Regret_hedge0[1:T],label = "Hedge 1")
plot!(1:T,Regret_care[1:T],label = "CARE")
plot!(1:T,Regret_normal[1:T],label = "NormalHedge")
# plot!(1:T,Regret_hedge2)

# p1
p2
savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/regret-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.png")

p3 = plot(2:T,entropy_hedge[2:T], label="Hedge", ylabel = "Entropy ", xlabel="Time",legend=:topright, symbol=:square)
# plot!(2:T,entropy_hedge0[2:T]  ,label = "Hedge 1")
plot!(2:T,entropy_care[2:T] ,label = "CARE")
# plot!([log(N0)], seriestype="hline", label="log N0")
# plot!([log(N)], seriestype="hline", label="log N")
plot!(2:T,entropy_normal[2:T],label = "NormalHedge")

p3
savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.png")

0
