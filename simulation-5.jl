using Plots
using Roots
using Distributions
using JLD2
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(6)

# N = 10^4
logN= 7
logN0d2 = 3
# N0 = 1


T = 1000000
T0 = 40000
draw_weightplot = false
draw_entplot = false

Ts = [1000:1000:9000;10000:10000:90000;100000:100000:900000;1000000:1000000:T]
# c = 1

c1 = sqrt(8)
# c2 = 1/8
c2 = 1


# c = c1*sqrt(log(N)+c2)
c = c1*sqrt(logN)
# c0 = c1*sqrt(c2)
c0 = c1*sqrt(c2)

cmeta = sqrt(8*log(2))

loss_hedge = zeros(T)
loss_hedge0 = zeros(T)
loss_care = zeros(T)
loss_metacare = zeros(T)

Loss_hedge = zeros(T)
Loss_hedge0 = zeros(T)
Loss_care = zeros(T)
Loss_metacare = zeros(T)

Loss_opt = zeros(T)

w_hedge = [zeros(2);1]
w_hedge0 = [zeros(2);1]
w_care =  [zeros(2);1]
w_meta_hedge =  [0.5,0.5]
w_metacare =  [zeros(2);1]



# entropy = w -> -(w[1:N0]' * log.(w[1:N0])) - ifelse(w[1+N0]==0,0, w[1+N0] * log(w[1+N0]) - w[1+N0]*log(N-N0))
# entropy = w -> -(w[1:N0]' * ifelse.(w[1:N0] .==0,0,log.(w[1:N0]))) - ifelse(w[1+N0]==0,0, w[1+N0] * log(w[1+N0]) - w[1+N0]*logN)
logNs = [logN0d2;logN0d2;logN]
entropy = w -> - sum(ifelse.(w .==0,0, w .*log.(w) .- w .* logNs)) # the last one should really be log(N-N0) not logN... we assume N~~N0^10 or more, so the difference is immaterial


# function expwt(lr,Loss)
#   wt_tot = sum(exp.(-lr*(Loss[1:N0] .-minimum(Loss[1:N0]))))+
#     (N-N0)*exp.(-lr*(Loss[1+N0] -minimum(Loss[1:N0])))
#   [exp.(-lr*(Loss[1:N0] .-minimum(Loss))) ; (N-N0)*exp.(-lr*(Loss[1+N0] .-minimum(Loss[1:N0])))]/wt_tot
# end

# function expwt(lr,Loss)
#   wt_tot = sum(exp.(-lr*(Loss[1:N0] .-minimum(Loss[1:N0]))))+
#     exp.(logN-lr*(Loss[1+N0] -minimum(Loss[1:N0])))
#   [exp.(-lr*(Loss[1:N0] .-minimum(Loss))) ; exp.(logN-lr*(Loss[1+N0] .-minimum(Loss[1:N0])))]/wt_tot
# end

function expwt(lr,Loss)
  wt_tot = sum(exp.(logNs .- lr*(Loss .- minimum(Loss))))
  exp.(logNs .- lr*(Loss .-minimum(Loss)))/wt_tot
end

# function implr(Loss,t)
#   expwtLoss = lr -> (lr-c1*sqrt((entropy(expwt(lr, Loss))+c2)/t)) 
#   lr_hat = find_zero(expwtLoss,(0,c1*sqrt((log(N)+c2)/t)))
#   lr_hat
# end
function implr(Loss,t)
  expwtLoss = lr -> (lr-c1*sqrt((entropy(expwt(lr, Loss))+c2)/t)) 
  lr_hat = find_zero(expwtLoss,(0,2*c1*sqrt((logN+c2)/t)))
  lr_hat
end

function impwt(Loss,t)
  expwt(implr(Loss,t),Loss)
end

# gain_dist = Multinomial(1, 1/N0 * [ones(N0); 0])
# losses = 1.0 .-rand(gain_dist)

# rand1 = rand(Bernoulli(0.5),N0)
losses = [1%2;2%2;0.6]
# losses = [rand.(Bernoulli.(min.(0.99,w_hedge0[1:N0])));0.8]

# losses = [rand.(Bernoulli.(min.(0.99,));0.8]

# which_idx = 1
# losses = losses = setindex!(ones(N0+1), zeros(N02), N02 .+ (1:N02) )

Losses = losses

loss_hedge[1] = losses' * w_hedge
Loss_hedge[1] = loss_hedge[1]

loss_hedge0[1] = losses' * w_hedge0
Loss_hedge0[1] = loss_hedge0[1]

loss_care[1] = losses' * w_care
Loss_care[1] = loss_care[1]

weight_meta_hedge = 0.5
Loss_metacare[1] = (1-weight_meta_hedge)*Loss_care[1]+weight_meta_hedge*Loss_hedge[1]


lr_hedge = zeros(T)
lr_hedge0 = zeros(T)
lr_care = zeros(T)
# lr_metacare = zeros(T)

entropy_hedge = zeros(T)
entropy_hedge0 = zeros(T)
entropy_care = zeros(T)
entropy_metacare = zeros(T)

norm_care = zeros(T)
norm_hedge0 = zeros(T)
norm_hedge = zeros(T)
norm_metacare = zeros(T)

for t in 2:T
  global w_hedge = expwt(c/sqrt(t), Losses)
  global w_hedge0 = expwt(c0/sqrt(t), Losses)
  global w_care = impwt(Losses,t)
  global w_meta_hedge = exp(-cmeta/sqrt(t)*Loss_hedge[t-1])/(exp(-cmeta/sqrt(t)*Loss_hedge[t-1])+exp(-cmeta/sqrt(t)*Loss_care[t-1]))
  global w_metacare = w_meta_hedge*w_hedge+(1-w_meta_hedge)*w_care

  # losses = 1.0 .-rand(gain_dist)
  # losses = [rand(Bernoulli(0.5),N0);1]
  losses = [t%2;(t+1)%2;0.6]
  # losses = [0.1,0.9]
  # rand0 = rand(Bernoulli(0.1))
  # rand1 = rand(Bernoulli(0.1),N0-2)
  # losses = [rand0;rand1;1-rand0;1]
  # losses = [rand.(Bernoulli.(min.(0.5,w_hedge[1:N0])));rand(Bernoulli(min(1.0,0.1+maximum(w_hedge))))]
  # losses = [rand.(Bernoulli.(min.(0.99,w_hedge0[1:N0])));0.51]
  # if t in Ts
  #   global which_idx=1-which_idx
  #   # global losses = 1*rand(loss_dist)
  #   global losses = setindex!(ones(N0+1), zeros(N02), which_idx*N02 .+ (1:N02) )
  # end
  # losses = setindex!(ones(N0+1), zeros(N02), (t%2)*N02 .+ (1:N02) )

  global loss_hedge[t] = losses' * w_hedge
  global Loss_hedge[t] = Loss_hedge[t-1] + loss_hedge[t]
  global lr_hedge[t]  = c/sqrt(t)
  global entropy_hedge[t]  = entropy(w_hedge)


  global loss_hedge0[t] = losses' * w_hedge0
  global Loss_hedge0[t] = Loss_hedge0[t-1] + loss_hedge0[t]
  global lr_hedge0[t]  = c0/sqrt(t)
  global entropy_hedge0[t]  = entropy(w_hedge0)


  global loss_care[t] = losses' * w_care
  global Loss_care[t] = Loss_care[t-1] + loss_care[t]
  global lr_care[t] = implr(Losses,t)
  global entropy_care[t]  = entropy(w_care)

  global loss_metacare[t] = losses' * w_metacare
  global Loss_metacare[t] = Loss_metacare[t-1] + loss_metacare[t]
  # global lr_metacare[t] =  cmeta/sqrt(t)
  global entropy_metacare[t]  = entropy(w_metacare)
  
  global Losses = Losses + losses
  global Loss_opt[t] = minimum(Losses)
  
  global norm_care[t] = sum(abs.(w_care .- [0.5;0.5;0]))
  global norm_hedge[t] = sum(abs.(w_hedge .- [0.5;0.5;0]))
  global norm_hedge0[t] = sum(abs.(w_hedge0 .- [0.5;0.5;0]))
  global norm_metacare[t] = sum(abs.(w_metacare .- [0.5;0.5;0]))


  if t in Ts
    print("t=T"*string(t)*"\n")
    global Regret_hedge = Loss_hedge - Loss_opt
    global Regret_hedge0 = Loss_hedge0 - Loss_opt
    global Regret_care = Loss_care - Loss_opt
    global Regret_metacare = Loss_metacare - Loss_opt

    
    p2 = plot(1:t,Regret_hedge[1:t], label="Hedge sqrtlog N", ylabel = "Regret", xlabel="Time",legend=:topleft,yaxis=:log, xaxis=:log, linecolor = :blue)
    plot!(1:t,Regret_hedge0[1:t],label = "Hedge 1", linecolor = :red)
    plot!(1:t,Regret_care[1:t],label = "CARE", linecolor = :green)
    plot!(1:t,Regret_metacare[1:t],label = "Meta-CARE", linecolor = :purple)
    # plot!(1:t,1:t,label = "T", linestyle = :dot)
    # plot!(1:t, sqrt.((1+logN0d2+log(2))*(1:t)),label = "sqrt(T (1+logN0))", linestyle = :dot)
    # plot!(1:t, (1+log(N0))*sqrt.(1:t),label = "sqrt(T) (1+logN0) ", linestyle = :dot)    
    # # plot!(1:t, sqrt.(log(N)*(1:t)),label = "sqrt(T logN )", linestyle = :dot)
    # plot!(1:t, sqrt.(logN*(1:t)),label = "sqrt(T logN )", linestyle = :dot)

    # plot!(1:T,Regret_hedge2)

    # p1
    # p2
    savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/regret-T"*string(T)*"-logN"*string(logN)*"-logN0"*string(logN0d2+log(2))*"-v2.pdf")
    
    # p3 = plot(2:t,lr_hedge[2:t] .* sqrt.(2:t), label="Hedge sqrtlog N", ylabel = "Learning Rate * sqrt(time)", xlabel="Time",legend=:right,fmt=:pdf)
    # plot!(2:t,lr_hedge0[2:t] .* sqrt.(2:t) ,label = "Hedge 1")
    # plot!(2:t,lr_care[2:t] .* sqrt.(2:t) ,label = "CARE")
    # p3
    # savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/lr-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")
    if ( draw_entplot)
      p3 = plot(2:t,entropy_hedge[2:t] , label="Hedge sqrtlog N", ylabel = "Entropy", xlabel="Time",legend=:topright,fmt=:pdf,yaxis=:log, xaxis=:log, xlim=(1,t), linecolor = :blue)
      plot!(2:t,entropy_hedge0[2:t] ,label = "Hedge 1", linecolor = :red)
      plot!(2:t,entropy_care[2:t] ,label = "CARE", linecolor = :green)
      plot!(2:t,entropy_metacare[2:t] ,label = "Meta-CARE", linecolor = :purple)
      plot!([log(N0)], seriestype="hline", label="log N0", linestyle = :dot)
      # plot!([log(N)], seriestype="hline", label="log N", linestyle = :dot)
      plot!([logN], seriestype="hline", label="log N", linestyle = :dot)
      savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/entropy-T"*string(T)*"-logN"*string(logN)*"-N0"*string(N0)*"-v2.pdf")

    end
    # p3
    # savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")
    
    if ( draw_weightplot && t >= T0 )
      p4 = plot(T0:t,norm_care[T0:t]/2 , label="Care", ylabel = "1-Norm of Weights from Uniform", xlabel="Time",legend=:bottomleft,fmt=:pdf, xaxis=:log, yaxis=:log, xlim=(1,t), linecolor = :green)
      plot!(T0:t,norm_hedge0[T0:t]/2 ,label = "Hedge 1", linecolor = :red)
      plot!(T0:t,norm_hedge[T0:t]/2 ,label = "Hedge sqrtlog N", linecolor = :blue, alpha = 0.2)
      plot!(T0:t,norm_metacare[T0:t]/2 ,label = "Meta-CARE", linecolor = :purple, alpha = 0.2)
      # p4
      savefig(p4, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/norms-T"*string(T)*"-logN"*string(logN)*"-N0"*string(N0)*"-v2.pdf")
    end
    # @save ("~/github/semi-adversarial/compiled-results/experiments/workspace-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*".jld")

  end
end

Regret_hedge = Loss_hedge - Loss_opt
Regret_hedge0 = Loss_hedge0 - Loss_opt
Regret_care = Loss_care - Loss_opt
t=T
p2 = plot(1:t,Regret_hedge[1:t], label="Hedge sqrtlog N", ylabel = "Regret", xlabel="Time",legend=:topleft,yaxis=:log, xaxis=:log, linecolor = :blue)
plot!(1:t,Regret_hedge0[1:t],label = "Hedge 1", linecolor = :red)
plot!(1:t,Regret_care[1:t],label = "CARE", linecolor = :green)
plot!(1:t,Regret_metacare[1:t],label = "Meta-CARE", linecolor = :purple)
# plot!(1:t,1:t,label = "T", linestyle = :dot)
# plot!(1:t, sqrt.((1+log(N0))*(1:t)),label = "sqrt(T (1+logN0))", linestyle = :dot)
# plot!(1:t, (1+log(N0))*sqrt.(1:t),label = "sqrt(T) (1+logN0) ", linestyle = :dot)    
# # plot!(1:t, sqrt.(log(N)*(1:t)),label = "sqrt(T logN )", linestyle = :dot)
# plot!(1:t, sqrt.(logN*(1:t)),label = "sqrt(T logN )", linestyle = :dot)

# plot!(1:T,Regret_hedge2)

# p1
# p2
savefig(p2, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/regret-T"*string(T)*"-logN"*string(logN)*"-N0"*string(N0)*"-v2.pdf")

# p3 = plot(2:t,lr_hedge[2:t] .* sqrt.(2:t), label="Hedge sqrtlog N", ylabel = "Learning Rate * sqrt(time)", xlabel="Time",legend=:right,fmt=:pdf)
# plot!(2:t,lr_hedge0[2:t] .* sqrt.(2:t) ,label = "Hedge 1")
# plot!(2:t,lr_care[2:t] .* sqrt.(2:t) ,label = "CARE")
# p3
# savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/lr-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

if draw_entplot & t >= T0 
  p3 = plot(2:t,entropy_hedge[2:t] , label="Hedge sqrtlog N", ylabel = "Entropy", xlabel="Time",legend=:topright,fmt=:pdf,yaxis=:log, xaxis=:log, xlim=(1,t), linecolor = :blue)
  plot!(2:t,entropy_hedge0[2:t] ,label = "Hedge 1", linecolor = :red)
  plot!(2:t,entropy_care[2:t] ,label = "CARE", linecolor = :green)
  plot!(2:t,entropy_metacare[2:t] ,label = "Meta-CARE", linecolor = :purple)
  plot!([log(N0)], seriestype="hline", label="log N0", linestyle = :dot)
  # plot!([log(N)], seriestype="hline", label="log N", linestyle = :dot)
  plot!([logN], seriestype="hline", label="log N", linestyle = :dot)
  savefig(p3, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/entropy-T"*string(T)*"-logN"*string(logN)*"-N0"*string(N0)*"-v2.pdf")

end
# p3
# savefig(p3, "~/github/semi-adversarial/compiled-results/experiments/entropy-T"*string(T)*"-N"*string(N)*"-N0"*string(N0)*"-v2.pdf")

if draw_weightplot & t >= T0 
  p4 = plot(T0:t,norm_care[T0:t]/2 , label="Care", ylabel = "1-Norm of Weights from Uniform", xlabel="Time",legend=:bottomleft,fmt=:pdf, xaxis=:log, yaxis=:log, xlim=(1,t), linecolor = :green)
  plot!(T0:t,norm_hedge0[T0:t]/2 ,label = "Hedge 1", linecolor = :red)
  plot!(T0:t,norm_hedge[T0:t]/2 ,label = "Hedge sqrtlog N", linecolor = :blue)
  plot!(T0:t,norm_metacare[T0:t]/2 ,label = "Meta-CARE", linecolor = :purple)
  # p4
  savefig(p4, "~/github/semi-adversarial/semi-adversarial-project-1/experiments/norms-T"*string(T)*"-logN"*string(logN)*"-N0"*string(N0)*"-v2.pdf")
end
