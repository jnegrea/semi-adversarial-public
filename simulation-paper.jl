using Plots
using Roots
using Distributions
using StatsBase
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(6)

# pyplot()
gr()
theme(:wong2)

## OUTPUT PATH
# JN
output_path = "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/paper"
# BB
# output_path = "/Users/blairbilodeau/Documents/Research/Projects/Semi Adversarial Learning/semi-adversarial/semi-adversarial-project-1/experiments/figures/test"

## Key Parameters
# c_hedge = sqrt(8)
c_hedge = sqrt(2/3)
c_meta = sqrt(8*log(2))
# c_meta = 100

c_care1 = sqrt(2/3)
c_care2 = 1
## Helper Functions

# generate bernoullis
# by generating uniform and indicator with probvec
function loss_generator(N,T,scenario,parms,nsims)
  losses = zeros(T,N,nsims)
  if scenario == "gap"
    # for "gap", parms[1]=Δ is the gap
    N0 = parms[1]
    Δ = parms[2]
    noise = rand(T,N,nsims)
    losses[:,1:N0,:] = noise[:,1:N0,:] .<= 1/2-Δ/2
    losses[:,(N0+1):N,:] = noise[:,(N0+1):N,:] .<= 1/2+Δ/2
  elseif scenario == "alternating"
    # for "alternating", parms is a tuple containing
    #   parms[1] = N0/2
    #   parms[2] = Δ
    N02 = parms[1]
    Δ = parms[2]
    if N02>0
      # losses[:,1:N02,:] .= 1/2 .- Δ .+ Δ/2 * ((1:T) .%2)
      # losses[:,(N02+1):(2*N02),:] .= 1/2 .- Δ .+ Δ/2 * (1 .- (1:T) .%2)
      losses[:,1:N02,:] .= (1-Δ) * ((1:T) .%2)
      losses[:,(N02+1):(2*N02),:] .= (1-Δ) * (1 .- (1:T) .%2)
    end
    if 2*N02<N
      # noise = rand(T,N-2*N02,nsims)
      # losses[:,(2*N02+1):N,:] .= noise .<= 1/2 + Δ/2
      losses[:,(2*N02+1):N,:] .= 1
    end
  end
  losses
end

# function entropy(w)
#   -(w' * ifelse.(w.==0,0,log.(w)))
# end

function hedgewt(lr,Loss)
  exp.(-lr*(Loss .-minimum(Loss)))/sum(exp.(-lr*(Loss .-minimum(Loss))))
end

function implr(Loss,N)
  expwtLoss = lr -> (lr-c_care1*sqrt((entropy(hedgewt(lr, Loss))+c_care2)))
  lr_hat = find_zero(expwtLoss,(0,2*c_care1*sqrt((log(N)+c_care2))+1))
  lr_hat
end

function carewt(Loss,N)
  hedgewt(implr(Loss,N),Loss)
end

# function normalwt(Loss,Loss_normal,t,N)
#   R = max.(Loss_normal .- Loss,0)
#   R2 = R.^2 ./2
#   R2max = maximum(R2)
#   R2cen = R2 .- R2max
#
#   foo = c -> 1+log(N) - log(sum(exp.(R2cen./c))) - R2max/c
#   c = find_zero(foo,1)
#
#   wt = R ./ c .* exp.(R2 ./c)
#
#   wt ./sum(wt)
# end

function weight_policy(Losses,losses, T, N,nsims)
    # can compute all the weights for all times by broadcasting expwt to apply to each row of Lossmatrix
    # after scaling the cumulative losses by the 1/sqrt(t) pointwise
  shiftLosses = [zeros(1,N,nsims); Losses[1:(T-1),:,:]]
  weights_hedge = mapslices(L -> hedgewt(c_hedge*sqrt(log(N)),L), shiftLosses./sqrt.(1:T),dims=2)
  # Losses of an alg are rowise dot products
  Losses_hedge = cumsum(sum(weights_hedge .* losses, dims=2),dims=1)
  shiftLosses_hedge = [zeros(1,1,nsims); Losses_hedge[2:T,:,:]]
  # can compute all the weights for all times by broadcasting carewt to apply to each row of Lossmatrix
  # after scaling the cumulative losses by the 1/sqrt(t) pointwise
  weights_care = mapslices(L -> carewt(L,N), shiftLosses./sqrt.(1:T),dims=2)
  # Losses of an alg are rowise dot products
  Losses_care = cumsum(sum(weights_care .* losses, dims=2),dims=1)
  shiftLosses_care = [zeros(1,1,nsims); Losses_care[2:T,:,:]]

  # can compute all the weights for all times by computing the meta weight for each time
  # and pointwise metaweighting
  w_meta_hedge = exp.(-c_meta ./sqrt.(1:T) .* (shiftLosses_hedge-shiftLosses_care)) ./ (exp.(-c_meta ./sqrt.(1:T) .*(shiftLosses_hedge-shiftLosses_care)) .+ 1)
  weights_meta = w_meta_hedge .* weights_hedge + (1 .-w_meta_hedge) .* weights_care
  # Losses of an alg are rowise dot products
  Losses_meta = cumsum(sum(weights_meta .* losses, dims=2),dims=1)

return (weights_hedge, weights_care, weights_meta, Losses_hedge, Losses_care, Losses_meta)
end

## Core Simulator

function sim_game(N, T, scenario, parms,nsims)
  # generate all losses
  lossvecs = loss_generator(N,T,scenario,parms,nsims)

  # initialize all vectors
  Lossvecs = cumsum(lossvecs,dims=1)

  (weights_hedge, weights_care, weights_meta, Losses_hedge, Losses_care, Losses_meta) = weight_policy(Lossvecs, lossvecs, T, N,nsims)

  LossOpt = minimum(Lossvecs,dims=2)

  Regret_hedge = [mean(Losses_hedge - LossOpt,dims=3)...]
  Regret_care = [mean(Losses_care - LossOpt,dims=3)...]
  Regret_meta = [mean(Losses_meta - LossOpt,dims=3)...]

  entropy_hedge = [mean(mapslices(entropy,weights_hedge,dims=2),dims=3)...]
  entropy_care = [mean(mapslices(entropy,weights_care,dims=2),dims=3)...]
  entropy_meta = [mean(mapslices(entropy,weights_meta,dims=2),dims=3)...]

  return (entropy_hedge, entropy_care, entropy_meta, Regret_hedge, Regret_care, Regret_meta)
end

function plot_game(N,T, scenario, parms, nsims)

  logT = floor(Int8, log10(T))
  if scenario == "gap" && parms[2]>0 && parms[1]==1
    title = ("logT="*string(logT)*", N="*string(N)*", Stochastic-w-Gap="*string(parms[2]))
  elseif scenario == "gap" && parms[2]>0 && parms[1]>1
    title = ("logT="*string(logT)*", N="*string(N)*", "*string(parms[1])*"-EE-Stochastic-w-Gap="*string(parms[2]))
  elseif scenario == "gap" && parms[2]==0
    title = ("logT="*string(logT)*", N="*string(N)*", Stochastic-NoGap")
  elseif scenario == "alternating" && 0<2*parms[1]<N
    title = ("logT="*string(logT)*", N="*string(N)*", "*string(2*parms[1])*"-EffectiveExperts")
  elseif scenario == "alternating" && 2*parms[1]==N
    title = ("logT="*string(logT)*", N="*string(N)*", AdversarialAlternating")
  elseif scenario == "alternating" && parms[1]==0
    title = ("logT="*string(logT)*", N="*string(N)*", Stochastic-NoGap")
  end

  filename = replace(title, ", "=>"--" ) *"--c_meta="*string(round(c_meta; sigdigits=2))
  pngfile = output_path*"/png/regret_"*filename*".png"
  pdffile = output_path*"/pdf/regret_"*filename*".pdf"

  if isfile(pngfile)
    print("File: "*pngfile*" already exists")
  elseif isfile(pdffile)
    print("File: "*pdffile*" already exists")
  else
    (entropy_hedge, entropy_care, entropy_meta, Regret_hedge, Regret_care, Regret_meta) = sim_game(N, T, scenario, parms,nsims)

    Plots.scalefontsizes(1.5)
    # Ts = [1 .+ 10 .^(1:Int(log10(T))), T-1]
    eq_spaced_odd_log10 = Int.(2*floor.(10* 10 .^(0:0.1:0.9) ./2) .+1)
    Ts = vcat([1;3;5;7;9], eq_spaced_odd_log10 ,eq_spaced_odd_log10*(10 .^(1:(Int(log10(T))-2)))' .+1..., T-1)
    # maxy = 10^(ceil(2*log10(max(maximum(Regret_hedge),maximum(Regret_care),maximum(Regret_meta))))/2)
    p_regret = plot(Ts,Regret_hedge[Ts], label="Hedge", ylabel = "Regret", xlabel="Time",
    legend=:topleft, linestyle = :dot, thickness_scaling=1, legendfontsize=14,
    dpi = 150,
    linewidth =5,yaxis=:log, xaxis=:log, ylims=(10^(-.5),10^2.5))
    #,yaxis=:log, xaxis=:log)
    plot!(Ts,Regret_care[Ts],label = "FTRL-CARE", linestyle = :dash, linewidth =4)
    plot!(Ts,Regret_meta[Ts],label = "Meta-CARE", linestyle = :solid, linewidth =2)
    savefig(p_regret, pngfile)
    savefig(p_regret, pdffile)
    # p_entropy = plot(1:T,entropy_hedge, label="Hedge", ylabel = "Entropy", xlabel="Time", legend=:bottomright, title=title)
    # #,yaxis=:log, xaxis=:log)
    # plot!(p_entropy,1:T,entropy_care,label = "FTRL-CARE")
    # plot!(p_entropy,1:T,entropy_meta,label = "MetaCare")
    # savefig(p_entropy, output_path*"/entropy"*filename*".pdf")
    Plots.scalefontsizes(1/1.5)
    print(title*"\n")
  end
end

T=10000
theme(:wong2)
# plot_game(100,1000,"gap",(2,0.5),10)

# Alternating Cases
Ns = [4;16;256]
N02s = [1; 0]
scenario = "alternating"
for i in 1:length(N02s)
  for j in 1:length(Ns)
    N02 = ifelse(N02s[i]==0,Ns[j]÷2,N02s[i])
    if 2*N02s[i]<=Ns[j]
      plot_game(Ns[j], T, scenario, (N02 , 1/2), 1)
    end
  end
end

gaps = [0; 1/2]
Ns = [16]
N0s = [1]
# Stochastic Cases
# scenario = "gap"
# for i in 1:length(gaps)
#   for j in 1:length(Ns)
#     for k in 1:length(N0s)
#       N0 = ifelse(N0s[k]==0,Ns[j]÷2,N0s[k])
#       if N0<=Ns[j] && (gaps[i]!=0 || k==1)
#         if gaps[i]>0
#           plot_game(Ns[j], T, scenario, (N0,gaps[i]), 10)
#         else
#           plot_game(Ns[j], T, scenario, (N0,gaps[i]), 10)
#         end
#       end
#     end
#   end
# end
