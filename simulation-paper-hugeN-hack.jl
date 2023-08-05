using Plots
using Roots
using Distributions
using StatsBase
# using LinearAlgebra
# LinearAlgebra.BLAS.set_num_threads(6)

gr()
theme(:wong2)

## OUTPUT PATH
# JN
output_path = "~/github/semi-adversarial/semi-adversarial-project-1/experiments/figures/paper"
# BB
# output_path = "/Users/blairbilodeau/Documents/Research/Projects/Semi Adversarial Learning/semi-adversarial/semi-adversarial-project-1/experiments/figures/paper"

## Key Parameters
c_hedge = sqrt(8)
c_meta = sqrt(8*log(2))
c_meta = 100

# c_care1 = sqrt(2/3)
c_care1 = sqrt(8)
c_care2 = 1
## Helper Functions

# generate bernoullis
# by generating uniform and indicator with probvec
function loss_generator(T,scenario,parms,nsims)
  if scenario == "gap"
    # for "gap", parms[1]=Δ is the gap
    N0 = parms[1]
    losses = zeros(T,N0+1,nsims)

    Δ = parms[2]
    noise = rand(T,N0+1,nsims)
    losses[:,1:N0,:] = noise[:,1:N0,:] .<= 1 - Δ
    # losses[:,2,:] = noise[:,2,:] .<= 1/2+Δ/2
    losses[:,N0+1,:] .= 1
  elseif scenario == "alternating"
    # for "alternating", parms is a tuple containing
    #   parms[1] = N0/2
    #   parms[2] = Δ
    N02 = parms[1]
    Δ = parms[2]
    losses = zeros(T,2*N02+1,nsims)
    losses[:,1:N02,:] .= 2*(1-Δ) * ((1:T) .%2)
    losses[:,(N02+1):(2*N02),:] .= 2*(1-Δ) * (1 .- (1:T) .%2)
    # noise = rand(T,1,nsims)
    # losses[:,(2*N02+1),:] = noise .<= 3/4 + Δ/4
    losses[:,(2*N02+1),:] .= 1
  elseif scenario == "link-gap"
    # for "alternating", parms is a tuple containing
    #   parms[1] = N0/2
    #   parms[2] = Δ
    N0 = 2
    losses = zeros(T,N0+1,nsims)
    Δ = parms[2]
    noise = rand(T,N0+1,nsims)
    losses[:,1,:] = noise[:,1,:] .<= 1/2
    losses[:,2,:] = 1 .-losses[:,1,:]
    # losses[:,2,:] = noise[:,2,:] .<= 1/2+Δ/2
    losses[:,3,:] .= 1/2 .+ Δ
  end
  losses
end

# function entropy(w)
#   -(w' * ifelse.(w.==0,0,log.(w)))
# end

function entropy_hack(w,N0,logN)
  -(w[1:N0]' * ifelse.(w[1:N0] .==0,0,log.(w[1:N0]))) - ifelse(w[1+N0]==0,0, w[1+N0] * log(w[1+N0]) - w[1+N0]*logN)
end


function hedgewt(lr,Loss,N0,logN)
  wt_tot = sum(exp.(-lr*(Loss[1:N0] .-minimum(Loss[1:N0]))))+
    exp.(logN-lr*(Loss[1+N0] -minimum(Loss[1:N0])))
  [exp.(-lr*(Loss[1:N0] .-minimum(Loss))) ; exp.(logN-lr*(Loss[1+N0] .-minimum(Loss[1:N0])))]/wt_tot
end

function implr(Loss,N0,logN)
  expwtLoss = lr -> (lr-c_care1*sqrt((entropy_hack(hedgewt(lr, Loss,N0,logN),N0,logN)+c_care2)))
  lr_hat = find_zero(expwtLoss,(0,c_care1*sqrt((logN+c_care2))+1))
  lr_hat
end

function carewt(Loss,N0,logN)
  hedgewt(implr(Loss,N0,logN),Loss,N0,logN)
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

function weight_policy(Losses,losses, T, N0, logN,nsims)
  N = N0+1
  weights_hedge = zeros(T,N,nsims)
  weights_care = zeros(T,N,nsims)
  weights_meta = zeros(T,N,nsims)

  Losses_hedge = zeros(T,1,nsims)
  Losses_care = zeros(T,1,nsims)
  Losses_meta = zeros(T,1,nsims)

  Threads.@threads for sim in 1:nsims
    # can compute all the weights for all times by broadcasting expwt to apply to each row of Lossmatrix
    # after scaling the cumulative losses by the 1/sqrt(t) pointwise
    shiftLosses = [zeros(1,N,1); Losses[1:(T-1),:,sim]]
    weights_hedge[:,:,sim] = mapslices(L -> hedgewt(c_hedge*sqrt(logN),L,N0,logN), shiftLosses./sqrt.(1:T),dims=2)
    # Losses of an alg are rowise dot products
    Losses_hedge[:,:,sim]  = cumsum(sum(weights_hedge[:,:,sim] .* losses[:,:,sim], dims=2),dims=1)
    shiftLosses_hedge = [zeros(1,1,1); Losses_hedge[2:T,:,sim]]
    # can compute all the weights for all times by broadcasting carewt to apply to each row of Lossmatrix
    # after scaling the cumulative losses by the 1/sqrt(t) pointwise
    weights_care[:,:,sim] = mapslices(L -> carewt(L,N0,logN), shiftLosses./sqrt.(1:T),dims=2)
    # Losses of an alg are rowise dot products
    Losses_care[:,:,sim] = cumsum(sum(weights_care[:,:,sim] .* losses[:,:,sim], dims=2),dims=1)
    shiftLosses_care = [zeros(1,1,1); Losses_care[2:T,:,sim]]

    # can compute all the weights for all times by computing the meta weight for each time
    # and pointwise metaweighting
    w_meta_hedge = exp.(-c_meta ./sqrt.(1:T) .* (shiftLosses_hedge-shiftLosses_care)) ./ (exp.(-c_meta ./sqrt.(1:T) .*(shiftLosses_hedge-shiftLosses_care)) .+ 1)
    weights_meta[:,:,sim] = w_meta_hedge .* weights_hedge[:,:,sim] + (1 .-w_meta_hedge) .* weights_care[:,:,sim]
    # Losses of an alg are rowise dot products
    Losses_meta[:,:,sim] = cumsum(sum(weights_meta[:,:,sim] .* losses[:,:,sim], dims=2),dims=1)
  end
  LossOpt = minimum(Losses,dims=2)

  Regret_hedge = [mean(Losses_hedge - LossOpt,dims=3)...]
  Regret_care = [mean(Losses_care - LossOpt,dims=3)...]
  Regret_meta = [mean(Losses_meta - LossOpt,dims=3)...]

  entropy_hedge = [mean(mapslices(entropy,weights_hedge,dims=2),dims=3)...]
  entropy_care = [mean(mapslices(entropy,weights_care,dims=2),dims=3)...]
  entropy_meta = [mean(mapslices(entropy,weights_meta,dims=2),dims=3)...]
return (entropy_hedge, entropy_care, entropy_meta, Regret_hedge, Regret_care, Regret_meta)
end

## Core Simulator

function sim_game(logN, T, scenario, parms,nsims)
  if scenario == "gap"
    N0=parms[1]
    N=N0+1
  elseif scenario == "alternating"
    N02=parms[1]
    N0=2*N02
    N=N0+1
  elseif scenario == "link-gap"
    N0=2
    N=N0+1
  end
  # generate all losses
  lossvecs = loss_generator(T,scenario,parms,nsims)

  # initialize all vectors
  Lossvecs = cumsum(lossvecs,dims=1)

  return weight_policy(Lossvecs,lossvecs, T, N0, logN,nsims)
end

function plot_game(log2N, T, scenario, parms,nsims)

  logT = floor(Int8, log10(T))
  if scenario=="gap" && parms[1]==1
    title = ("logT="*string(logT)*", log2N="*string(log2N)*", Stochastic-w-Gap="*string(parms[2]))
  elseif scenario=="gap" && parms[1]>1
    title = ("logT="*string(logT)*", log2N="*string(log2N)*", "*string(parms[1])*"-EE-Stochastic-w-Gap="*string(parms[2]))
  elseif scenario=="alternating"
    title = ("logT="*string(logT)*", log2N="*string(log2N)*", Alternating, N0="*string(2*parms[1]))
  elseif scenario=="link-gap"
    title = ("logT="*string(logT)*", log2N="*string(log2N)*", 2-EE-Coupled-Stochastic-w-Gap="*string(parms[2]))
  end

  filename = replace(title, ", "=>"--" ) *"--c_meta="*string(round(c_meta; sigdigits=2))*"--c_care1="*string(round(c_care1; sigdigits=2))
  pngfile = output_path*"/huge/png/regret_"*filename*".png"
  pdffile = output_path*"/huge/pdf/regret_"*filename*".pdf"

  if isfile(pngfile)
    print("File: "*pngfile*" already exists")
  elseif isfile(pdffile)
    print("File: "*pdffile*" already exists")
  else
    (entropy_hedge, entropy_care, entropy_meta, Regret_hedge, Regret_care, Regret_meta) = sim_game(log2N/log(2), T, scenario, parms,nsims)

    # Ts = 10 .^(1:Int(log10(T)))
    # T2s = [2 .^(1:Int(floor(log2(T)))); T]
    # # T2s = 1:2:T
    # T4s = [4 .^(1:Int(floor(log2(T)/2))); T]
    #
    # maxy = 10^(ceil(2*log10(max(maximum(Regret_hedge),maximum(Regret_care),maximum(Regret_meta))))/2)
    #
    # p_regret = plot(T2s,Regret_hedge[T2s], label="Hedge", ylabel = "Regret", xlabel="Time",
    # legend=:topleft, title=title, markershape = :diamond, markersize=7*(in.(T2s,Ref(T4s))),
    # linewidth =3,yaxis=:log, xaxis=:log, ylim=(1,maxy))
    # #,yaxis=:log, xaxis=:log)
    # plot!(T2s,Regret_care[T2s],label = "FTRL-CARE", markershape = :star, markersize=7*(in.(T2s,Ref(T4s))), linewidth =3)
    # plot!(T2s,Regret_meta[T2s],label = "MetaCare", markershape = :utriangle, markersize=7*(in.(T2s,Ref(T4s))), linewidth =3)
    # savefig(p_regret, pngfile)
    # savefig(p_regret, pdffile)


    Plots.scalefontsizes(1.5)
    # Ts = [1 .+ 10 .^(1:Int(log10(T))), T-1]
    eq_spaced_odd_log10 = Int.(2*floor.(10* 10 .^(0:0.1:0.9) ./2) .+1)
    Ts = vcat([1;3;5;7;9], eq_spaced_odd_log10 ,eq_spaced_odd_log10*(10 .^(1:(Int(log10(T))-2)))' .+1..., T-1)
    # maxy = 10^(ceil(2*log10(max(maximum(Regret_hedge),maximum(Regret_care),maximum(Regret_meta))))/2)
    p_regret = plot(Ts,Regret_hedge[Ts], label="Hedge", ylabel = "Regret", xlabel="Time",
    legend=:topleft, linestyle = :dot, thickness_scaling=1, legendfontsize=14,
    dpi = 150,
    linewidth =5,yaxis=:log, xaxis=:log, ylims=(10^(-1),10^4))
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

    # p_entropy = plot(2:2:T,entropy_hedge[2:2:T], label="Hedge", ylabel = "Entropy", xlabel="Time", legend=:bottomright, title=title)#,yaxis=:log, xaxis=:log)
    # plot!(2:2:T,entropy_care[2:2:T],label = "FTRL-CARE")
    # plot!(2:2:T,entropy_meta[2:2:T],label = "MetaCare")
    # savefig(p_entropy, output_path*"/entropy_"*title*".pdf")
    print(title*"\n")
  end
end

function plot_game_N(log2Ns, Ts, scenario, parms,nsims)
  numNs = length(log2Ns)
  numTs = length(Ts)
  Tmax = maximum(Ts)

  entropy_hedge = zeros(Tmax,numNs)
  entropy_care = zeros(Tmax,numNs)
  entropy_meta = zeros(Tmax,numNs)
  Regret_hedge = zeros(Tmax,numNs)
  Regret_care = zeros(Tmax,numNs)
  Regret_meta = zeros(Tmax,numNs)


  logT = floor(Int8, log10(T))
  if scenario == "gap" && parms[1]==1
    title = ("logT="*string(logT)*", Stochastic-w-Gap="*string(parms[2]))
  elseif  scenario == "gap" && parms[1]>1
    title = ("logT="*string(logT)*", "*string(parms[1])*"-EE-Stochastic-w-Gap="*string(parms[2]))
  elseif  scenario == "link-gap" && parms[1]>1
    title = ("logT="*string(logT)*", 2-EE-Coupled-Stochastic-w-Gap="*string(parms[2]))
  elseif scenario == "alternating"
    title = ("logT="*string(logT)*", Alternating, N0=2")
  end

  filename = replace(title, ", "=>"--" )
  pngfile = output_path*"/huge/png/regret_"*filename*".png"
  pdffile = output_path*"/huge/pdf/regret_"*filename*".pdf"


  # if isfile(pngfile)
  #   print("File: "*pngfile*" already exists")
  # elseif isfile(pdffile)
  #   print("File: "*pdffile*" already exists")
  # else
    Threads.@threads for i in 1:numNs
      if scenario == "gap"
        (entropy_hedge[:,i], entropy_care[:,i], entropy_meta[:,i],
        Regret_hedge[:,i], Regret_care[:,i], Regret_meta[:,i]) =
          # getindex.(sim_game(log2Ns[i]/log(2), T, scenario, parms*(log2Ns[i]/log(2)/maximum(log2Ns))^(2),nsims),T)
          sim_game(log2Ns[i]/log(2), Tmax, scenario, parms,nsims)
      else
        (entropy_hedge[:,i], entropy_care[:,i], entropy_meta[:,i],
        Regret_hedge[:,i], Regret_care[:,i], Regret_meta[:,i]) =
          # getindex.(sim_game(log2Ns[i]/log(2), T, scenario, (parms[1],parms[2]*(log2Ns[i]/log(2)/maximum(log2Ns))^(2)),nsims),T)
          sim_game(log2Ns[i]/log(2), Tmax, scenario, parms,nsims)
      end
    end

    log2Nss = minimum(log2Ns):((maximum(log2Ns) - minimum(log2Ns))÷8):maximum(log2Ns)

    maxy = 10^(ceil(2*log10(max(maximum(Regret_hedge[Ts,:]),maximum(Regret_care[Ts,:]),maximum(Regret_meta[Ts,:]))))/2)
    miny = 10^(floor(2*log10(min(minimum(Regret_hedge[Ts,:]),minimum(Regret_care[Ts,:]),minimum(Regret_meta[Ts,:]))))/2)

    for T in Ts
      logT = floor(Int8, log10(T))
      if scenario == "gap" && parms[1]==1
        title = ("logT="*string(logT)*", Stochastic-w-Gap="*string(parms[2]))
      elseif  scenario == "gap" && parms[1]>1
        title = ("logT="*string(logT)*", "*string(parms[1])*"-EE-Stochastic-w-Gap="*string(parms[2]))
      elseif  scenario == "link-gap" && parms[1]>1
        title = ("logT="*string(logT)*", 2-EE-Coupled-Stochastic-w-Gap="*string(parms[2]))
      elseif scenario == "alternating"
        title = ("logT="*string(logT)*", Alternating, N0=2")
      end

      filename = replace(title, ", "=>"--" )
      pngfile = output_path*"/huge/png/regret_"*filename*".png"
      pdffile = output_path*"/huge/pdf/regret_"*filename*".pdf"
      Plots.scalefontsizes(1.5)
      p_regret = plot(log2Ns,Regret_hedge[T,:], label="Hedge", ylabel = "Regret", xlabel="log2(N)",
      legend=:bottomright, ylims=(1,maxy), yaxis=:log, xaxis=:log,
      linestyle = :dot, linewidth =5, thickness_scaling=1, legendfontsize=14,
      dpi = 150)#,yaxis=:log, xaxis=:log)
      plot!(log2Ns,Regret_care[T,:],label = "FTRL-CARE", linestyle = :dash, linewidth =4)
      plot!(log2Ns,Regret_meta[T,:],label = "MetaCare", linestyle = :solid, linewidth =2)
      savefig(p_regret, pngfile)
      savefig(p_regret, pdffile)
      Plots.scalefontsizes(1/1.5)
      # p_entropy = plot(log2Ns,entropy_hedge, label="Hedge", ylabel = "Entropy", xlabel="Time", legend=:bottomright, title=title)#,yaxis=:log, xaxis=:log)
      # plot!(log2Ns,entropy_care,label = "FTRL-CARE")
      # plot!(log2Ns,entropy_meta,label = "MetaCare")
      # savefig(p_entropy, output_path*"/entropy_"*filename*".pdf")
      print(title*"\n")

    end

  # end
end
T = 1000000
log2N = 16

# Ts=[100,1000,10000,100000]
Ts=[100,1000,10000]

# gaps = [0; 1/16; 1/4; 1/2]
log2Ns = [4;8;16; 32; 64; 128]
# log2Ns = [4;16; 64]

# 2EE Stochastic Cases
# scenario = "gap"
# for j in 1:length(log2Ns)
#   plot_game(log2Ns[j], T, scenario, (2,0.95), 10)
# end
# scenario = "link-gap"
# for j in 1:length(log2Ns)
#   plot_game(log2Ns[j], T, scenario, (2,.5), 10)
# end


# Alternating Cases
# scenario = "alternating"
# for j in 1:length(log2Ns)
#   N02 = 1
#   plot_game(log2Ns[j], T, scenario, (N02 , .5), 1)
# end


scenario = "alternating"
    N02 = 1
    Ts=[100,1000,10000,100000,1000000]
    log2Ns = 4:4:128
    plot_game_N(log2Ns, Ts, scenario, (N02 , .5), 1)

# Stochastic Cases
# T = 100000
# scenario = "gap"
# for j in 1:length(log2Ns)
#   plot_game(log2Ns[j], T, scenario, (1,.5), 10)
# end
