module JuliaMOR

using PyCall
using PyPlot
using JLD, HDF5
unshift!(PyVector(pyimport("sys")["path"]), "")

export rbfile, params, S, ns, sigma, sigma2, N, N2, Nh, Afem, bfem, U, U2, Xh, r


# data and fem related functions #

export jldimport, @femimport, load

# restart global variables #
function clean()
    global params = 0
    global S = 0
    global ns = 0
    global sigma = 0
    global sigma2 = 0
    global N = 0
    global N2 = 0
    global Nh = 0
    global Afem = 0
    global bfem = 0
    global U = 0
    global U2 = 0
    global Xh = 0
    global r = 0
end

# import jld file with data from a RB problem #
function jldimport(name::AbstractString)
    clean()
    global rbfile = jldopen(name * ".jld", "r+")

    nothing
end

# create a new jld file for RB data #
function jldnew(name::AbstractString)
    clean()
    global rbfile = jldopen(name * ".jld", "w")

    nothing
end

# import python fem solver #
macro femimport(eq, eqname...)
    eval(:(@pyimport $eq as fem))

    len = length(eqname)

    if len > 0 && (len != 2 || eqname[1] != :as)
        throw(ArgumentError("The correct use is @femimport eq or @femimport eq as eqname"))
    elseif len == 2
        name = string(eqname[2])
    else
        name = string(eq)
    end

    try
        jldimport(name)
    catch
        jldnew(name)
    end
end

# return variable from the data #
macro load(var, inds...)
    len = length(inds)
    name = string(var)

    if len == 0
        q = quote
            if $var == 0
                global $var = read(rbfile, $name)
            end

            $var
        end
    else
        #temp code until getindex in JLD module is fixed for arrays of inds
        i1 = inds[1]

        if len == 1
            q = quote
                if $var == 0
                    global $var = read(rbfile, $name)
                end

                $var[$i1]
            end
        else
            i2 = inds[2]

            q = quote
                if $var == 0
                    global $var = read(rbfile, $name)
                end

                $var[$i1, $i2]
            end
        end
    end
end

# load variable readable from REPL #
function load(var::Symbol)
    name = string(var)

    q = quote
        if $var == 0
            global $var = read(rbfile, $name)
        end

        $var
    end

    eval(q)
end

##


# pod related functions #

export poderror, pod, podsub

# compute for which N the truncation of pod gives the error bound indicated #
function poderror(error::Float64 = 1.0e-6, sigma::Vector{Float64} = @load sigma)
    sigma2 = sigma.^2
    s1 = sum(sigma2[2:end])

    r = @load r
    N = 1

    while s1 > error && N < r
        N += 1
        s1 -= sigma2[N]
    end

    N
end

# compute pod error for N elements #
function poderror(N::Int64, sigma::Vector{Float64} = @load sigma)
	  sigma2 = sigma.^2

    sum(sigma2[N+1:end])
end

# compute pod for the selected parameters #
function pod(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}, Vector{Vector{Float64}}}, error::Float64 = 1.0e-6)
    offsol(params)

    pod(error)
end

function pod(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}, Vector{Vector{Float64}}}, N::Int64)
    offsol(params)

    pod(N)
end

function pod(error = 1.0e-6, S::Matrix{Float64} = @load S)
    pod(0, S)

    o_delete(rbfile, "N")
    rbfile["N"] = poderror(error)

    o_delete(rbfile, "N2")
    rbfile["N2"] = poderror(error, @load sigma2)

    nothing
end

function sparsepytojl(A::SparseMatrixCSC{Float64,Int64}, o::PyObject)
    ind = o[:nonzero]()
    indx, indy = ind[1] + 1, ind[2] + 1
    odata = o[:data]

    for (i, data) in enumerate(odata)
        A[indx[i], indy[i]] = data
    end
end

function pod(N1::Int64, S::Matrix{Float64} = @load S)
    @load r
    @load ns
    @load Nh

    temp = fem.Xh()
    Xh = spzeros(Nh, Nh)

    sparsepytojl(Xh, temp)

    rbfile["Xh"] = Xh

    U = eigfact(Symmetric(S' * Xh * S), ns-r+1:ns)

    sigma = sqrt(abs(U[:values][end:-1:1]))
    U = S * U[:vectors][:, end:-1:1] * spdiagm(1 ./ sigma)

    rbfile["U"] = U
    rbfile["sigma"] = sigma
    rbfile["N"] = N1

    F = svdfact(S)

    rbfile["U2"] = F[:U][:, 1:r]
    rbfile["sigma2"] = F[:S][1:r]
    rbfile["N2"] = N1

    if fem.isfreeA()
        temp = fem.Afem()
        Afem = spzeros(Nh, Nh)

        sparsepytojl(Afem, temp)

        rbfile["Afem"] = Afem
    end

    if fem.isfreeb()
        rbfile["bfem"] = fem.bfem()
    end

    nothing
end


# compute pod for a subset of parameters #
function podsub(inds::Vector{Int64}, name::AbstractString)
    params1 = @load(params, inds)
    S1 = @load(S, :, inds)
    @load(Afem)

    jldnew(name)

    pod(params1, S1, Afem)
end


##

# parameters manipulation related functions #

# vectorize parameters
paramsvec(params::Matrix{Vector{Float64}}) = vec(params)

function paramsvec(params::FloatRange{Float64})
    params1 = Vector{Float64}[]

    for param in params
        push!(params1, [param])
    end

    params1
end

paramsvec(param::Float64) = [param]


# offline related functions #

export offsol, offtime

# compute the snapshots and prepare the jld file with the information #
function offsol(params1::Vector{Vector{Float64}})
    if exists(rbfile, "params") # check if there exists snapshots in params1 already computed in the jld
        @load S
        @load params
        @load ns

        ind = find(x -> length(find(y -> y == x , params)) != 0, params1) # indexes of the snapshots already in the jld from params1
        deleteat!(params1, ind) # delete those parameters in ind

        # compute and merge the new snapshots and the related info
        S1 = offcoef(params1)
        ns1 = length(params1)

        global S = [S S1]
        global ns = ns + ns1
        global params = [params; params1]

        # delete the old info in the jld
        o_delete(rbfile, "S")
        o_delete(rbfile, "ns")
        o_delete(rbfile, "params")
        o_delete(rbfile, "r")
    else
        global S = offcoef(params1)
        global ns = length(params1)
        global params = params1
        global Nh = fem.Nh()

        rbfile["Nh"] = Nh
    end

    rbfile["S"] = S
    rbfile["ns"] = ns
    rbfile["params"] = params
    rbfile["r"] = rank(S)

    nothing
end

# these two offsol are other cases of parameters that call the one above #
offsol(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}}) = offsol(paramsvec(params))

#offsol(params::FloatRange{Float64}) = offsol(paramsvec(params))


# call the fem solver for the parameters, returns the coefficients of the
# solutions in each column #
function offcoef(params::Vector{Vector{Float64}})
    Nh = fem.Nh()
    ns = length(params)

    sols = Matrix{Float64}(Nh, ns)

    for (ind, p) in enumerate(params)
        sols[:, ind] = offcoef(p)
    end

    sols
end

# call the fem solver for a param #
offcoef(param::Vector{Float64}) = fem.offcoef(param)

# test how much time will take to finish the offline stage #
function offtime(params::Vector{Vector{Float64}})
	  offcoef(rand(params))

	  tic()
	  offcoef(rand(params))
	  time = toc();

    ns = length(params)
	  time = time * ns

    timeprint(time)
end

function timeprint(time::Float64)
    if time <= 60
		    print(time)
		    print(" sec")
	  elseif 60 < time && time <= 3600
		    print(time/60)
		    print(" min")
	  else
		    print(time/3600)
		    print(" hr")
	  end
end


offtime(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}}) = offtime(paramsvec(params))

# call the fem solver for a parameter and returns its solution as a function #
function offsol(param::Vector{Float64})
    fem.offsol(param)

    x -> fem.offeval(x)
end

##


# errors related functions

export errors, poderrors, singularvalues, offontime

offontime(param::Float64) = offontime(paramsvec(param))

function offontime(param::Vector{Float64})
    println("Offline computations:")
    tic()
    offcoef(param)
    off = toc()
    println("")

    println("Online computations:")
    tic()
    oncoef(param)
    on = toc()
    println("")

    diff = off - on

    if off >= on
        speedup = round(Int, 100 * diff / on)
        print("There is a speed up of $speedup%")
    else
        print("Online computations are slower than the offline ones with a difference of $diff sec")
    end
end

# compute the l2 errors from 1 to N for a parameter #
function errors(param::Vector{Float64}, N::Int64, UXh::Bool = true, normXh::Bool = true)
    if normXh
        Xh = @load Xh
        normfun = x -> sqrt(dot(x, Xh*x))
    else
        normfun = norm
    end

    off = offcoef(param)

    Afem = @load Afem
    bfem = fem.bfem(param)
    U = UXh ? rbfile["U"][:, 1:N] : rbfile["U2"][:, 1:N]

    errors = Float64[]

    for i = 1:N
        RB = U[:, 1:i]
        RBt = RB'

        on = RB * (((RBt * Afem * RB)') \ (RBt * bfem))

        push!(errors, normfun(off - on))
    end

    errors
end

errors(param::Float64, N::Int64) = errors(paramsvec(param), N)

# compute the pod errors from 1 to N #
function poderrors(N::Int64, sigma::Vector{Float64} = @load sigma)
    errors = Float64[]

    for i = 1:N
        push!(errors, poderror(i, sigma))
    end

    errors
end

singularvalues(N::Int64, sigma::Vector{Float64} = @load sigma) = sigma[1:N]

export onbasis

function onbasis(i::Int64)
    coef = vec(rbfile["U"][:,i])

    fem.offsolcoef(coef)

    x -> fem.offeval(x)
end

##


# S related functions

export Srank

# compute rank of S #
Srank(S::Matrix{Float64} = @load S) = rank(S)

##

##

# Functional interpolation related functions

export funinterpolation

# compute Chebychev interpolation of a functional in 2D
funinterpolation(fun::Function, x1::Vector{Float64}, x2::Vector{Float64}) = Fun(fun, x1, x1)


###################################################################################################

# Compute the online coefficients for param

@generated function oncoef(param::Vector{Float64})
    Apde = @load Afem
    RB = rbfile["U"][:, 1:@load N]
    RBt = RB'

    if fem.isfreeA()
        ARB = factorize((RBt * Apde * RB)')
        return :($ARB \ ($RBt * fem.bpde(param)))
    else
        return :(($RBt * fem.Afem(param) * $RB)' \ ($RBt * fem.bfem(param)))
    end
end




# Solve the equation for param and put in evals
# the full evaluations of some prevaluated values
# of the base functions

offsolve!(param::Vector{Float64}, evals::Vector{Float64}, values::Vector{Vector{Float64}}) = solve!(param, evals, values, offcoef)


onsolve!(param::Vector{Float64}, evals::Vector{Float64}, values::Vector{Vector{Float64}}) = solve!(param, evals, values, oncoef)


function solve!(param::Vector{Float64}, evals::Vector{Float64}, values::Vector{Vector{Float64}}, coefun::Function)
    coef = coefun(param)

    for (ind, vals) in enumerate(values)
        evals[ind] = dot(coef, vals)
    end

    nothing
end


# Evaluate the base in x

@generated function offbase(x::Vector{Float64})
    fem.buildbase()

    :(fem.offbase(x))
end

@generated function onbase(x::Vector{Float64})
    if !fem.isRB()
        RBt = readdlm(filenames("RB"))'
        fem.buildRB(RBt)
    end

    :(fem.onbase(x))
end


# graphs related functions

export elementsurf, funsurf


function funsurf(u::Function, points::FloatRange{Float64} = 0:0.01:1)
    len = length(points)

    Z = Array{Float64, 2}(len,len)
    x = [0.0, 0.0]

    for p1 in 1:len
        for p2 in 1:len
            x[1] = points[p1]
            x[2] = points[p2]
            Z[p1,p2] = u(x)
        end
    end

    surf(points,points,Z)
    return Z
end




# params related functions

export paramexist

function paramexist(param::Vector{Float64})
    load(:params)

    length(find(x -> x == param, params)) != 0
end

end

function uhexact(param::Vector{Float64})
    #@load Apdei

    tmp = fem.uhexact(param)

    ind = tmp[:, 1]
    values = tmp[:, 2]

    coefs = dot(values, Apdei[:, ind])
end

