module jMOR

import Base: .*
using PyCall
using JLD, HDF5
using Base.SparseArrays 
unshift!(PyVector(pyimport("sys")["path"]), "")

export rbfile, fem, jldimport, @femimport, load

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
        jldimport(name, "r+")
    catch
        jldimport(name, "w")
    end

    clean()
end

# import jld file with data from the RB problem #
function jldimport(name::AbstractString, access::AbstractString)
    global rbfile = jldopen(name * ".jld", access)

    # initialize all the variables
    if access == "r+"
        for var in map(x -> Symbol(x), names(rbfile))
            q = quote
                global $var = 0
                export $var
            end

            eval(q)
        end
    end

    nothing
end

export params, S, ns, sigma, sigma2, N, N2, Nh, Afem, bfem, U, U2, Xh, r 

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
    global r  = 0
    
    nothing
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
            export $var
        end

        $var
    end

    eval(q)
end

# pod related functions #

export poderror, pod, podsub, offsol, offtime

# compute pod for the selected parameters #
ParamsUnion = Union{Matrix{Vector{Float64}}, FloatRange{Float64}, Vector{Vector{Float64}}}
function pod(params::ParamsUnion, error::Float64 = 1.0e-6)
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

    affinedim = fem.affinedim()

    if affinedim[1] == -1
        temp = fem.Afem()
        Afem = spzeros(Nh, Nh)

        sparsepytojl(Afem, temp)

        rbfile["Afem"] = Afem
    end

    if affinedim[2] == -1
        rbfile["bfem"] = fem.bfem()
    end

    Ut = U'

    for i = 1:affinedim[1]
        temp = fem.Afem(i)
        Afem = spzeros(Nh, Nh)

        sparsepytojl(Afem, temp)

        rbfile["Afem$i"] = (Ut * Afem * U)'
    end

    for i = 1:affinedim[2]
        rbfile["bfem$i"] = Ut * fem.bfem(i)
    end

    nothing
end

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
        global ns += ns1
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

# this offsol function change the structure of params to be compatible with the function above #
offsol(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}}) = offsol(paramsvec(params))
# vectorize parameters #
paramsvec(params::Matrix{Vector{Float64}}) = vec(params)
paramsvec(params::Union{FloatRange{Float64}, Float64}) = map(x -> [x], params)

# call the fem solver for a parameter and returns its solution as a function #
function offsol(param::Vector{Float64})
    fem.offsol(param)

    x -> fem.offeval(x)
end

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

# call the fem solver for a singular param #
offcoef(param::Vector{Float64}) = fem.offcoef(param)

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

# auxiliar function to convert scipy sparse matrix to julia sparse structure #
function sparsepytojl(A::SparseMatrixCSC{Float64,Int64}, o::PyObject)
    ind = o[:nonzero]()
    indx = ind[1] + 1
    indy = ind[2] + 1
    odata = filter(x -> x != 0, o[:data])

    for i in 1:length(indx)
        A[indx[i], indy[i]] = odata[i]
    end
end

# compute pod for a subset of parameters #
function podsub(inds::Vector{Int64}, name::AbstractString)
    params1 = @load(params, inds)
    S1 = @load(S, :, inds)
    @load(Afem)

    jldnew(name)

    pod(params1, S1, Afem)
end

# test how much time will take to finish the offline stage #
offtime(params::Union{Matrix{Vector{Float64}}, FloatRange{Float64}}) = offtime(paramsvec(params))
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

#################################################################

# errors related functions #

export errors, poderrors, singularvalues, offontime

# compare the offline and online times #
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
errors(param::Float64, N::Int64) = errors(paramsvec(param), N)
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

# compute the pod errors from 1 to N #
poderrors(N::Int64, sigma::Vector{Float64} = @load sigma) = map(x -> poderror(x, sigma), 1:N)

# return the first N singular values
singularvalues(N::Int64, sigma::Vector{Float64} = @load sigma) = sigma[1:N]

# online stage related functions

export onbasis, onload, oncoef, onsol, ontofem

# return i-th base of the reduced basis #
function onbasis(i::Int64)
    coef = vec(rbfile["U"][:,i])

    fem.offsolcoef(coef)

    x -> fem.offeval(x)
end

.*(a::Vector{Vector{Float64}}, b::Vector{Float64}) = map(i -> a[i] * b[i], 1:length(b))
.*(a::Vector{SparseMatrixCSC{Float64,Int64}}, b::Vector{Float64}) = map(i -> a[i] * b[i], 1:length(b))

function prepare()
   @load Afem
   nothing
end

# compute the online coefficients for param #

@generated function oncoef(param::Vector{Float64})
    affinedim = fem.affinedim()

    RB = @load U
    RBt = RB'

    if affinedim[1] == -1 # Afem independent
        Apde = @load Afem
        ARB = factorize((RBt * Apde * RB)')
        if affinedim[2] == 0 # bfem not affine
            return :($ARB \ ($RBt * fem.bfem(param)))
        else # bfem affine
            bfems = Vector{Float64}[]

            for i = 1:affinedim[2]
                push!(bfems, read(rbfile, "bfem$i"))
            end

            return :($ARB \ sum($bfems .* fem.bcoefs(param)))
        end
    elseif affinedim[1] == 0 # Afem not affine
        if affinedim[2] == -1 # bfem independent
            bRB = RBt * (@load bfem)

            return :(($RBt * fem.Afem(param) * $RB)' \ $bRB)
        elseif affinedim[2] == 0 # bfem not affine
            return :(($RBt * fem.Afem(param) * $RB)' \ ($RBt * fem.bfem(param)))
        else # bfem affine
            bfems = Vector{Float64}[]

            for i = 1:affinedim[2]
                push!(bfems, read(rbfile, "bfem$i"))
            end

            return :(($RBt * fem.Afem(param) * $RB)' \ sum($bfems .* fem.bcoefs(param)))
        end
    else # Afem affine
        Afems = Matrix{Float64}[] # change this

        for i = 1:affinedim[1]
            push!(Afems, read(rbfile, "Afem$i"))
        end

        if affined[2] == -1 # bfem independent
            bRB = RBt * (@load bfem)

            return :(sum($Afems .* fem.Acoefs(param)) \ $bRB)
        elseif affined[2] == 0 # bfem not affine
            return :(sum($Afems .* fem.Acoefs(param)) \ ($RBt * fem.bfem(param)))
        else
            bfems = Vector{Float64}[]

            for i = 1:affinedim[2]
                push!(bfems, read(rbfile, "bfem$i"))
            end

            return :(sum($Afems .* fem.Acoefs(param)) \ sum($bfems .* fem.bcoefs(param)))
        end
    end
end

# return the online solution as a function #
function onsol(param::Vector{Float64})
    fem.offsolcoef(ontofem(oncoef(param)))

    x -> fem.offeval(x)
end

# convert the reduced basis coefficients to the fem coefficients #
@generated function ontofem(param::Vector{Float64})
    RB = @load U

    return :(RB * oncoef(param))
end

#=

# Solve the equation for a param and put in evals
# the full evaluations of some prevaluated values
# of the basis functions

export offsol!, onsol!, offbase, onbase, solve!, oncoef, test

test() = @load Afem

offsol!(param::Vector{Float64}, evals::Vector{Float64}, values::Vector{Vector{Float64}}) = solve!(param, evals, values, offcoef)

function onsol!(param::Vector{Float64}, evals::Vector{Float64}, values::Vector{Vector{Float64}})
    #@load Afem
    #@load U

    solve!(param, evals, values, oncoef)
end


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
        RBt = (@load U)'
        fem.buildRB(RBt)
    end

    :(fem.onbase(x))
end

=#

####################################################################################

# EIM functions

export offeim, oneim, eim, eimevals, eimevalsgrad

function eimevals(g::Function, paramsx::Vector{Vector{Float64}}, paramsmu::Vector{Vector{Float64}})
    evals = Vector{Float64}[]
    gs_x = map(mu -> (x -> g(x, mu)), paramsmu)

    for g_x in gs_x
        push!(evals, map(g_x, paramsx))
    end

    hcat(evals...)
end

function eimevalsgrad(g::Function, paramsx::Vector{Vector{Float64}}, paramsmu::Vector{Vector{Float64}})
    results = Array{DiffBase.DiffResult{1, Float64, Tuple{Array{Float64, 1}}}, 1}[]
    gs_x = map(mu -> (x -> g(x, mu)), paramsmu)

    result = DiffBase.GradientResult(paramsx[1])

    for g_x in gs_x
        push!(results, map(x -> ForwardDiff.gradient!(copy(result), g_x, x), paramsx))
    end

    evals = hcat(map(x -> map(DiffBase.value, x), results)...)
    grad1 = hcat(map(x -> map(y -> DiffBase.gradient(y)[1], x), results)...)
    grad2 = hcat(map(x -> map(y -> DiffBase.gradient(y)[2], x), results)...)

    (evals, grad1, grad2)
end


function offeim(evals::Matrix{Float64}, Q::Matrix{Float64}, J::Vector{Int64}, Jmu::Vector{Int64}, iter_max::Int64 = 2^11, tol::Float64 = 200eps())
    M = size(Q)[2]
    error_M = tol + 1

    siz = size(evals)

    Q = hcat(Q, Matrix{Float64}(siz[1], iter_max - M))

    minmaxeval = [0., 0.]
    function findmax(values::Matrix{Float64})
        imax = indmax(values)
        imin = indmin(values)
        minmaxeval[1] = values[imax]
        minmaxeval[2] = abs(values[imin])

        imax = (indmax(minmaxeval) == 1) ? imax : imin

        ind2sub(siz, imax)
    end

    Q_M = Q[:, 1:M]
    allres = evals - Q_M * (Q_M[J, :] \ evals[J, :])

    ind_mu_M = findmax(allres)

    error_M = abs(allres[ind_mu_M[1], ind_mu_M[2]])

    r = allres[:, ind_mu_M[2]]

    count = 0

    while M < iter_max && count < 8
        tic()
        M += 1

        #i_M = indmax(abs(r))
        i_M = ind_mu_M[1]
        rho_M = r / r[i_M]

        Q[:, M] = rho_M
        push!(J, i_M)
        push!(Jmu, ind_mu_M[2])

        Q_M = Q[:, 1:M]

        allres = evals - Q_M * (Q_M[J, :] \ evals[J, :])
        #=
        temp = Q_M[J, :] \ evals[J, :]
        maxvalue = 0
        for i in 1:siz[2]
            allres = evals[:, i] - Q_M * temp[:, i]
        end
        =#

        ind_mu_M = findmax(allres)

        error_M = abs(allres[ind_mu_M[1], ind_mu_M[2]])

        r = allres[:, ind_mu_M[2]]

        if error_M <= tol
            count += 1
        end

        toc()
        println(error_M)
    end

    (Q[:,1:M], J, Jmu, error_M)
end
function offeim(evals::Matrix{Float64}, iter_max::Int64 = 1500, tol::Float64 = 1e-11)
    M = 0
    error_M = tol + 1

    siz = size(evals)
    iter_max = min(iter_max, siz[2])

    Q = Matrix{Float64}(siz[1], iter_max)
    J = Int64[]
    Jmu = Int64[]

    minmaxeval = [0., 0.]
    function findmax(values::Matrix{Float64})
        imax = indmax(values)
        imin = indmin(values)
        minmaxeval[1] = values[imax]
        minmaxeval[2] = abs(values[imin])

        imax = (indmax(minmaxeval) == 1) ? imax : imin

        ind2sub(siz, imax)
    end

    ind_mu_M = findmax(evals)

    r = evals[:, ind_mu_M[2]]

    count = 0

    while M < iter_max && count < 1
        tic()
        M += 1

        #i_M = indmax(abs(r))
        i_M = ind_mu_M[1]
        rho_M = r / r[i_M]

        Q[:, M] = rho_M
        push!(J, i_M)
        push!(Jmu, ind_mu_M[2])

        Q_M = Q[:, 1:M]

        allres = evals - Q_M * (Q_M[J, :] \ evals[J, :])
        #=
        temp = Q_M[J, :] \ evals[J, :]
        maxvalue = 0
        for i in 1:siz[2]
            allres = evals[:, i] - Q_M * temp[:, i]
        end
        =#

        ind_mu_M = findmax(allres)

        error_M = abs(allres[ind_mu_M[1], ind_mu_M[2]])

        r = allres[:, ind_mu_M[2]]

        if error_M <= tol
            count += 1
        end

        toc()
        println(error_M)
    end

    (Q[:,1:M], J, Jmu, error_M)
end

function offeim(evals::Matrix{Float64}, grad1::Matrix{Float64}, grad2::Matrix{Float64}, iter_max::Int64 = 2^11, tol::Float64 = 200eps())
    M = 0
    error_M = tol + 1
    error_Mgrad1 = 0
    error_Mgrad2 = 0

    siz = size(evals)

    Q = Matrix{Float64}(siz[1], iter_max)
    Qgrad1 = Matrix{Float64}(siz[1], iter_max)
    Qgrad2 = Matrix{Float64}(siz[1], iter_max)
    J = Int64[]

    minmaxeval = [0., 0.]
    function findmax(values::Matrix{Float64})
        imax = indmax(values)
        imin = indmin(values)
        minmaxeval[1] = values[imax]
        minmaxeval[2] = abs(values[imin])

        imax = (indmax(minmaxeval) == 1) ? imax : imin

        ind2sub(siz, imax)
    end

    ind_mu_M = findmax(evals)

    r = evals[:, ind_mu_M[2]]
    rgrad1 = grad1[:, ind_mu_M[2]]
    rgrad2 = grad2[:, ind_mu_M[2]]

    rref = r

    count = 0

    while M < iter_max && count < 8
        tic()
        M += 1

        i_M = indmax(abs(rref))
        rho_M = r / rref[i_M]
        rho_Mgrad1 = rgrad1 / rref[i_M]
        rho_Mgrad2 = rgrad2 / rref[i_M]

        Q[:, M] = rho_M
        Qgrad1[:, M] = rho_Mgrad1
        Qgrad2[:, M] = rho_Mgrad2
        push!(J, i_M)

        Q_M = Q[:, 1:M]
        Q_Mgrad1 = Qgrad1[:, 1:M]
        Q_Mgrad2 = Qgrad2[:, 1:M]

        coefs = Q_M[J, :] \ evals[J, :]
        allres = evals - Q_M * coefs
        allresgrad1 = grad1 - Q_Mgrad1 * coefs
        allresgrad2 = grad2 - Q_Mgrad2 * coefs
        #=
        temp = Q_M[J, :] \ evals[J, :]
        maxvalue = 0
        for i in 1:siz[2]
            allres = evals[:, i] - Q_M * temp[:, i]
        end
        =#

        ind_mu_M = findmax(allres)
        ind_mu_Mgrad1 = findmax(allresgrad1)
        ind_mu_Mgrad2 = findmax(allresgrad2)

        error_M = abs(allres[ind_mu_M[1], ind_mu_M[2]])
        error_Mgrad1 = abs(allresgrad1[ind_mu_Mgrad1[1], ind_mu_Mgrad1[2]])
        error_Mgrad2 = abs(allresgrad2[ind_mu_Mgrad2[1], ind_mu_Mgrad2[2]])

        max123 = indmax([error_M, error_Mgrad1, error_Mgrad2])
        indref = ind_mu_M[2]

        if max123 == 1
            rref = r
        elseif max123 == 2
            indref = ind_mu_Mgrad1[2]
            rref = rgrad1
        else
            indref = ind_mu_Mgrad2[2]
            rref = rgrad2
        end

        r = allres[:, indref]
        rgrad1 = grad1[:, indref]
        rgrad2 = grad2[:, indref]

        if max123 == 1
            rref = r
        elseif max123 == 2
            indref = ind_mu_Mgrad1[2]
            rref = rgrad1
        else
            indref = ind_mu_Mgrad2[2]
            rref = rgrad2
        end

        if error_M <= tol && error_Mgrad1 <= tol && error_Mgrad2 <= tol
            count += 1
        end

        toc()
        println(error_M)
        println(error_Mgrad1)
        println(error_Mgrad2)
        println("ok")
    end

    (Q[:,1:M], Qgrad1[:,1:M], Qgrad2[:,1:M], J, error_M, error_Mgrad1, error_Mgrad2)
end

function offeim(g::Function, paramsx::Vector{Vector{Float64}}, paramsmu::Vector{Vector{Float64}}, iter_max::Int64 = 2^14, tol::Float64 = 200eps(), preevaluate::Bool = true)
    M = 0
    error_M = tol + 1

    siz = (length(paramsx), length(paramsmu))

    Q = Matrix{Float64}(siz[1], iter_max)
    J = Int64[]
    mus = Vector{Float64}[]

    if preevaluate
        evals = Vector{Float64}[]
        gs_x = map(mu -> (x -> g(x, mu)), paramsmu)

        for g_x in gs_x
            push!(evals, map(g_x, paramsx))
        end

        evals = hcat(evals...)

        ind_mu_M = ind2sub(siz, indmax(abs(evals)))

        r = evals[:, ind_mu_M[2]]

        count = 0

        while M < iter_max && count < 8
            tic()
            M += 1

            i_M = indmax(abs(r))
            rho_M = r / r[i_M]

            Q[:, M] = rho_M
            push!(J, i_M)

            Q_M = Q[:, 1:M]

            allres = evals - Q_M * (Q_M[J, :] \ evals[J, :])


            ind_mu_M = ind2sub(siz, indmax(abs(allres)))

            error_M = abs(allres[ind_mu_M[1], ind_mu_M[2]])

            r = allres[:, ind_mu_M[2]]

            if error_M <= tol
                count += 1
            end


            toc()
            println(error_M)
        end
    else
        mu_M = argmax(g, paramsx, paramsmu)[2]

        r = map(x -> g(x, mu_M), paramsx)

        count = 0

        while M < iter_max && count < 8
            tic()
            M += 1

            i_M = indmax(abs(r))
            rho_M = r / r[i_M]

            Q[:, M] = rho_M
            push!(J, i_M)

            Q_M = Q[:, 1:M]

            (error_M, mu_M) = argmax(g, Q_M, J, paramsx, paramsmu)

            gmu = map(x -> g(x, mu_M), paramsx)

            r = gmu - Q_M*(Q_M[J, :] \ gmu[J])

            if error_M <= tol
                count += 1
            end

            toc()
            println(error_M)
        end
    end

    return (Q[:,1:M], mus, J, error_M)
end

function argmax(g::Function, paramsx::Vector{Vector{Float64}}, paramsmu::Vector{Vector{Float64}})
    fx = map(mu -> (x -> g(x, mu)), paramsmu)

    maxval = 0
    maxmu = 0

    for (i, f) in enumerate(fx)
        m = maximum(abs(map(f, paramsx)))

        if m > maxval
            maxval = m
            maxmu = paramsmu[i]
        end
    end

    return (maxval, maxmu)
end

function argmax(g::Function, Q::Matrix{Float64}, J::Vector{Int64}, paramsx::Vector{Vector{Float64}}, paramsmu::Vector{Vector{Float64}})
    fx = map(mu -> (x -> g(x, mu)), paramsmu)

    len = length(J)

    maxval = 0
    maxmu = 0

    Q1 = factorize(Q[J, :])

    for (i, f) in enumerate(fx)
        gmu = map(f, paramsx)
        m = maximum(abs(gmu - Q*(Q1 \ gmu[J])))

        if m > maxval
            maxval = m
            maxmu = paramsmu[i]
        end
    end

    return (maxval, maxmu)
end

function oneim(mu::Vector{Float64}, g::Function, Q::Matrix{Float64}, J::Vector{Int64}, paramsx::Vector{Vector{Float64}})
    gJmu = map(x -> g(x, mu), paramsx[J])

    Q[J, :] \ gJmu
end

oneim(mu::Vector{Float64}, g::Function, QQ_J::Base.LinAlg.LU{Float64,Array{Float64,2}}, paramsx_J::Vector{Vector{Float64}}) = QQ_J \ map(x -> g(x, mu), paramsx_J)


function oneim(mu1, mu2, g::Function, QQ_J::Base.LinAlg.LU{Float64,Array{Float64,2}}, paramsx_J::Vector{Vector{Float64}})
    mu = [mu1, mu2]
    QQ_J \ map(x -> g(x, mu), paramsx_J)
end

function eim(g::Function, mus::Vector{Vector{Float64}}, xs::Vector{Vector{Float64}})
    basis = Function[]
    coefs = [0.0]

    push!(basis, x -> g(x, mus[1]) / g(xs[1], mus[1]))

    function Ig_mu(x::Vector{Float64}, M::Int64)
        result = 0

        for (j, rho) in enumerate(basis[1:M])
            result += coefs[j] * rho(x)
        end

        result
    end

    for (i, mu) in enumerate(mus[2:end])
        tic()
        B_M = Matrix{Float64}(i, i)
        g_M = map(x -> g(x, mu), xs[1:i])

        for (j, rho) in enumerate(basis)
            B_M[:, j] = map(rho, xs[1:i])
        end

        coefs = B_M \ g_M

        push!(basis, x -> (g(x, mu) - Ig_mu(x, i)) / (g(xs[i+1], mu) - Ig_mu(xs[i+1], i)))
        toc()
    end

    basis
end

#######################################################################################

# Unfinished functions

#=
function greedy(params::Vector{Vector{Float64}}, mu1::Vector{Float64}, errormax::Float64 = 1e-6, Nmax::Int64 = 2000)
    Nh = fem.Nh()

    temp = fem.Afem()
    Afem = spzeros(Nh, Nh)
    sparsepytojl(Afem, temp)
    temp = 0

    c = 1 / 1.e-5

    len = length(params)
    V = Matrix{Float64}(fem.Nh(), Nmax)
    bfems = Matrix{Float64}(Nh, len)
    sampling = Vector{Float64}[]

    tic()
    for i in 1:len
        bfems[:, i] = fem.bfem(params[i])
    end
    toc()

    muN = mu1
    N, error = 0, errormax + 1

    function gramschmidt(sol::Vector{Float64}, N::Int64)
        if N != 0
            sol = sol - V[:, 1:N] * (V[:, 1:N])' * sol
        end

        sol = sol / norm(sol)
        nothing
    end
    function argmax(N::Int64)
        err = 0
        mu = 0

        VN = V[:, 1:N]
        VNt = VN'
        onsols = (VNt * Afem * VN)' \ (VNt * bfems)
        res = c * (bfems - Afem * VN * onsols)

        for i in 1:size(res)[2]
            normres = norm(res[:, i])

            if normres >= err
                err = normres
                mu = param
            end
        end

        (err, mu)
    end
    while N < Nmax && error > errormax
        tic()
        N += 1

        sol = offcoef(muN)
        gramschmidt(sol, N-1)

        V[:, N] = sol
        push!(sampling, muN)

        error, muN = argmax(N)
        println(error)
        toc()
    end
end

=#


end
