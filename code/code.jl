# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load packages
using LinearAlgebra: LinearAlgebra, Diagonal, UniformScaling, I, diag, diagind, mul!, ldiv!, lu, lu!, norm
using SparseArrays: SparseArrays, sparse, issparse, dropzeros!
using Printf: @sprintf

using BracketingNonlinearSolve
using BracketingNonlinearSolve: AbstractBracketingAlgorithm
using SimpleNonlinearSolve
using SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm

using QuadGK: quadgk

using SummationByPartsOperators

using LaTeXStrings
using CairoMakie
set_theme!(theme_latexfonts();
           fontsize = 26,
           linewidth = 3,
           markersize = 16,
           Lines = (cycle = Cycle([:color, :linestyle], covary = true),),
           Scatter = (cycle = Cycle([:color, :marker], covary = true),))

using PrettyTables: PrettyTables, pretty_table


const FIGDIR = joinpath(dirname(@__DIR__), "figures")
if !isdir(FIGDIR)
    mkdir(FIGDIR)
end


#####################################################################
# Utility functions

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) /
                      log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end

function fit_order(Ns, errors)
    A = hcat(ones(length(Ns)), log.(Ns))
    c = A \ log.(errors)
    return -c[2]
end

change(x) = x .- first(x)
function absolute_change(x)
    y = abs.(x[(begin + 1):end] .- first(x))
    e = eps(eltype(y))
    @. y = max(y, e)
    return y
end



#####################################################################
# High-level interface of the equations and IMEX ode solver

rhs_stiff!(du, u, parameters, t) = rhs_stiff!(du, u, parameters.equation, parameters, t)
rhs_nonstiff!(du, u, parameters, t) = rhs_nonstiff!(du, u, parameters.equation, parameters, t)
operator(rhs_stiff!, parameters) = operator(rhs_stiff!, parameters.equation, parameters)
mass(u, parameters) = mass(u, parameters.equation, parameters)
energy(u, parameters) = energy(u, parameters.equation, parameters)


# IMEX Coefficients
"""
    ARS111(T = Float64)

First-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS111{T} end
ARS111(T = Float64) = ARS111{T}()
function coefficients(::ARS111{T}) where T
    l = one(T)

    A_stiff = [0 0;
               0 l]
    b_stiff = [0, l]
    c_stiff = [0, l]
    A_nonstiff = [0 0;
                  l 0]
    b_nonstiff = [l, 0]
    c_nonstiff = [0, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


"""
    ARS222(T = Float64)

Second-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS222{T} end
ARS222(T = Float64) = ARS222{T}()
function coefficients(::ARS222{T}) where T
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)
    δ = 1 - 1 / (2 * γ)

    A_stiff = [0 0 0;
               0 γ 0;
               0 1-γ γ]
    b_stiff = [0, 1-γ, γ]
    c_stiff = [0, γ, 1]
    A_nonstiff = [0 0 0;
                  γ 0 0;
                  δ 1-δ 0]
    b_nonstiff = [δ, 1-δ, 0]
    c_nonstiff = [0, γ, 1]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443(T = Float64)

Third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443{T} end
ARS443(T = Float64) = ARS443{T}()
function coefficients(::ARS443{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               0 l/2 0 0 0;
               0 l/6 l/2 0 0;
               0 -l/2 l/2 l/2 0;
               0 3*l/2 -3*l/2 l/2 l/2]
    b_stiff = [0, 3*l/2, -3*l/2, l/2, l/2]
    c_stiff = [0, l/2, 2*l/3, l/2, l]
    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443explicit(T = Float64)

Explicit part of the third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443explicit{T} end
ARS443explicit(T = Float64) = ARS443explicit{T}()
function coefficients(::ARS443explicit{T}) where T
    l = one(T)

    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]

    A_stiff = copy(A_nonstiff)
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    KenCarpARK437(T = Float64)

Fourth-order, type-II method of Kennedy and Carpenter (2019).
The implicit method is stiffly accurate.
"""
struct KenCarpARK437{T} end
KenCarpARK437(T = Float64) = KenCarpARK437{T}()
function coefficients(::KenCarpARK437{T}) where T
    γ = T(1235) / T(10_000)

    c2 = T(247) / T(1000)
    c3 = T(4276536705230) / T(10142255878289)
    c4 = T(67) / T(200)
    c5 = T(3) / T(40)
    c6 = T(7) / T(10)
    c7 = T(1)
    c_nonstiff = [0, c2, c3, c4, c5, c6, c7]
    b1 = T(0)
    b2 = b1
    b3 = T(9164257142617) / T(17756377923965)
    b4 = T(-10812980402763) / T(74029279521829)
    b5 = T(1335994250573) / T(5691609445217)
    b6 = T(2273837961795) / T(8368240463276)
    b7 = T(247) / T(2000)
    b_nonstiff = [b1, b2, b3, b4, b5, b6, b7]
    a21 = c2
    a31 = T(247) / T(4000)
    a32 = T(2694949928731) / T(7487940209513)
    a41 = T(464650059369) / T(8764239774964)
    a42 = T(878889893998) / T(2444806327765)
    a43 = T(-952945855348) / T(12294611323341)
    a51 = T(476636172619) / T(8159180917465)
    a52 = T(-1271469283451) / T(7793814740893)
    a53 = T(-859560642026) / T(4356155882851)
    a54 = T(1723805262919) / T(4571918432560)
    a61 = T(6338158500785) / T(11769362343261)
    a62 = T(-4970555480458) / T(10924838743837)
    a63 = T(3326578051521) / T(2647936831840)
    a64 = T(-880713585975) / T(1841400956686)
    a65 = T(-1428733748635) / T(8843423958496)
    a71 = T(760814592956) / T(3276306540349)
    a72 = a71
    a73 = T(-47223648122716) / T(6934462133451)
    a74 = T(71187472546993) / T(9669769126921)
    a75 = T(-13330509492149) / T(9695768672337)
    a76 = T(11565764226357) / T(8513123442827)
    A_nonstiff = [0 0 0 0 0 0 0;
                  a21 0 0 0 0 0 0;
                  a31 a32 0 0 0 0 0;
                  a41 a42 a43 0 0 0 0;
                  a51 a52 a53 a54 0 0 0;
                  a61 a62 a63 a64 a65 0 0;
                  a71 a72 a73 a74 a75 a76 0]
    @assert c_nonstiff ≈ sum(A_nonstiff, dims = 2)

    a21 = γ
    a31 = T(624185399699) / T(4186980696204)
    a32 = a31
    a41 = T(1258591069120) / T(10082082980243)
    a42 = a41
    a43 = T(-322722984531) / T(8455138723562)
    a51 = T(-436103496990) / T(5971407786587)
    a52 = a51
    a53 = T(-2689175662187) / T(11046760208243)
    a54 = T(4431412449334) / T(12995360898505)
    a61 = T(-2207373168298) / T(14430576638973)
    a62 = a61
    a63 = T(242511121179) / T(3358618340039)
    a64 = T(3145666661981) / T(7780404714551)
    a65 = T(5882073923981) / T(14490790706663)
    a71 = T(0)
    a72 = a71
    a73 = T(9164257142617) / T(17756377923965)
    a74 = T(-10812980402763) / T(74029279521829)
    a75 = T(1335994250573) / T(5691609445217)
    a76 = T(2273837961795) / T(8368240463276)
    A_stiff = [0 0 0 0 0 0 0;
               a21 γ 0 0 0 0 0;
               a31 a32 γ 0 0 0 0;
               a41 a42 a43 γ 0 0 0;
               a51 a52 a53 a54 γ 0 0;
               a61 a62 a63 a64 a65 γ 0;
               a71 a72 a73 a74 a75 a76 γ]
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)
    @assert c_stiff ≈ sum(A_stiff, dims = 2)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    KenCarpARK548(T = Float64)

Fifth-order, type-II method of Kennedy and Carpenter (2019).
The implicit method is stiffly accurate.
"""
struct KenCarpARK548{T} end
KenCarpARK548(T = Float64) = KenCarpARK548{T}()
function coefficients(::KenCarpARK548{T}) where T
    γ = T(2) / T(9)

    c2 = T(4) / T(9)
    c3 = T(6456083330201) / T(8509243623797)
    c4 = T(1632083962415) / T(14158861528103)
    c5 = T(6365430648612) / T(17842476412687)
    c6 = T(18) / T(25)
    c7 = T(191) / T(200)
    c8 = T(1)
    c_nonstiff = [0, c2, c3, c4, c5, c6, c7, c8]
    b1 = T(0)
    b2 = b1
    b3 = T(3517720773327) / T(20256071687669)
    b4 = T(4569610470461) / T(17934693873752)
    b5 = T(2819471173109) / T(11655438449929)
    b6 = T(3296210113763) / T(10722700128969)
    b7 = T(-1142099968913) / T(5710983926999)
    b8 = γ
    b_nonstiff = [b1, b2, b3, b4, b5, b6, b7, b8]
    a21 = c2
    a31 = T(1) / T(9)
    a32 = T(1183333538310) / T(1827251437969)
    a41 = T(895379019517) / T(9750411845327)
    a42 = T(477606656805) / T(13473228687314)
    a43 = T(-112564739183) / T(9373365219272)
    a51 = T(-4458043123994) / T(13015289567637)
    a52 = T(-2500665203865) / T(9342069639922)
    a53 = T(983347055801) / T(8893519644487)
    a54 = T(2185051477207) / T(2551468980502)
    a61 = T(-167316361917) / T(17121522574472)
    a62 = T(1605541814917) / T(7619724128744)
    a63 = T(991021770328) / T(13052792161721)
    a64 = T(2342280609577) / T(11279663441611)
    a65 = T(3012424348531) / T(12792462456678)
    a71 = T(6680998715867) / T(14310383562358)
    a72 = T(5029118570809) / T(3897454228471)
    a73 = T(2415062538259) / T(6382199904604)
    a74 = T(-3924368632305) / T(6964820224454)
    a75 = T(-4331110370267) / T(15021686902756)
    a76 = T(-3944303808049) / T(11994238218192)
    a81 = T(2193717860234) / T(3570523412979)
    a82 = a81
    a83 = T(5952760925747) / T(18750164281544)
    a84 = T(-4412967128996) / T(6196664114337)
    a85 = T(4151782504231) / T(36106512998704)
    a86 = T(572599549169) / T(6265429158920)
    a87 = T(-457874356192) / T(11306498036315)
    A_nonstiff = [0 0 0 0 0 0 0 0;
                  a21 0 0 0 0 0 0 0;
                  a31 a32 0 0 0 0 0 0;
                  a41 a42 a43 0 0 0 0 0;
                  a51 a52 a53 a54 0 0 0 0;
                  a61 a62 a63 a64 a65 0 0 0;
                  a71 a72 a73 a74 a75 a76 0 0;
                  a81 a82 a83 a84 a85 a86 a87 0]
    @assert c_nonstiff ≈ sum(A_nonstiff, dims = 2)

    a21 = γ
    a31 = T(2366667076620) / T(8822750406821)
    a32 = a31
    a41 = T(-257962897183) / T(4451812247028)
    a42 = a41
    a43 = T(128530224461) / T(14379561246022)
    a51 = T(-486229321650) / T(11227943450093)
    a52 = a51
    a53 = T(-225633144460) / T(6633558740617)
    a54 = T(1741320951451) / T(6824444397158)
    a61 = T(621307788657) / T(4714163060173)
    a62 = a61
    a63 = T(-125196015625) / T(3866852212004)
    a64 = T(940440206406) / T(7593089888465)
    a65 = T(961109811699) / T(6734810228204)
    a71 = T(2036305566805) / T(6583108094622)
    a72 = a71
    a73 = T(-3039402635899) / T(4450598839912)
    a74 = T(-1829510709469) / T(31102090912115)
    a75 = T(-286320471013) / T(6931253422520)
    a76 = T(8651533662697) / T(9642993110008)
    a81 = b1
    a82 = b2
    a83 = b3
    a84 = b4
    a85 = b5
    a86 = b6
    a87 = b7
    A_stiff = [0 0 0 0 0 0 0 0;
               a21 γ 0 0 0 0 0 0;
               a31 a32 γ 0 0 0 0 0;
               a41 a42 a43 γ 0 0 0 0;
               a51 a52 a53 a54 γ 0 0 0;
               a61 a62 a63 a64 a65 γ 0 0;
               a71 a72 a73 a74 a75 a76 γ 0;
               a81 a82 a83 a84 a85 a86 a87 γ]
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)
    @assert c_stiff ≈ sum(A_stiff, dims = 2)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

abstract type AbstractProjection end
"""
    NoProjection()

Do not use any relaxation/projection methods, just the baseline
time integration method.
"""
struct NoProjection <: AbstractProjection end
"""
    MassProjection()

Use a projection method that enforces mass conservation.
For the `CubicNLS` equation, this is the classical orthogonal
projection onto the manifold of constant mass. For the
`HyperbolizedCubicNLS` equation, it is a simplified projection
method in the orthogonal direction at the updated solution.
"""
struct MassProjection <: AbstractProjection end
"""
    FullRelaxation()

Use the new relaxation method that enforces both mass
and energy conservation by searching along a "geodesic line" of
constant mass connecting the previous and updated numerical solutions.
"""
struct FullRelaxation <: AbstractProjection end

# IMEX ARK solver
# This assumes that the stiff part is linear and that the stiff solver is
# diagonally implicit.
function solve_imex(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                    q0, tspan, parameters, alg;
                    dt,
                    relaxation::AbstractProjection = NoProjection(),
                    relaxation_alg = SimpleKlement(),
                    callback = Returns(nothing),
                    save_everystep = false)
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    if save_everystep
        sol_q = [copy(q0)]
        sol_t = [first(tspan)]
    end
    y = similar(q) # stage value
    z = similar(q) # stage update value
    t = first(tspan)
    tmp = similar(q)
    k_stiff_q = similar(q) # derivative of the previous state
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        if isnothing(a)
            factor = zero(dt)
        else
            factor = a * dt
        end
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        elseif W isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorization = W
        elseif W isa StiffOperatorCubicNLS
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorizations = nothing
        elseif factorization isa StiffOperatorCubicNLS
            factorizations = nothing
        else
            factorizations = Dict(factor => copy(factorization))
        end

        W, factorization, factorizations
    end

    # used for relaxation
    mass_old = mass(q, parameters)
    energy_old = energy(q, parameters)

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        # There are two possible formulations of a diagonally implicit RK method.
        # The "simple" one is
        #   y_i = q + h \sum_{j=1}^{i} a_{ij} f(y_j)
        # However, it can be better to use the smaller values
        #   z_i = (y_i - q) / h
        # so that the stage equations become
        #   q + h z_i = q + h \sum_{j=1}^{i} a_{ij} f(q + h z_j)
        # ⟺
        #   z_i - h a_{ii} f(q + h z_i) = \sum_{j=1}^{i-1} a_{ij} f(q + h z_j)
        # For a linear problem f(q) = T q, this becomes
        #   (I - h a_{ii} T z_i = a_{ii} T q + \sum_{j=1}^{i-1} a_{ij} T(q + h z_j)
        # We use this formulation and also avoid evaluating the stiff operator at
        # the numerical solutions (due to the large Lipschitz constant), but instead
        # rearrange the equation to obtain the required stiff RHS values as
        #   T(q + h z_i) = a_{ii}^{-1} (z_i - \sum_{j=1}^{i-1} a_{ij} f(q + h z_j))
        rhs_stiff!(k_stiff_q, q, parameters, t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i - 1)
                @. tmp += A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            # The right-hand side of the linear system formulated using the stages y_i
            # instead of the stage updates z_i would be
            #   @. tmp = q + dt * tmp
            # By using the stage updates z_i, we avoid the possibly different scales
            # for small dt.
            @. tmp = A_stiff[i, i] * k_stiff_q + tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(z, tmp)
            else
                factor = A_stiff[i, i] * dt

                if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
                    W = I - factor * rhs_stiff_operator
                    F = W
                    ldiv!(z, F, tmp)
                elseif factorization isa StiffOperatorCubicNLS
                    @assert parameters.equation isa CubicNLS
                    (; equation, D2, tmp1) = parameters
                    a = real(tmp, equation)
                    b = imag(tmp, equation)
                    z1 = real(z, equation)
                    z2 = imag(z, equation)
                    mul!(tmp1, D2, a)
                    @. tmp1 = b + factor * tmp1
                    ldiv!(z2, I + factor^2 * D2 * D2, tmp1)
                    mul!(tmp1, D2, z2)
                    @. z1 = a - factor * tmp1
                else
                    F = let W = W, factor = factor,
                            factorization = factorization,
                            rhs_stiff_operator = rhs_stiff_operator
                        get!(factorizations, factor) do
                            fill!(W, 0)
                            W[diagind(W)] .= 1
                            @. W -= factor * rhs_stiff_operator
                            if issparse(W)
                                lu!(factorization, W)
                            else
                                factorization = lu!(W)
                            end
                            copy(factorization)
                        end
                    end
                    ldiv!(z, F, tmp)
                end
            end

            # Compute new stage derivatives
            @. y = q + dt * z
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            else
                # The code below is equaivalent to
                #   rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
                # but avoids evaluating the stiff operator at the numerical solution.
                @. tmp = z
                for j in 1:(i-1)
                    @. tmp = tmp - A_stiff[i, j] * k_stiff[j] - A_nonstiff[i, j] * k_nonstiff[j]
                end
                @. k_stiff[i] = tmp / A_stiff[i, i]
            end
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        t = relaxation!(q, tmp, y, z, t, dt,
                        parameters.equation, parameters,
                        relaxation, relaxation_alg,
                        mass_old, energy_old)

        if save_everystep
            push!(sol_q, copy(q))
            append!(sol_t, t)
        end
        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    if save_everystep
        return (; u = sol_q,
                  t = sol_t)
    else
        return (; u = (q0, q),
                  t = (first(tspan), t))
    end
end



#####################################################################
# General interface

abstract type AbstractEquation end
Base.Broadcast.broadcastable(equation::AbstractEquation) = (equation,)


#####################################################################
# CubicNLS discretization

struct CubicNLS{T} <: AbstractEquation
    β::T
end

Base.real(q, equation::CubicNLS) = get_qi(q, equation, 0)
Base.imag(q, equation::CubicNLS) = get_qi(q, equation, 1)
function get_qi(q, equation::CubicNLS, i)
    N = length(q) ÷ 2
    return view(q, (i * N + 1):((i + 1) * N))
end

function density(q, equation::CubicNLS)
    v = real(q, equation)
    w = imag(q, equation)
    return @. v^2 + w^2
end

function rhs_stiff!(dq, q, equation::CubicNLS, parameters, t)
    (; D2) = parameters
    N = size(D2, 2)

    dv = view(dq, (0 * N + 1):(1 * N))
    dw = view(dq, (1 * N + 1):(2 * N))

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    (; β) = equation

    # dv = M^{-1} A_2 w = -D_2 w
    mul!(dv, D2, w)
    @. dv = -dv

    # dw = -M^{-1} A_2 v = D_2 v
    mul!(dw, D2, v)

    return nothing
end

struct StiffOperatorCubicNLS{TypeofD2}
    D2::TypeofD2
end
Base.:*(::Number, op::StiffOperatorCubicNLS) = op
Base.:-(::UniformScaling, op::StiffOperatorCubicNLS) = op
Base.iszero(op::StiffOperatorCubicNLS) = false

function operator(::typeof(rhs_stiff!), equation::CubicNLS, parameters)
    if parameters.D2 isa FourierPolynomialDerivativeOperator
        return StiffOperatorCubicNLS(parameters.D2)
    elseif parameters.D2 isa PeriodicUpwindOperators
        Dm = sparse(parameters.D2.minus)
        Dp = sparse(parameters.D2.plus)
        return Dm * Dp
    else
        D2 = sparse(parameters.D2)
        O = zero(D2)
        jac = [O -D2;
               D2 O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::CubicNLS, parameters, t)
    (; D2, tmp1) = parameters
    N = size(D2, 2)

    dv = view(dq, (0 * N + 1):(1 * N))
    dw = view(dq, (1 * N + 1):(2 * N))

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    (; β) = equation
    @. tmp1 = v^2 + w^2

    # dv = -β (v^2 + w^2) w
    @. dv = -β * tmp1 * w

    # dw = +β (v^2 + w^2) v
    @. dw = β * tmp1 * v

    return nothing
end

function mass(q, equation::CubicNLS, parameters)
    (; D2, tmp1) = parameters
    N = size(D2, 2)

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    @. tmp1 = v^2 + w^2

    return integrate(tmp1, D2)
end

function energy(q, equation::CubicNLS, parameters)
    (; D2, tmp1) = parameters
    N = size(D2, 2)

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    (; β) = equation

    # Kinetic energy
    mul!(tmp1, D2, v)
    @. tmp1 = -v * tmp1
    kinetic = integrate(tmp1, D2)

    mul!(tmp1, D2, w)
    @. tmp1 = -w * tmp1
    kinetic += integrate(tmp1, D2)

    # Potential energy
    @. tmp1 = (v^2 + w^2)^2
    potential = -0.5 * β * integrate(tmp1, D2)

    return kinetic + potential
end

function energy_naive(q, equation::CubicNLS, parameters)
    (; D2, tmp1) = parameters
    N = size(D2, 2)

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    (; β) = equation

    # Wrong kinetic energy discretization for Fourier operators
    mul!(tmp1, D2.D1, v)
    kinetic = integrate(abs2, tmp1, D2)
    mul!(tmp1, D2.D1, w)
    kinetic += integrate(abs2, tmp1, D2)

    # Potential energy
    @. tmp1 = (v^2 + w^2)^2
    potential = -0.5 * β * integrate(tmp1, D2)

    return kinetic + potential
end

function setup(u_func, equation::CubicNLS, tspan, D2)
    x = grid(D2)
    u0 = u_func.(tspan[begin], x, equation)
    v0 = real.(u0)
    w0 = imag.(u0)
    q0 = vcat(v0, w0)

    tmp1 = similar(v0)
    # for semidiscrete mass/energy rates
    dq1 = similar(q0)
    dq2 = similar(q0)

    parameters = (; equation, D2, tmp1, dq1, dq2)
    return (; q0, parameters)
end

function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::NoProjection, relaxation_alg,
                     mass_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::MassProjection, relaxation_alg,
                     mass_old, energy_old)
    @. q = q + dt * tmp
    mass_new = mass(q, parameters)
    q .= q .* sqrt(mass_old / mass_new)
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::FullRelaxation, relaxation_alg,
                     mass_old, energy_old)
    # First, we project the mass
    @. y = q + dt * tmp
    mass_new = mass(y, parameters)
    y .= y .* sqrt(mass_old / mass_new)

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    if relaxation_alg isa AbstractBracketingAlgorithm
        prob = IntervalNonlinearProblem{false}((0.8, 1.2)) do gamma, _
            @. z = (1 - gamma) * q + gamma * y
            mass_tmp = mass(z, parameters)
            z .= z .* sqrt(mass_old / mass_tmp)
            energy(z, parameters) - energy_old
        end
        sol = solve(prob, relaxation_alg)
        gamma = sol.u
    else
        prob = NonlinearProblem{false}(1.0) do gamma, _
            @. z = (1 - gamma) * q + gamma * y
            mass_tmp = mass(z, parameters)
            z .= z .* sqrt(mass_old / mass_tmp)
            energy(z, parameters) - energy_old
        end
        sol = solve(prob, relaxation_alg)
        gamma = sol.u
    end
    if !(gamma > 0.1)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = (1 - gamma) * q + gamma * y
    q .= z .* sqrt(mass_old / mass(z, parameters))
    return t + gamma * dt
end

one_soliton(t, x::Number, equation::CubicNLS) = cis(t) * sech(x)
get_β(::typeof(one_soliton)) = 2 * 1^2

function two_solitons(t, x::Number, equation::CubicNLS)
    num = 2 * exp(x + 9im * t) * (3 * exp(2x + 8im * t) + 3 * exp(4x + 8im * t) + exp(6x) + 1)
    den = 3 * exp(4x + 16im * t) + 4 * exp(2x + 8im * t) + 4 * exp(6x + 8im * t) + exp(8x + 8im * t) + exp(8im * t) + 3 * exp(4x)
    return num / den
end
get_β(::typeof(two_solitons)) = 2 * 2^2

function three_solitons(t, x::Number, equation::CubicNLS)
    # The appendix of Biswas and Ketcheson (2024) contains the following expression:
    # num = (80 * exp(7x + 49im * t)
    #        + 2 * exp(x + 25im * t)
    #        + 16 * exp(3x + 33im * t)
    #        + 36 * exp(5x + 33im * t)
    #        + 20 * exp(5x + 49im * t)
    #        + 32 * exp(7x + 25im * t)
    #        + 10 * exp(9x + 9im * t)
    #        + 90 * exp(9x + 41im * t)
    #        + 40 * exp(9x + 57im * t)
    #        + 32 * exp(11x + 25im * t)
    #        + 80 * exp(11x + 49im * t)
    #        + 32 * exp(13x + 33im * t)
    #        + 20 * exp(13x + 49im * t)
    #        + 16 * exp(15x + 33im * t)
    #        + 2 * exp(17x + 25im * t))
    # den = (64 * exp(12x + 24im * t)
    #        + 36 * exp(8x + 24im * t)
    #        + 18 * exp(4x + 16im * t)
    #        + 64 * exp(6x + 24im * t)
    #        + 45 * exp(10x + 40im * t)
    #        + 10 * exp(12x + 48im * t)
    #        + 45 * exp(8x + 40im * t)
    #        + 18 * exp(4x + 32im * t)
    #        + 10 * exp(6x + 48im * t)
    #        + 9 * exp(2x + 24im * t)
    #        + 45 * exp(8x + 8im * t)
    #        + 45 * exp(10x + 8im * t)
    #        + 36 * exp(10x + 24im * t)
    #        + 18 * exp(14x + 16im * t)
    #        + 18 * exp(14x + 32im * t)
    #        + 9 * exp(16x + 24im * t)
    #        + exp(18x + 24im * t)
    #        + exp(24im * t)
    #        + 10 * exp(6x)
    #        + 10 * exp(12x))
    # However, this is wrong since it is not even symmetric in x for t = 0.
    # Their reproducibility repository led me to the following version.
    num = (2*(3*exp(t*25*im)*exp(x) + 15*exp(t*9*im)*exp(9*x) + 48*exp(t*25*im)*exp(7*x) + 48*exp(t*25*im)*exp(11*x) + 24*exp(t*33*im)*exp(3*x) + 54*exp(t*33*im)*exp(5*x) + 3*exp(t*25*im)*exp(17*x) + 54*exp(t*33*im)*exp(13*x) + 24*exp(t*33*im)*exp(15*x) + 135*exp(t*41*im)*exp(9*x) + 30*exp(t*49*im)*exp(5*x) + 120*exp(t*49*im)*exp(7*x) + 120*exp(t*49*im)*exp(11*x) + 30*exp(t*49*im)*exp(13*x) + 60*exp(t*57*im)*exp(9*x)))
    den = (3*(exp(t*24*im) + 10*exp(6*x) + 10*exp(12*x) + 45*exp(t*8*im)*exp(8*x) + 45*exp(t*8*im)*exp(10*x) + 18*exp(t*16*im)*exp(4*x) + 9*exp(t*24*im)*exp(2*x) + 18*exp(t*16*im)*exp(14*x) + 64*exp(t*24*im)*exp(6*x) + 36*exp(t*24*im)*exp(8*x) + 36*exp(t*24*im)*exp(10*x) + 64*exp(t*24*im)*exp(12*x) + 18*exp(t*32*im)*exp(4*x) + 9*exp(t*24*im)*exp(16*x) + exp(t*24*im)*exp(18*x) + 18*exp(t*32*im)*exp(14*x) + 45*exp(t*40*im)*exp(8*x) + 45*exp(t*40*im)*exp(10*x) + 10*exp(t*48*im)*exp(6*x) + 10*exp(t*48*im)*exp(12*x)))
    return num / den
end
get_β(::typeof(three_solitons)) = 2 * 3^2


#####################################################################
# LDG operators and homogeneous Neumann boundary conditions
struct OperatorLDG{TypeofD, TypeofDsparse}
    D::TypeofD
    Dsparse::TypeofDsparse
end
function OperatorLDG(D::Union{UpwindOperators, PeriodicUpwindOperators})
    Dp = sparse(D.plus)
    M = mass_matrix(D)
    Dsparse = -inv(M) * Dp' * M * Dp
    return OperatorLDG(D, Dsparse)
end
Base.size(op::OperatorLDG) = size(op.D)
Base.size(op::OperatorLDG, i::Int) = size(op.D, i)
function SparseArrays.sparse(op::OperatorLDG)
    return copy(op.Dsparse)
end
SummationByPartsOperators.grid(op::OperatorLDG) = grid(op.D)
function SummationByPartsOperators.integrate(func, u::AbstractVector, op::OperatorLDG)
    return integrate(func, u, op.D)
end
function LinearAlgebra.mul!(out::AbstractVector, op::OperatorLDG, u::AbstractVector)
    mul!(out, op.Dsparse, u)
    return out
end


struct OperatorHomogeneousNeumann{T}
    D::T
end
Base.size(op::OperatorHomogeneousNeumann) = size(op.D)
Base.size(op::OperatorHomogeneousNeumann, i::Int) = size(op.D, i)
function SparseArrays.sparse(op::OperatorHomogeneousNeumann)
    D = sparse(op.D)

    dL = derivative_left(op.D, Val{1}())
    wL = left_boundary_weight(op.D)
    for i in eachindex(dL)
        di = dL[i]
        if !iszero(di)
            D[1, i] += di / wL
        end
    end

    dR = derivative_right(op.D, Val{1}())
    wR = right_boundary_weight(op.D)
    for i in eachindex(dR)
        di = dR[i]
        if !iszero(di)
            D[end, i] -= di / wR
        end
    end

    return D
end
SummationByPartsOperators.grid(op::OperatorHomogeneousNeumann) = grid(op.D)
function SummationByPartsOperators.integrate(func, u::AbstractVector, op::OperatorHomogeneousNeumann)
    return integrate(func, u, op.D)
end
function LinearAlgebra.mul!(out::AbstractVector, op::OperatorHomogeneousNeumann, u::AbstractVector)
    mul!(out, op.D, u)
    out[begin] += derivative_left(op.D, u, Val{1}()) / left_boundary_weight(op.D)
    out[end] -= derivative_right(op.D, u, Val{1}()) / right_boundary_weight(op.D)
    return out
end


#####################################################################
# HyperbolizedCubicNLS discretization
struct HyperbolizedCubicNLS{T} <: AbstractEquation
    β::T
    τ::T
end
function HyperbolizedCubicNLS(β, τ)
    β, τ = promote(β, τ)
    return HyperbolizedCubicNLS{typeof(β)}(β, τ)
end

Base.real(q, equation::HyperbolizedCubicNLS) = get_qi(q, equation, 0)
Base.imag(q, equation::HyperbolizedCubicNLS) = get_qi(q, equation, 1)
function get_qi(q, equation::HyperbolizedCubicNLS, i)
    N = length(q) ÷ 4
    return view(q, (i * N + 1):((i + 1) * N))
end

function density(q, equation::HyperbolizedCubicNLS)
    v = real(q, equation)
    w = imag(q, equation)
    return @. v^2 + w^2
end

function rhs_stiff!(dq, q, equation::HyperbolizedCubicNLS, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    dv = view(dq, (0 * N + 1):(1 * N))
    dw = view(dq, (1 * N + 1):(2 * N))
    dvx = view(dq, (2 * N + 1):(3 * N))
    dwx = view(dq, (3 * N + 1):(4 * N))

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))
    vx = view(q, (2 * N + 1):(3 * N))
    wx = view(q, (3 * N + 1):(4 * N))

    (; β, τ) = equation

    if D1 isa PeriodicUpwindOperators
        mul!(dv, D1.minus, wx)
        @. dv = -dv

        mul!(dw, D1.minus, vx)

        mul!(dvx, D1.plus, w)
        @. dvx = (dvx - wx) / τ

        mul!(dwx, D1.plus, v)
        @. dwx = (vx - dwx) / τ
    else
        error("Stiff operator not implemented for derivative operator $(typeof(D1))")
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::HyperbolizedCubicNLS, parameters)
    if parameters.D1 isa PeriodicUpwindOperators
        (; τ) = equation
        Dm = sparse(parameters.D1.minus)
        Dp = sparse(parameters.D1.plus)
        O = zero(Dm)
        jac = [O O O -Dm;
               O O Dm O;
               O Dp/τ O -I/τ;
               -Dp/τ O I/τ O]
        dropzeros!(jac)
        return jac
    else
        error("Stiff operator not implemented for derivative operator $(typeof(parameters.D1))")
    end
end

function rhs_nonstiff!(dq, q, equation::HyperbolizedCubicNLS, parameters, t)
    (; D1, tmp1) = parameters
    N = size(D1, 2)

    dv = view(dq, (0 * N + 1):(1 * N))
    dw = view(dq, (1 * N + 1):(2 * N))
    dvx = view(dq, (2 * N + 1):(3 * N))
    dwx = view(dq, (3 * N + 1):(4 * N))

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))

    (; β, τ) = equation
    @. tmp1 = v^2 + w^2

    # dv = -β (v^2 + w^2) w
    @. dv = -β * tmp1 * w

    # dw = +β (v^2 + w^2) v
    @. dw = β * tmp1 * v

    @. dvx = 0
    @. dwx = 0

    return nothing
end

function mass(q, equation::HyperbolizedCubicNLS, parameters)
    (; D1) = parameters
    N = size(D1, 2)

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))
    vx = view(q, (2 * N + 1):(3 * N))
    wx = view(q, (3 * N + 1):(4 * N))

    (; τ) = equation
    q2 = integrate(abs2, v, D1) + integrate(abs2, w, D1)
    p2 = integrate(abs2, vx, D1) + integrate(abs2, wx, D1)
    return q2 + τ * p2
end

function energy(q, equation::HyperbolizedCubicNLS, parameters)
    (; D1, tmp1) = parameters
    N = size(D1, 2)

    v = view(q, (0 * N + 1):(1 * N))
    w = view(q, (1 * N + 1):(2 * N))
    vx = view(q, (2 * N + 1):(3 * N))
    wx = view(q, (3 * N + 1):(4 * N))

    (; β) = equation

    # Kinetic energy
    mul!(tmp1, D1.plus, v)
    @. tmp1 = vx * tmp1
    kinetic = 2 * integrate(tmp1, D1)
    @. tmp1 = vx^2
    kinetic -= integrate(tmp1, D1)

    mul!(tmp1, D1.plus, w)
    @. tmp1 = wx * tmp1
    kinetic += 2 * integrate(tmp1, D1)
    @. tmp1 = wx^2
    kinetic -= integrate(tmp1, D1)

    # Potential energy
    @. tmp1 = (v^2 + w^2)^2
    potential = -0.5 * β * integrate(tmp1, D1)

    return kinetic + potential
end

function setup(u_func, equation::HyperbolizedCubicNLS, tspan, D1)
    x = grid(D1)
    u0 = u_func.(tspan[begin], x, CubicNLS(equation.β))
    v0 = real.(u0)
    w0 = imag.(u0)
    vx0 = D1.plus * v0
    wx0 = D1.plus * w0
    q0 = vcat(v0, w0, vx0, wx0)

    tmp1 = similar(v0)

    parameters = (; equation, D1, tmp1)
    return (; q0, parameters)
end

function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::NoProjection, relaxation_alg,
                     mass_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::MassProjection, relaxation_alg,
                     mass_old, energy_old)
    # Simplified projection method using the gradient/direction of
    # the current value
    @. q = q + dt * tmp
    project_mass!(q, equation, parameters, mass_old)
    return t + dt
end
function project_mass!(q, equation, parameters, mass_old)
    (; τ) = equation
    (; D1) = parameters
    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    q2 = integrate(abs2, v, D1) + integrate(abs2, w, D1)
    p2 = integrate(abs2, vx, D1) + integrate(abs2, wx, D1)
    c = mass_old
    factor_q = (p2 * (τ - 1) * τ^2 + sqrt(-p2 * q2 * (τ - 1)^2 * τ + c * (q2 + p2 * τ^3))) / (q2 + p2 * τ^3)
    factor_p = (q2 * (1 - τ) + τ * sqrt(-p2 * q2 * (τ - 1)^2 * τ + c * (q2 + p2 * τ^3))) / (q2 + p2 * τ^3)
    @. v = factor_q * v
    @. w = factor_q * w
    @. vx = factor_p * vx
    @. wx = factor_p * wx

    return nothing
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::FullRelaxation, relaxation_alg,
                     mass_old, energy_old)
    # First, we project the mass
    @. y = q + dt * tmp
    project_mass!(y, equation, parameters, mass_old)

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    if relaxation_alg isa AbstractBracketingAlgorithm
        prob = IntervalNonlinearProblem{false}((0.8, 1.2)) do gamma, _
            @. z = (1 - gamma) * q + gamma * y
            project_mass!(z, equation, parameters, mass_old)
            energy(z, parameters) - energy_old
        end
        sol = solve(prob, relaxation_alg)
        gamma = sol.u
    else
        prob = NonlinearProblem{false}(1.0) do gamma, _
            @. z = (1 - gamma) * q + gamma * y
            project_mass!(z, equation, parameters, mass_old)
            energy(z, parameters) - energy_old
        end
        sol = solve(prob, relaxation_alg)
        gamma = sol.u
    end
    @. z = (1 - gamma) * q + gamma * y
    project_mass!(z, equation, parameters, mass_old)
    @. q = z
    return t + gamma * dt
end



#####################################################################
# Numerical experiments reported in the paper

function kinetic_energy_comparison(; tspan = (0.0, 5.0),
                                     alg = KenCarpARK548(),
                                     dt = 1.0e-4,
                                     initial_condition = two_solitons,
                                     β = get_β(initial_condition), # 2 n^2 for n solitons
                                     kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0
    equation = CubicNLS(β)

    fig = Figure(size = (1200, 400)) # default size is (600, 450)

    for (N, column) in ((2^6, 1), (2^6 + 1, 2))
        D = fourier_derivative_operator(xmin, xmax, N)
        D2 = D^2

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        series_energy_naive = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy, series_energy_naive = series_energy_naive
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))
                push!(series_energy_naive, energy_naive(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        ax_invariants = Axis(fig[1, column];
                            xlabel = L"Time $t$",
                            ylabel = "Abs. change of invariants",
                            title = "$(N) nodes",
                            xscale = log10, yscale = log10)
        lines!(ax_invariants, absolute_change(series_t), absolute_change(series_mass); label = "mass")
        lines!(ax_invariants, absolute_change(series_t), absolute_change(series_energy); label = "energy (correct)")
        lines!(ax_invariants, absolute_change(series_t), absolute_change(series_energy_naive); label = "energy (naive)")

        if column == 1
            axislegend(ax_invariants; position = :rc, framevisible = false)
        else
            axislegend(ax_invariants; position = :lt, framevisible = false)
        end
    end

    filename = joinpath(FIGDIR, "kinetic_energy_comparison.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function semidiscrete_conservation(; tspan = (0.0, 5.0),
                                     alg = KenCarpARK548(),
                                     dt = 1.0e-4,
                                     initial_condition = two_solitons,
                                     β = get_β(initial_condition), # 2 n^2 for n solitons
                                     kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0
    equation = CubicNLS(β)

    global_dofs = 2^6

    fig = Figure(size = (1200, 800)) # default size is (600, 450)

    @info "Periodic FD"
    ax_fd = Axis(fig[1, 1];
                 xlabel = L"Time $t$",
                 ylabel = "Abs. change of invariants",
                 title = "Periodic FD",
                 xscale = log10, yscale = log10)
    for accuracy_order in 2:2:6
        N = global_dofs
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order, xmin, xmax, N)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_fd, absolute_change(series_t), absolute_change(series_mass); label = "mass, order $(accuracy_order)")
        lines!(ax_fd, absolute_change(series_t), absolute_change(series_energy); label = "energy, order $(accuracy_order)")
    end
    axislegend(ax_fd; position = :lt, framevisible = false)


    @info "Periodic CG"
    ax_cg = Axis(fig[1, 2];
                 xlabel = L"Time $t$",
                 ylabel = "Abs. change of invariants",
                 title = "Periodic CG",
                 xscale = log10, yscale = log10)
    for p in 2:4
        D2_local = legendre_second_derivative_operator(-1.0, 1.0, p + 1)
        N = global_dofs ÷ p
        mesh = UniformPeriodicMesh1D(xmin, xmax, N)
        D2 = couple_continuously(D2_local, mesh)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_cg, absolute_change(series_t), absolute_change(series_mass); label = L"mass, $p = %$(p)$")
        lines!(ax_cg, absolute_change(series_t), absolute_change(series_energy); label = L"energy, $p = %$(p)$")
    end
    axislegend(ax_cg; position = :lt, framevisible = false)


    @info "Periodic LDG"
    ax_dg = Axis(fig[1, 3];
                 xlabel = L"Time $t$",
                 ylabel = "Abs. change of invariants",
                 title = "Periodic LDG",
                 xscale = log10, yscale = log10)
    for p in 2:4
        D1_local = legendre_derivative_operator(-1.0, 1.0, p + 1)
        N = global_dofs ÷ (p + 1)
        mesh = UniformPeriodicMesh1D(xmin, xmax, N)
        D = upwind_operators(couple_discontinuously, D1_local, mesh)
        D2 = OperatorLDG(D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_dg, absolute_change(series_t), absolute_change(series_mass); label = L"mass, $p = %$(p)$")
        lines!(ax_dg, absolute_change(series_t), absolute_change(series_energy); label = L"energy, $p = %$(p)$")
    end
    axislegend(ax_dg; position = :lt, framevisible = false)


    linkyaxes!(ax_fd, ax_cg, ax_dg)
    hideydecorations!(ax_cg; grid = false)
    hideydecorations!(ax_dg; grid = false)


    @info "Nonperiodic FD"
    ax_fd_neumann = Axis(fig[2, 1];
                         xlabel = L"Time $t$",
                         ylabel = "Abs. change of invariants",
                         title = "Nonperiodic FD",
                         xscale = log10, yscale = log10)
    for accuracy_order in 2:2:6
        N = global_dofs
        D = derivative_operator(MattssonNordström2004();
                                derivative_order = 2,
                                accuracy_order, xmin, xmax, N)
        D2 = OperatorHomogeneousNeumann(D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_fd_neumann, absolute_change(series_t), absolute_change(series_mass); label = "mass, order $(accuracy_order)")
        lines!(ax_fd_neumann, absolute_change(series_t), absolute_change(series_energy); label = "energy, order $(accuracy_order)")
    end
    axislegend(ax_fd_neumann; position = :lt, framevisible = false)


    @info "Nonperiodic CG"
    ax_cg_neumann = Axis(fig[2, 2];
                         xlabel = L"Time $t$",
                         ylabel = "Abs. change of invariants",
                         title = "Periodic CG",
                         xscale = log10, yscale = log10)
    for p in 2:4
        D2_local = legendre_second_derivative_operator(-1.0, 1.0, p + 1)
        N = global_dofs ÷ p
        mesh = UniformMesh1D(xmin, xmax, N)
        D = couple_continuously(D2_local, mesh)
        D2 = OperatorHomogeneousNeumann(D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_cg_neumann, absolute_change(series_t), absolute_change(series_mass); label = L"mass, $p = %$(p)$")
        lines!(ax_cg_neumann, absolute_change(series_t), absolute_change(series_energy); label = L"energy, $p = %$(p)$")
    end
    axislegend(ax_cg_neumann; position = :lt, framevisible = false)


    @info "Nonperiodic LDG"
    ax_dg_neumann = Axis(fig[2, 3];
                         xlabel = L"Time $t$",
                         ylabel = "Abs. change of invariants",
                         title = "Nonperiodic LDG",
                         xscale = log10, yscale = log10)
    for p in 2:4
        D1_local = legendre_derivative_operator(-1.0, 1.0, p + 1)
        N = global_dofs ÷ (p + 1)
        mesh = UniformMesh1D(xmin, xmax, N)
        D = upwind_operators(couple_discontinuously, D1_local, mesh)
        D2 = OperatorLDG(D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, callback, kwargs...)

        lines!(ax_dg_neumann, absolute_change(series_t), absolute_change(series_mass); label = L"mass, $p = %$(p)$")
        lines!(ax_dg_neumann, absolute_change(series_t), absolute_change(series_energy); label = L"energy, $p = %$(p)$")
    end
    axislegend(ax_dg_neumann; position = :lt, framevisible = false)


    linkyaxes!(ax_fd_neumann, ax_cg_neumann, ax_dg_neumann)
    hideydecorations!(ax_cg_neumann; grid = false)
    hideydecorations!(ax_dg_neumann; grid = false)


    linkxaxes!(ax_fd, ax_fd_neumann)
    hidexdecorations!(ax_fd; grid = false)
    linkxaxes!(ax_cg, ax_cg_neumann)
    hidexdecorations!(ax_cg; grid = false)
    linkxaxes!(ax_dg, ax_dg_neumann)
    hidexdecorations!(ax_dg; grid = false)


    filename = joinpath(FIGDIR, "semidiscrete_conservation.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function convergence_in_space(; tspan = (0.0, 1.0),
                                alg = KenCarpARK548(),
                                dt = 5.0e-5,
                                initial_condition = two_solitons,
                                β = get_β(initial_condition), # 2 n^2 for n solitons
                                kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0
    equation = CubicNLS(β)

    fig = Figure(size = (1200, 800)) # default size is (600, 450)

    @info "Periodic FD"
    ax_fd = Axis(fig[1, 1];
                 xlabel = "Number of degrees of freedom",
                 ylabel = L"Error at $t = %$(tspan[end])$",
                 title = "Periodic FD",
                 xscale = log10, yscale = log10)
    for accuracy_order in 2:2:6
        num_nodes = 2 .^ (9:12)
        final_errors = Float64[]
        for N in num_nodes
            D2 = periodic_derivative_operator(derivative_order = 2;
                                              accuracy_order,
                                              xmin, xmax, N)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_fd, num_nodes, final_errors; label = "order $(accuracy_order)")

        o = fit_order(num_nodes, final_errors)
        fit = num_nodes .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_fd, num_nodes, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_fd; position = :lb, framevisible = false, nbanks = 2)


    @info "Periodic CG"
    ax_cg = Axis(fig[1, 2];
                 xlabel = "Number of elements",
                 ylabel = L"Error at $t = %$(tspan[end])$",
                 title = "Periodic CG",
                 xscale = log10, yscale = log10)
    for p in 2:4
        num_elements = 2 .^ (7:10)
        final_errors = Float64[]

        for N in num_elements
            D2_local = legendre_second_derivative_operator(-1.0, 1.0, p + 1)
            mesh = UniformPeriodicMesh1D(xmin, xmax, N)
            D2 = couple_continuously(D2_local, mesh)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_cg, num_elements, final_errors; label = L"p = %$(p)")

        o = fit_order(num_elements, final_errors)
        fit = num_elements .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_cg, num_elements, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_cg; position = :lb, framevisible = false, nbanks = 2)


    @info "Periodic LDG"
    ax_dg = Axis(fig[1, 3];
                 xlabel = "Number of elements",
                 ylabel = L"Error at $t = %$(tspan[end])$",
                 title = "Periodic LDG",
                 xscale = log10, yscale = log10)
    for p in 2:4
        num_elements = 2 .^ (7:10)
        final_errors = Float64[]

        for N in num_elements
            D1_local = legendre_derivative_operator(-1.0, 1.0, p + 1)
            mesh = UniformPeriodicMesh1D(xmin, xmax, N)
            D = upwind_operators(couple_discontinuously, D1_local, mesh)
            D2 = OperatorLDG(D)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_dg, num_elements, final_errors; label = L"p = %$(p)")

        o = fit_order(num_elements, final_errors)
        fit = num_elements .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_dg, num_elements, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_dg; position = :lb, framevisible = false, nbanks = 2)


    linkyaxes!(ax_fd, ax_cg, ax_dg)
    hideydecorations!(ax_cg; grid = false)
    hideydecorations!(ax_dg; grid = false)


    @info "Nonperiodic FD"
    ax_fd_neumann = Axis(fig[2, 1];
                         xlabel = "Number of degrees of freedom",
                         ylabel = L"Error at $t = %$(tspan[end])$",
                         title = "Nonperiodic FD",
                         xscale = log10, yscale = log10)
    for accuracy_order in 2:2:6
        num_nodes = 2 .^ (9:12)
        final_errors = Float64[]
        for N in num_nodes
            D = derivative_operator(MattssonNordström2004();
                                    derivative_order = 2,
                                    accuracy_order, xmin, xmax, N)
            D2 = OperatorHomogeneousNeumann(D)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_fd_neumann, num_nodes, final_errors; label = "order $(accuracy_order)")

        o = fit_order(num_nodes, final_errors)
        fit = num_nodes .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_fd_neumann, num_nodes, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_fd_neumann; position = :lb, framevisible = false, nbanks = 2)


    @info "Nonperiodic CG"
    ax_cg_neumann = Axis(fig[2, 2];
                         xlabel = "Number of elements",
                         ylabel = L"Error at $t = %$(tspan[end])$",
                         title = "Periodic CG",
                         xscale = log10, yscale = log10)
    for p in 2:4
        num_elements = 2 .^ (7:10)
        final_errors = Float64[]

        for N in num_elements
            D2_local = legendre_second_derivative_operator(-1.0, 1.0, p + 1)
            mesh = UniformMesh1D(xmin, xmax, N)
            D = couple_continuously(D2_local, mesh)
            D2 = OperatorHomogeneousNeumann(D)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_cg_neumann, num_elements, final_errors; label = L"p = %$(p)")

        o = fit_order(num_elements, final_errors)
        fit = num_elements .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_cg_neumann, num_elements, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_cg_neumann; position = :lb, framevisible = false, nbanks = 2)


    @info "Nonperiodic LDG"
    ax_dg_neumann = Axis(fig[2, 3];
                         xlabel = "Number of elements",
                         ylabel = L"Error at $t = %$(tspan[end])$",
                         title = "Nonperiodic LDG",
                         xscale = log10, yscale = log10)
    for p in 2:4
        num_elements = 2 .^ (7:10)
        final_errors = Float64[]

        for N in num_elements
            D1_local = legendre_derivative_operator(-1.0, 1.0, p + 1)
            mesh = UniformMesh1D(xmin, xmax, N)
            D = upwind_operators(couple_discontinuously, D1_local, mesh)
            D2 = OperatorLDG(D)

            (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, kwargs...)

            (; tmp1) = parameters
            x = grid(D2)
            v = real(sol.u[end], equation)
            w = imag(sol.u[end], equation)
            t = sol.t[end]
            tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
            tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
            push!(final_errors, sqrt(integrate(tmp1, D2)))
        end

        scatter!(ax_dg_neumann, num_elements, final_errors; label = L"p = %$(p)")

        o = fit_order(num_elements, final_errors)
        fit = num_elements .^ (-o)
        @. fit = final_errors[end] / fit[end] * fit
        lines!(ax_dg_neumann, num_elements, fit; label = @sprintf("slope %.1f", o))
    end
    axislegend(ax_dg_neumann; position = :lb, framevisible = false, nbanks = 2)


    linkyaxes!(ax_fd_neumann, ax_cg_neumann, ax_dg_neumann)
    hideydecorations!(ax_cg_neumann; grid = false)
    hideydecorations!(ax_dg_neumann; grid = false)


    linkxaxes!(ax_fd, ax_fd_neumann)
    hidexdecorations!(ax_fd; grid = false)
    linkxaxes!(ax_cg, ax_cg_neumann)
    hidexdecorations!(ax_cg; grid = false)
    linkxaxes!(ax_dg, ax_dg_neumann)
    hidexdecorations!(ax_dg; grid = false)


    filename = joinpath(FIGDIR, "convergence_in_space.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function riemann_data(t, x::Number, equation::CubicNLS)
    # Parameters of the dispersive Riemann problem
    rho_l = 2.0
    rho_r = 1.0

    rho = 0.5 * (rho_l + rho_r) + 0.5 * (rho_r - rho_l) * tanh(100 * x)
    theta = 0.0

    return sqrt(rho) * cis(theta)
end
get_β(::typeof(riemann_data)) = -1

function dispersive_shock_wave(; alg = KenCarpARK548(),
                                 tspan = (0.0, 100.0),
                                 dt = 0.05,
                                 N = 2^16,
                                 kwargs...)
    fig = Figure(size = (1200, 600)) # default size is (600, 450)

    @info "Dispersive shock wave"
    initial_condition = riemann_data
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    xmin = -1600.0
    xmax = +1600.0
    ax_rho = Axis(fig[1, 1];
                  xlabel = L"$\xi = x / t$ at time $t = %$(round(Int, last(tspan)))$",
                  ylabel = L"Mass density $\varrho = |u|^2$")
    ax_v = Axis(fig[2, 1];
                xlabel = L"$\xi = x / t$ at time $t = %$(round(Int, last(tspan)))$",
                ylabel = L"Velocity $v = \theta_x$")
    D2 = fourier_derivative_operator(xmin, xmax, N)^2

    (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                            rhs_nonstiff!,
                            q0, tspan, parameters, alg;
                            dt, kwargs...)

    x = grid(D2)
    idx = -400 .< x .< 500
    xi = x[idx] ./ sol.t[end]
    abs2_u = density(sol.u[end], equation)[idx]
    lines!(ax_rho, xi, abs2_u)
    xlims!(ax_rho, extrema(xi))
    xlims!(ax_v, extrema(xi))

    # Compute the angle theta and make it continuous
    v = real(sol.u[end], equation)
    w = imag(sol.u[end], equation)
    theta = atan.(w, v)
    for i in eachindex(theta)[begin:(end - 1)]
        if theta[i] > theta[i + 1] + 1.9 * π
            theta[(i + 1):end] .+= 2 * π
        end
    end
    # To remove the factor 1/2 of Dhaouadie, Favrie, Gavrilyuk (2018)
    v = diff(theta) * sqrt(2) / step(x)
    v = v[idx[begin:(end - 1)]]
    lines!(ax_v, xi, v)


    # Parameters of the dispersive Riemann problem
    rho_l = 2.0
    rho_r = 1.0
    v_l = 0.0
    v_r = 0.0
    rho_0 = 0.25 * (sqrt(rho_r) + sqrt(rho_l) + 0.5 * (v_l - v_r))^2
    v_0 = 0.5 * (v_l + v_r) + sqrt(rho_l) - sqrt(rho_r)
    xi_1 = v_r + (8 * rho_0 - 8 * sqrt(rho_0 * rho_r) + rho_r) / (2 * sqrt(rho_0) - rho_r)
    xi_2 = v_r + sqrt(rho_0)
    xi_3 = v_0 - sqrt(rho_0)
    xi_4 = v_l - sqrt(rho_l)
    # To remove the factor 1/2 of Dhaouadie, Favrie, Gavrilyuk (2018)
    xticks = [xi_1, xi_2, xi_3, xi_4] * sqrt(2)
    xticklabels = [L"\xi_1", L"\xi_2", L"\xi_3", L"\xi_4"]
    for ax in (ax_rho, ax_v)
        ax.xticks = (xticks, xticklabels)
    end
    let ax = ax_rho
        yticks = [rho_l, rho_0, rho_r]
        yticklabels = [L"\varrho_L", L"\varrho_0", L"\varrho_R"]
        ax.yticks = (yticks, yticklabels)
    end
    let ax = ax_v
        yticks = [v_l, v_0]
        yticklabels = [L"v_L", L"v_0"]
        ax.yticks = (yticks, yticklabels)
    end


    linkxaxes!(ax_rho, ax_v)
    hidexdecorations!(ax_rho; grid = false)


    filename = joinpath(FIGDIR, "dispersive_shock_wave.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function fully_discrete_conservation(; tspan = (0.0, 4.3),
                                       alg = ARS443(),
                                       dt2 = 1.0e-2,
                                       dt3 = 1.0e-3,
                                       N = 2^10,
                                       kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0

    fig = Figure(size = (1200, 750)) # default size is (600, 450)

    initial_condition = two_solitons
    β = get_β(initial_condition) # 2 n^2 for n solitons
    equation = CubicNLS(β)

    D = fourier_derivative_operator(xmin, xmax, N)
    D2 = D^2

    ax2_sol = Axis(fig[1, 1];
                   xlabel = L"Space $x$",
                   ylabel = L"Mass density $|u|^2$",
                   title = "Two solitons at time $(tspan[end])")
    lines!(ax2_sol, grid(D2), abs2.(initial_condition.(tspan[end], grid(D2), equation)); label = "exact", linestyle = :dot, color = :gray)

    ax2_invariants = Axis(fig[1, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Two solitons")
    ax2_invariants_inset = Axis(fig[1, 2],
                                width = Relative(0.4),
                                height = Relative(0.3),
                                halign = 0.99,
                                valign = 0.23,
                                title = "Zoom")

    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt2, relaxation, callback, kwargs...)

        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax2_sol, grid(D2), density(sol.u[end], equation); label)

        lines!(ax2_invariants, series_t, change(series_energy); label = label * ": energy")
        lines!(ax2_invariants, series_t, change(series_mass); label = label * ": mass")

        lines!(ax2_invariants_inset, series_t, change(series_energy))
        lines!(ax2_invariants_inset, series_t, change(series_mass))
    end

    axislegend(ax2_sol; position = :lt, framevisible = false)
    xlims!(ax2_sol, -8, 8)

    axislegend(ax2_invariants; position = :lt, framevisible = false)

    ylims!(ax2_invariants_inset, -4.0e-13, 4.0e-13)
    hidexdecorations!(ax2_invariants_inset; grid = false)


    initial_condition = three_solitons
    β = get_β(initial_condition) # 2 n^2 for n solitons
    equation = CubicNLS(β)

    D = fourier_derivative_operator(xmin, xmax, N)
    D2 = D^2

    ax3_sol = Axis(fig[2, 1];
                   xlabel = L"Space $x$",
                   ylabel = L"Mass density $|u|^2$",
                   title = "Three solitons at time $(tspan[end])")
    lines!(ax3_sol, grid(D2), abs2.(initial_condition.(tspan[end], grid(D2), equation)); label = "exact", linestyle = :dot, color = :gray)

    ax3_invariants = Axis(fig[2, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Three solitons")
    ax3_invariants_inset = Axis(fig[2, 2],
                                width = Relative(0.4),
                                height = Relative(0.3),
                                halign = 0.99,
                                valign = 0.15,
                                title = "Zoom")

    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt3, relaxation, callback, kwargs...)

        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax3_sol, grid(D2), density(sol.u[end], equation); label)

        lines!(ax3_invariants, series_t, change(series_energy); label = label * ": energy")
        lines!(ax3_invariants, series_t, change(series_mass); label = label * ": mass")

        lines!(ax3_invariants_inset, series_t, change(series_energy))
        lines!(ax3_invariants_inset, series_t, change(series_mass))
    end

    axislegend(ax3_sol; position = :lt, framevisible = false)
    xlims!(ax3_sol, -8, 8)

    axislegend(ax3_invariants; position = :lt, framevisible = false)

    ylims!(ax3_invariants_inset, -4.0e-13, 4.0e-13)
    hidexdecorations!(ax3_invariants_inset; grid = false)


    linkxaxes!(ax2_sol, ax3_sol)
    hidexdecorations!(ax2_sol; grid = false)
    linkxaxes!(ax2_invariants, ax3_invariants)
    hidexdecorations!(ax2_invariants; grid = false)


    colgap!(fig.layout, 0)


    filename = joinpath(FIGDIR, "fully_discrete_conservation.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function convergence_in_time(; tspan = (0.0, 1.0),
                               kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0

    algorithms = [
        (ARS443(), "ARS3"),         # ARS(4,4,3)
        (KenCarpARK437(), "KC4"), # ARK4(3)7L[2]SA₁
        (KenCarpARK548(), "KC5")  # ARK5(4)8L[2]SA₂
    ]

    fig = Figure(size = (1200, 550)) # default size is (600, 450)

    @info "Two solitons"
    initial_condition = two_solitons
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    ax2 = Axis(fig[1, 1];
               xlabel = L"Time step size $\Delta t$",
               ylabel = L"Error at $t = %$(tspan[end])$",
               title = "Two solitons",
               xscale = log10, yscale = log10)
    D2 = fourier_derivative_operator(xmin, xmax, 2^12)^2
    for (alg, alg_name) in algorithms
        @show alg_name
        for relaxation in (NoProjection(), FullRelaxation())
            @show relaxation
            dts = 2.0 .^ range(-6, -12, length = 7)
            final_errors = Float64[]
            for dt in dts
                (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
                @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                       rhs_nonstiff!,
                                       q0, tspan, parameters, alg;
                                       dt, relaxation,
                                       kwargs...)

                (; tmp1) = parameters
                x = grid(D2)
                v = real(sol.u[end], equation)
                w = imag(sol.u[end], equation)
                t = sol.t[end]
                tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
                tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
                push!(final_errors, sqrt(integrate(tmp1, D2)))
            end

            label = alg_name * (relaxation isa NoProjection ? "" : "+R")
            scatter!(ax2, dts, final_errors; label = label)

            o = fit_order(dts[2:end], final_errors[2:end])
            fit = dts .^ (-o)
            @. fit = final_errors[end] / fit[end] * fit
            lines!(ax2, dts, fit; label = @sprintf("slope %.1f", -o))
        end
    end
    axislegend(ax2; position = :rb, framevisible = false, nbanks = 2, colgap = 5)
    ylims!(ax2, high = 3.0)


    @info "Three solitons"
    initial_condition = three_solitons
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    ax3 = Axis(fig[1, 2];
               xlabel = L"Time step size $\Delta t$",
               ylabel = L"Error at $t = %$(tspan[end])$",
               title = "Three solitons",
               xscale = log10, yscale = log10)
    D2 = fourier_derivative_operator(xmin, xmax, 2^12)^2
    for (alg, alg_name) in algorithms
        @show alg_name
        for relaxation in (NoProjection(), FullRelaxation())
            @show relaxation
            dts = 2.0 .^ range(-8, -13, length = 6)
            final_errors = Float64[]
            for dt in dts
                (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
                @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                       rhs_nonstiff!,
                                       q0, tspan, parameters, alg;
                                       dt, relaxation,
                                       kwargs...)

                (; tmp1) = parameters
                x = grid(D2)
                v = real(sol.u[end], equation)
                w = imag(sol.u[end], equation)
                t = sol.t[end]
                tmp1 .= (v .- real.(initial_condition.(t, x, equation))).^2
                tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
                push!(final_errors, sqrt(integrate(tmp1, D2)))
            end

            label = alg_name * (relaxation isa NoProjection ? "" : "+R")
            scatter!(ax3, dts, final_errors; label = label)

            o = fit_order(dts[2:end], final_errors[2:end])
            fit = dts .^ (-o)
            @. fit = final_errors[end] / fit[end] * fit
            lines!(ax3, dts, fit; label = @sprintf("slope %.1f", -o))
        end
    end
    axislegend(ax3; position = :rb, framevisible = false, nbanks = 2, colgap = 5)
    ylims!(ax3, high = 3.0)


    linkyaxes!(ax2, ax3)
    hideydecorations!(ax3; grid = false)


    filename = joinpath(FIGDIR, "convergence_in_time.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function error_growth(; alg = KenCarpARK548(),
                        tspan = (0.0, 100.0),
                        dt2 = 1.0e-2,
                        N2 = 2^10,
                        dt3 = 2.0e-3,
                        N3 = 2^10,
                        kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0

    fig = Figure(size = (1200, 400)) # default size is (600, 450)

    @info "Two solitons"
    initial_condition = two_solitons
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    ax2 = Axis(fig[1, 1];
               xlabel = L"Time $t$",
               ylabel = L"Error at time $t$",
               title = "Two solitons",
               xscale = log10, yscale = log10)
    D2 = fourier_derivative_operator(xmin, xmax, N2)^2
    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_tmp1 = parameters.tmp1
                p_D2 = parameters.D2

                push!(series_t, t)

                v = real(q, parameters.equation)
                w = imag(q, parameters.equation)

                x = grid(p_D2)
                for i in eachindex(x, p_tmp1)
                    ic = initial_condition(t, x[i], parameters.equation)
                    p_tmp1[i] = (v[i] - real(ic))^2 + (w[i] - imag(ic))^2
                end
                push!(series_error, sqrt(integrate(p_tmp1, p_D2)))

                return nothing
            end
        end
        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt2, relaxation, callback, kwargs...)
        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax2, series_t, series_error; label = label)
    end
    t = [1.0, 1.0e2]
    lines!(ax2, t, t.^2 .* 1.5e-3; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
    lines!(ax2, t, t .* 3.0e-5; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
    axislegend(ax2; position = :rb, framevisible = false, nbanks = 2, colgap = 5)


    @info "Three solitons"
    initial_condition = three_solitons
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    ax3 = Axis(fig[1, 2];
               xlabel = L"Time $t$",
               ylabel = L"Error at time $t$",
               title = "Three solitons",
               xscale = log10, yscale = log10)
    D2 = fourier_derivative_operator(xmin, xmax, N3)^2
    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_tmp1 = parameters.tmp1
                p_D2 = parameters.D2

                push!(series_t, t)

                v = real(q, parameters.equation)
                w = imag(q, parameters.equation)

                x = grid(p_D2)
                for i in eachindex(x, p_tmp1)
                    ic = initial_condition(t, x[i], parameters.equation)
                    p_tmp1[i] = (v[i] - real(ic))^2 + (w[i] - imag(ic))^2
                end
                push!(series_error, sqrt(integrate(p_tmp1, p_D2)))

                return nothing
            end
        end
        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt3, relaxation, callback, kwargs...)
        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax3, series_t, series_error; label = label)
    end
    t = [1.0, 1.0e2]
    lines!(ax3, t, t.^2 .* 7.0e-3; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
    lines!(ax3, t, t .* 2.0e-4; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
    axislegend(ax3; position = :rb, framevisible = false, nbanks = 2, colgap = 5)


    linkyaxes!(ax2, ax3)
    hideydecorations!(ax3; grid = false)


    filename = joinpath(FIGDIR, "error_growth.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function gray_soliton(t, x::Number, equation::CubicNLS)
    # Parameters of the gray soliton
    b1 = 1.5
    b2 = 1.0
    c = 2.0
    xmin, xmax = domain(gray_soliton)

    # To remove the factor 1/2 of Dhaouadie, Favrie, Gavrilyuk (2018)
    x /= sqrt(2)
    xmin /= sqrt(2)
    xmax /= sqrt(2)

    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
    rho = b1 - (b1 - b2) / cosh(sqrt(b1 - b2) * x_t)^2
    # Dhaouadie, Favrie, Gavrilyuk (2018) give the velocity v = ∇ θ.
    # However, we need the angle θ. Thus, we have to integrate the
    # velocity.
    # v = c - b1 * sqrt(b2) / rho
    theta, _ = quadgk(xmin, x_t; atol = 1.0e-14, rtol = 1.0e-14) do xi
        rho_xi = b1 - (b1 - b2) / cosh(sqrt(b1 - b2) * xi)^2
        return c - b1 * sqrt(b2) / rho_xi
    end

    return sqrt(rho) * cis(theta)
end
get_β(::typeof(gray_soliton)) = -1
function domain(::typeof(gray_soliton))
    xmin = -30.0
    # The following solution has been obtained by setting this value
    # temporarily to 90.0 and executing
    # julia> prob = NonlinearProblem(33.0) do x, _
    #            abs(gray_soliton(0.0, x, CubicNLS(-1)) - sqrt(1.5))
    #        end; sol = solve(prob, SimpleKlement())
    # retcode: Success
    # u: 33.94120063695117
    xmax = 33.94120063695117
    return xmin, xmax
end

function error_growth_gray_soliton(; alg = KenCarpARK548(),
                                     tspan = (0.0, 500.0),
                                     dt = 0.05,
                                     N = 2^8,
                                     kwargs...)
    fig = Figure(size = (1200, 400)) # default size is (600, 450)

    @info "Gray soliton"
    initial_condition = gray_soliton
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    xmin, xmax = domain(initial_condition)
    ax = Axis(fig[1, 1];
              xlabel = L"Time $t$",
              ylabel = L"Error at time $t$",
              title = "Total error",
              xscale = log10, yscale = log10)
    ax_mass = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   title = "Mass density error",
                   xscale = log10, yscale = log10)
    D2 = fourier_derivative_operator(xmin, xmax, N)^2
    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        series_error_mass = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_tmp1 = parameters.tmp1
                p_D2 = parameters.D2

                push!(series_t, t)

                v = real(q, parameters.equation)
                w = imag(q, parameters.equation)

                x = grid(p_D2)
                for i in eachindex(x, p_tmp1)
                    ic = initial_condition(t, x[i], parameters.equation)
                    p_tmp1[i] = (v[i] - real(ic))^2 + (w[i] - imag(ic))^2
                end
                push!(series_error, sqrt(integrate(p_tmp1, p_D2)))

                for i in eachindex(x, p_tmp1)
                    ic = initial_condition(t, x[i], parameters.equation)
                    p_tmp1[i] = (v[i]^2 + w[i]^2 - abs2(ic))^2
                end
                push!(series_error_mass, sqrt(integrate(p_tmp1, p_D2)))

                return nothing
            end
        end
        (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, relaxation, callback, kwargs...)
        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax, series_t, series_error; label)
        lines!(ax_mass, series_t, series_error_mass; label)
    end
    t = [1.0, tspan[end]]
    lines!(ax, t, t.^2 .* 5.0e-7; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
    lines!(ax, t, t .* 2.0e-7; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
    axislegend(ax; position = :lt, framevisible = false, nbanks = 2, colgap = 5)

    axislegend(ax_mass; position = :lt, framevisible = false)


    linkyaxes!(ax, ax_mass)
    hideydecorations!(ax_mass; grid = false)


    filename = joinpath(FIGDIR, "error_growth_gray_soliton.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

# Helper function for the performance comparison
function solve_relaxation_crank_nicolson(; initial_condition = two_solitons,
                                           tspan = (0.0, 2.0),
                                           dt = 5.0e-3,
                                           D2 = periodic_derivative_operator(derivative_order = 2; accuracy_order = 6, xmin = -35.0, xmax = 35.0, N = 2^10))
    # Implementation of the method described by Besse et al. (2021) and Besse (2004)
    N = size(D2, 2)
    minus_half_dt_D2 = (-0.5 * dt) * sparse(D2)

    β = get_β(initial_condition)
    equation = CubicNLS(β)

    # Initial condition
    u0 = initial_condition.(tspan[1], grid(D2), equation)
    v0 = real.(u0)
    w0 = imag.(u0)
    q0 = vcat(v0, w0)
    um12 = initial_condition.(tspan[1] - dt / 2, grid(D2), equation)
    y0 = abs2.(um12)

    # Prepare time-stepping loop
    t = tspan[begin]
    q = copy(q0)
    y = copy(y0)
    qnew = similar(q)
    rhs = similar(q)

    # Initialize factorization
    A = minus_half_dt_D2 - (0.5 * dt * β) * Diagonal(y)
    mat = [I -A; A I]
    factorization = lu(mat)

    # Time-stepping loop
    runtime = @elapsed while t < tspan[end]
        v = view(q, 1:N)
        w = view(q, N .+ (1:N))

        # Step 1: compute y^{n+1/2}
        @. y = 2 * (v^2 + w^2) - y

        # Step 2: compute v^{n+1}, w^{n+1}
        A = minus_half_dt_D2 - (0.5 * dt * β) * Diagonal(y)
        mat = [I -A;
               A I]
        mul!(view(rhs, 1:N), A, w)
        view(rhs, 1:N) .= v .+ view(rhs, 1:N)
        mul!(view(rhs, N .+ (1:N)), A, v)
        view(rhs, N .+ (1:N)) .= w .- view(rhs, N .+ (1:N))
        lu!(factorization, mat)
        ldiv!(view(qnew, 1:2N), factorization, rhs) # update vnew, wnew

        # Step 3: update time and solution
        t += dt
        @. q = qnew
    end

    # Compute error
    x = grid(D2)
    v = view(q, 1:N)
    w = view(q, N .+ (1:N))
    tmp1 = (v .- real.(initial_condition.(t, x, equation))).^2
    tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(t, x, equation))).^2
    final_error = sqrt(integrate(tmp1, D2))

    return (; runtime, final_error)
end

# Helper function for the performance comparison
function solve_geodesic_relaxation(; initial_condition = two_solitons,
                                     tspan = (0.0, 2.0),
                                     dt = 5.0e-3,
                                     alg = KenCarpARK548(),
                                     D2 = periodic_derivative_operator(derivative_order = 2; accuracy_order = 6, xmin = -35.0, xmax = 35.0, N = 2^10))
    β = get_β(initial_condition)
    equation = CubicNLS(β)
    (; q0, parameters) = setup(initial_condition, equation, tspan, D2)
    op = operator(rhs_stiff!, parameters)
    runtime = @elapsed begin
        sol = solve_imex(rhs_stiff!, op, rhs_nonstiff!,
                         q0, tspan, parameters, alg;
                         dt, relaxation = FullRelaxation())
    end

    # Compute error
    x = grid(D2)
    v = real(sol.u[end], equation)
    w = imag(sol.u[end], equation)
    tmp1 = (v .- real.(initial_condition.(sol.t[end], x, equation))).^2
    tmp1 .= tmp1 .+ (w .- imag.(initial_condition.(sol.t[end], x, equation))).^2
    final_error = sqrt(integrate(tmp1, D2))

    return (; runtime, final_error)
end

function performance_comparison(; num_trials = 3, tspan = (0.0, 2.0))
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    @info "Two solitons"
    initial_condition = two_solitons
    ax2 = Axis(fig[1, 1];
               xlabel = "Runtime in seconds",
               ylabel = L"Error at $t = %$(tspan[end])$",
               title = "Two solitons",
               xscale = log10, yscale = log10)
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^10)
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.05), log10(0.0001), length = 6)
        @info "Besse et al., N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_relaxation_crank_nicolson(; initial_condition,
                                                           tspan, dt, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"Besse et al., FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^11)
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.05), log10(0.00005), length = 6)
        @info "Besse et al., N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_relaxation_crank_nicolson(; initial_condition,
                                                           tspan, dt, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"Besse et al., FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^10)
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.04), log10(0.005), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"New, FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^11)
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.04), log10(0.003), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"New, FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = fourier_derivative_operator(xmin, xmax, 2^10)^2
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.04), log10(0.001), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"New, Fourier, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = fourier_derivative_operator(xmin, xmax, 2^11)^2
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.04), log10(0.0002), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax2, runtimes, errors; label = L"New, Fourier, $N = %$(size(D2, 2))$")
    end

    fig[1, 3] = Legend(fig, ax2, framevisible = false)

    @info "Three solitons"
    initial_condition = three_solitons
    ax3 = Axis(fig[1, 2];
               xlabel = "Runtime in seconds",
               ylabel = L"Error at $t = %$(tspan[end])$",
               title = "Three solitons",
               xscale = log10, yscale = log10)
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^10)
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.05), log10(0.0001), length = 6)
        @info "Besse et al., N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_relaxation_crank_nicolson(; initial_condition,
                                                           tspan, dt, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"Besse et al., FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^11)
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.05), log10(0.00005), length = 6)
        @info "Besse et al., N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_relaxation_crank_nicolson(; initial_condition,
                                                           tspan, dt, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"Besse et al., FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^10)
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.01), log10(0.005), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"New, FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = periodic_derivative_operator(derivative_order = 2;
                                          accuracy_order = 8,
                                          xmin, xmax, N = 2^11)
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.04), log10(0.001), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"New, FD, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = fourier_derivative_operator(xmin, xmax, 2^10)^2
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.01), log10(0.001), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"New, Fourier, $N = %$(size(D2, 2))$")
    end
    let
        sleep(0.1)
        D2 = fourier_derivative_operator(xmin, xmax, 2^11)^2
        alg = KenCarpARK548()
        errors = Float64[]
        runtimes = Float64[]
        dts = 10.0 .^ range(log10(0.01), log10(0.0001), length = 6)
        @info "New method, N = $(size(D2, 2)), $(length(dts) * num_trials) runs"
        for dt in dts
            t = Inf
            for trial in 1:num_trials
                result = solve_geodesic_relaxation(; initial_condition,
                                                     tspan, dt, alg, D2)
                isone(trial) && push!(errors, result.final_error)
                t = min(t, result.runtime)
                print(".")
            end
            push!(runtimes, t)
        end
        println()
        scatterlines!(ax3, runtimes, errors; label = L"New, Fourier, $N = %$(size(D2, 2))$")
    end


    linkyaxes!(ax2, ax3)
    hideydecorations!(ax3; grid = false)


    filename = joinpath(FIGDIR, "performance_comparison.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function fully_discrete_conservation_hyperbolization(; tspan = (0.0, 4.3),
                                                       alg = ARS443(),
                                                       dt2 = 1.0e-2,
                                                       dt3 = 1.0e-3,
                                                       N = 2^10,
                                                       accuracy_order = 6,
                                                       τ = 1.0e-4,
                                                       kwargs...)
    # Initialization of physical parameters
    xmin = -35.0
    xmax = +35.0

    fig = Figure(size = (1200, 750)) # default size is (600, 450)

    initial_condition = two_solitons
    β = get_β(initial_condition) # 2 n^2 for n solitons
    equation = HyperbolizedCubicNLS(β, τ)

    D1 = upwind_operators(periodic_derivative_operator;
                          accuracy_order, xmin, xmax, N)

    ax2_sol = Axis(fig[1, 1];
                   xlabel = L"Space $x$",
                   ylabel = L"Mass density $|u|^2$",
                   title = "Two solitons at time $(tspan[end])")
    lines!(ax2_sol, grid(D1), abs2.(initial_condition.(tspan[end], grid(D1), CubicNLS(equation.β))); label = "exact", linestyle = :dot, color = :gray)

    ax2_invariants = Axis(fig[1, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Two solitons")
    ax2_invariants_inset = Axis(fig[1, 2],
                                width = Relative(0.4),
                                height = Relative(0.3),
                                halign = 0.99,
                                valign = 0.23,
                                title = "Zoom")

    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt2, relaxation, callback, kwargs...)

        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax2_sol, grid(D1), density(sol.u[end], equation); label)

        lines!(ax2_invariants, series_t, change(series_energy); label = label * ": energy")
        lines!(ax2_invariants, series_t, change(series_mass); label = label * ": mass")

        lines!(ax2_invariants_inset, series_t, change(series_energy))
        lines!(ax2_invariants_inset, series_t, change(series_mass))
    end

    axislegend(ax2_sol; position = :lt, framevisible = false)
    xlims!(ax2_sol, -8, 8)

    axislegend(ax2_invariants; position = :lt, framevisible = false)

    ylims!(ax2_invariants_inset, -4.0e-13, 4.0e-13)
    hidexdecorations!(ax2_invariants_inset; grid = false)


    initial_condition = three_solitons
    β = get_β(initial_condition) # 2 n^2 for n solitons
    equation = HyperbolizedCubicNLS(β, τ)

    D1 = upwind_operators(periodic_derivative_operator;
                          accuracy_order, xmin, xmax, N)

    ax3_sol = Axis(fig[2, 1];
                   xlabel = L"Space $x$",
                   ylabel = L"Mass density $|u|^2$",
                   title = "Three solitons at time $(tspan[end])")
    lines!(ax3_sol, grid(D1), abs2.(initial_condition.(tspan[end], grid(D1), CubicNLS(equation.β))); label = "exact", linestyle = :dot, color = :gray)

    ax3_invariants = Axis(fig[2, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Three solitons")
    ax3_invariants_inset = Axis(fig[2, 2],
                                width = Relative(0.4),
                                height = Relative(0.3),
                                halign = 0.99,
                                valign = 0.15,
                                title = "Zoom")

    for relaxation in (NoProjection(), FullRelaxation())
        @show relaxation

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_energy = series_energy
            function (q, parameters, t)
                (; equation) = parameters

                push!(series_t, t)
                push!(series_mass, mass(q, equation, parameters))
                push!(series_energy, energy(q, equation, parameters))

                return nothing
            end
        end

        (; q0, parameters) = setup(initial_condition, equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt3, relaxation, callback, kwargs...)

        label = relaxation isa NoProjection ? "baseline" : "relaxation"
        lines!(ax3_sol, grid(D1), density(sol.u[end], equation); label)

        lines!(ax3_invariants, series_t, change(series_energy); label = label * ": energy")
        lines!(ax3_invariants, series_t, change(series_mass); label = label * ": mass")

        lines!(ax3_invariants_inset, series_t, change(series_energy))
        lines!(ax3_invariants_inset, series_t, change(series_mass))
    end

    axislegend(ax3_sol; position = :lt, framevisible = false)
    xlims!(ax3_sol, -8, 8)

    axislegend(ax3_invariants; position = :lt, framevisible = false)

    ylims!(ax3_invariants_inset, -4.0e-13, 4.0e-13)
    hidexdecorations!(ax3_invariants_inset; grid = false)


    linkxaxes!(ax2_sol, ax3_sol)
    hidexdecorations!(ax2_sol; grid = false)
    linkxaxes!(ax2_invariants, ax3_invariants)
    hidexdecorations!(ax2_invariants; grid = false)


    colgap!(fig.layout, 0)


    filename = joinpath(FIGDIR, "fully_discrete_conservation_hyperbolization.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end
