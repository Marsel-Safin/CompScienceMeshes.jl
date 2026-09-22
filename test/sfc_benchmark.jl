# ClusterTrees vs H2Trees: tests, benchmark and PlotlyJS visualization.

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CompScienceMeshes
using H2Trees
import ClusterTrees
using LinearAlgebra
using PlotlyJS
using Printf
using Random
using StaticArrays
using Test

println("H2Trees loaded from: ", pathof(H2Trees))

const Point2 = SVector{2,Float64}
const Point3 = SVector{3,Float64}
const BENCH_N = 10000
const PLOT_N = BENCH_N
const REPEATS = 5

# Uniform points in the full 2D unit disk (not only on the circumference).
function disk_points(n)
    rng = MersenneTwister(1)
    p = Point2[]
    sizehint!(p, n)
    for _ in 1:n
        # sqrt(rand) is required for uniform density with respect to disk area.
        r = sqrt(rand(rng))
        theta = 2pi * rand(rng)
        push!(p, Point2(r*cos(theta), r*sin(theta)))
    end
    return p
end

# Uniform points in the full square [-1,1] x [-1,1] (not only on its boundary).
function square_area_points(n)
    rng = MersenneTwister(2)
    p = Point2[]
    sizehint!(p, n)
    for _ in 1:n
        x = 2*rand(rng) - 1
        y = 2*rand(rng) - 1
        push!(p, Point2(x, y))
    end
    return p
end

function sphere_points(n)
    p = Point3[]
    golden = pi * (3 - sqrt(5.0))
    for k in 0:n-1
        y = 1 - 2 * (k + 0.5) / n
        r = sqrt(1 - y*y)
        a = k * golden
        push!(p, Point3(r*cos(a), y, r*sin(a)))
    end
    shuffle!(MersenneTwister(3), p)
    return p
end

function cube_points(n)
    rng = MersenneTwister(4)
    p = Point3[]
    for _ in 1:n
        u, v = 2*rand(rng) - 1, 2*rand(rng) - 1
        face = rand(rng, 1:6)
        push!(p,
            face == 1 ? Point3( 1, u, v) :
            face == 2 ? Point3(-1, u, v) :
            face == 3 ? Point3(u,  1, v) :
            face == 4 ? Point3(u, -1, v) :
            face == 5 ? Point3(u, v,  1) : Point3(u, v, -1))
    end
    return p
end

function torus_points(n)
    rng = MersenneTwister(5)
    p = Point3[]
    R, r = 1.0, 0.35
    for _ in 1:n
        u, v = 2pi*rand(rng), 2pi*rand(rng)
        push!(p, Point3((R+r*cos(v))*cos(u), (R+r*cos(v))*sin(u), r*sin(v)))
    end
    return p
end

const CASES = (
    ("2D disk", disk_points),
    ("2D square area", square_area_points),
    ("3D sphere", sphere_points),
    ("3D cube", cube_points),
    ("3D torus", torus_points),
)

# Old CompScienceMeshes sort_sfc, kept only as the benchmark baseline.
function sort_sfc_cluster(points)
    ct, sz = CompScienceMeshes.boundingbox(points)
    tree = ClusterTrees.LevelledTrees.LevelledTree(ct, sz, Int[])
    smb = sz / 2^log(length(points) + 1)

    for (i, pt) in enumerate(points)
        dest = (smallest_box_size=smb, target_point=pt)
        state = ClusterTrees.LevelledTrees.rootstate(tree, dest)
        ClusterTrees.update!(tree, state, i, dest) do tree, node, i
            push!(ClusterTrees.data(tree, node).values, i)
        end
    end
    
    sorted = Int[]
    for node in ClusterTrees.DepthFirstIterator(tree, ClusterTrees.root(tree))
        append!(sorted, ClusterTrees.data(tree, node).values)
    end
    return sorted
end

path_length(points, order) =
    sum(norm(points[order[i+1]] - points[order[i]]) for i in 1:length(order)-1)

# Build exactly the same uniform H2Trees tree used by CompScienceMeshes.sort_sfc.
function build_uniform_sfc_tree(points)
    builder = H2Trees.TwoNTreeBuilder(
        minhalfsize=0.0,
        minvalues=1,
        protrusion=H2Trees.NoProtrusionCheck(),
        uniform=true,
    )
    return H2Trees.buildtree(points; builder=builder)
end

# Verify that every occupied leaf is on one level and that traversing the tree
# produces exactly the same permutation as CompScienceMeshes.sort_sfc.
function check_uniform_leaves(name, points)
    tree = build_uniform_sfc_tree(points)
    leafnodes = H2Trees.leaves(tree)
    leaflevels = H2Trees.level.(Ref(tree), leafnodes)
    uniquelevels = sort!(unique(leaflevels))

    @test !isempty(leafnodes)
    @test H2Trees.checkbalancedtree(tree)
    @test length(uniquelevels) == 1
    @test all(!isempty(H2Trees.values(H2Trees.data(tree, leaf))) for leaf in leafnodes)
    @test all(length(H2Trees.values(H2Trees.data(tree, leaf))) == 1 for leaf in leafnodes)
    @test length(leafnodes) == length(points)

    treeorder = Int[]
    sizehint!(treeorder, length(points))
    H2Trees.appendvalues!(treeorder, tree, H2Trees.root(tree))
    @test sort(treeorder) == collect(eachindex(points))

    leaflevel = only(uniquelevels)
    @printf("%-16s leaf level: %-4d leaves: %-6d uniform: true\n",
        name, leaflevel, length(leafnodes))

    return tree
end

function measure(f, points)
    f(points) # warm-up
    times, bytes = Float64[], Int[]
    for _ in 1:REPEATS
        GC.gc()
        m = @timed f(points)
        push!(times, m.time)
        push!(bytes, m.bytes)
    end
    sort!(times); sort!(bytes)
    i = cld(REPEATS, 2)
    return times[i], bytes[i]
end

@testset "sort_sfc" begin
    for (_, makepoints) in CASES
        points = makepoints(1000)
        order = CompScienceMeshes.sort_sfc(points)
        @test sort(order) == collect(eachindex(points))
        @test path_length(points, order) < path_length(points, eachindex(points))
    end
end

println("\nUniform H2Trees leaf-level check: 1000 points per case")
@testset "uniform H2Trees leaves" begin
    for (name, makepoints) in CASES
        check_uniform_leaves(name, makepoints(1000))
    end
end

println("\nSFC benchmark: $BENCH_N points, median of $REPEATS runs")

@printf("%-16s %-14s %12s %12s %12s %16s\n",
    "case", "backend", "time ms", "alloc MiB", "speedup", "alloc ratio")

println("-"^88)

for (name, makepoints) in CASES
    points = makepoints(BENCH_N)

    old_time, old_bytes = measure(sort_sfc_cluster, points)
    new_time, new_bytes = measure(CompScienceMeshes.sort_sfc, points)

    speedup = old_time / new_time
    alloc_ratio = old_bytes / max(new_bytes, 1)

    @printf("%-16s %-14s %12.3f %12.3f %12s %16s\n",
        name, "ClusterTrees",
        old_time * 1000,
        old_bytes / 2^20,
        "-", "-")

    @printf("%-16s %-14s %12.3f %12.3f %11.2fx %15.2fx\n",
        name, "H2Trees",
        new_time * 1000,
        new_bytes / 2^20,
        speedup,
        alloc_ratio)
end

# Plot style follows H2Trees/docs/plots/hilbert_curve_2d.jl and _3d.jl.
function show_sfc(name, points)
    p = points[CompScienceMeshes.sort_sfc(points)] #trennen
    order = 0:length(p)-1

    fig = if length(first(p)) == 2
        plot(scatter(
                x=getindex.(p, 1), y=getindex.(p, 2), mode="lines+markers",
                line=attr(color="rgb(170,170,170)", width=1.5),
                marker=attr(size=5, color=order, colorscale="Viridis")),
            Layout(title=name, showlegend=false,
                xaxis=attr(visible=false, scaleanchor="y", scaleratio=1),
                yaxis=attr(visible=false)))
    else
        plot(scatter3d(
                x=getindex.(p, 1), y=getindex.(p, 2), z=getindex.(p, 3),
                mode="lines+markers",
                line=attr(color=order, colorscale="Viridis", width=4),
                marker=attr(size=3, color=order, colorscale="Viridis")),
            Layout(title=name, showlegend=false,
                scene=attr(xaxis=attr(visible=false), yaxis=attr(visible=false),
                           zaxis=attr(visible=false), aspectmode="data")))
    end

    mkpath(joinpath(@__DIR__, "output"))
    filename = replace(name, " " => "_") * ".html"
    savefig(fig, joinpath(@__DIR__, "output", filename))
end

show_sfc("2D disk", disk_points(PLOT_N))
show_sfc("2D square area", square_area_points(PLOT_N))
show_sfc("3D sphere", sphere_points(PLOT_N))
show_sfc("3D cube", cube_points(PLOT_N))
show_sfc("3D torus", torus_points(PLOT_N))
