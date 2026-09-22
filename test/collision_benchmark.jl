# CollisionDetection.jl vs H2Trees collision-tree benchmark.
# Measures construction and repeated broad-phase queries without materializing query results.
#
# IMPORTANT: this script intentionally does not import/use Pkg.
# Run it with the already configured test environment, e.g.:
#   julia -O1 --startup-file=no --project=test test\collision_benchmark.jl

println("Active project: ", Base.active_project())
println("Julia version:   ", VERSION)

using CompScienceMeshes
import CollisionDetection
import H2Trees
isdefined(H2Trees, :CollisionTree) || error(
    "The loaded H2Trees does not contain the CollisionTree implementation. " *
    "Set H2TREES_PATH to the modified local H2Trees directory."
)
using Printf
using Random
using StaticArrays
using Test

println("CompScienceMeshes loaded from: ", pathof(CompScienceMeshes))
println("H2Trees loaded from:          ", pathof(H2Trees))
println("CollisionDetection loaded from: ", pathof(CollisionDetection))

const NOBJECTS = 20_000
const NQUERIES = 2_000
const REPEATS = 5
const CORRECTNESS_QUERIES = 100

function make_objects(::Val{N}, n, seed; finite_radius=true) where {N}
    rng = MersenneTwister(seed)
    points = Vector{SVector{N,Float64}}(undef, n)
    radii = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        points[i] = SVector{N,Float64}(ntuple(_ -> 2rand(rng) - 1, N))
        radii[i] = finite_radius ? 0.005 + 0.025rand(rng) : 0.0
    end
    return points, radii
end

function make_queries(::Val{N}, n, seed) where {N}
    rng = MersenneTwister(seed)
    queries = Vector{Tuple{SVector{N,Float64},Float64}}(undef, n)
    @inbounds for i in 1:n
        center = SVector{N,Float64}(ntuple(_ -> 2rand(rng) - 1, N))
        halfsize = 0.02 + 0.08rand(rng)
        queries[i] = (center, halfsize)
    end
    return queries
end

build_old(points, radii) = CollisionDetection.Octree(points, radii)
build_new(points, radii) = H2Trees.Octree(points, radii)

function candidates_old(tree, query)
    qc, qh = query
    ids = Int[]
    pred = (c, s) -> CollisionDetection.boxesoverlap(c, s, qc, qh)
    for box in CollisionDetection.boxes(tree, pred)
        append!(ids, box)
    end
    sort!(ids)
    return ids
end

function candidates_new(tree, query)
    qc, qh = query
    ids = Int[]
    pred = (c, s) -> H2Trees.boxesoverlap(c, s, qc, qh)
    for box in H2Trees.boxes(tree, pred)
        append!(ids, box)
    end
    sort!(ids)
    return ids
end

# Allocation-sensitive workload: count candidate ids instead of collecting them.
function scan_old(tree, queries)
    count = 0
    checksum = 0
    @inbounds for (qc, qh) in queries
        pred = (c, s) -> CollisionDetection.boxesoverlap(c, s, qc, qh)
        for box in CollisionDetection.boxes(tree, pred)
            count += length(box)
            for id in box
                checksum += id
            end
        end
    end
    return count, checksum
end

function scan_new(tree, queries)
    count = 0
    checksum = 0
    @inbounds for (qc, qh) in queries
        pred = (c, s) -> H2Trees.boxesoverlap(c, s, qc, qh)
        for box in H2Trees.boxes(tree, pred)
            count += length(box)
            for id in box
                checksum += id
            end
        end
    end
    return count, checksum
end

function measure(f)
    f() # compile/warm up
    times = Float64[]
    bytes = Int[]
    for _ in 1:REPEATS
        GC.gc()
        m = @timed f()
        push!(times, m.time)
        push!(bytes, m.bytes)
    end
    sort!(times)
    sort!(bytes)
    i = cld(REPEATS, 2)
    return times[i], bytes[i]
end

const CASES = (
    ("2D points", Val(2), false, 101),
    ("2D finite-radius", Val(2), true, 102),
    ("3D points", Val(3), false, 201),
    ("3D finite-radius", Val(3), true, 202),
)

@testset "H2Trees CollisionDetection compatibility" begin
    for (name, dim, finite_radius, seed) in CASES
        points, radii = make_objects(dim, 2_000, seed; finite_radius=finite_radius)
        queries = make_queries(dim, CORRECTNESS_QUERIES, seed + 10_000)
        oldtree = build_old(points, radii)
        newtree = build_new(points, radii)

        @test oldtree.center ≈ newtree.center
        @test oldtree.halfsize ≈ newtree.halfsize

        for query in queries
            @test candidates_old(oldtree, query) == candidates_new(newtree, query)
        end

        # Also cover the containment predicate used by weld/embedding paths.
        for (qc, qh) in queries[1:min(20, length(queries))]
            @test CollisionDetection.fitsinbox(qc, qh / 4, oldtree.center, oldtree.halfsize) ==
                  H2Trees.fitsinbox(qc, qh / 4, newtree.center, newtree.halfsize)
        end
        @printf("%-18s compatibility: OK\n", name)
    end
end

println("\nCollision-tree benchmark: $NOBJECTS objects, $NQUERIES queries, median of $REPEATS runs")
@printf("%-18s %-20s %11s %12s %11s %12s\n",
    "case", "backend", "build ms", "build MiB", "query ms", "query MiB")
println("-"^92)

for (name, dim, finite_radius, seed) in CASES
    points, radii = make_objects(dim, NOBJECTS, seed; finite_radius=finite_radius)
    queries = make_queries(dim, NQUERIES, seed + 10_000)

    old_build_t, old_build_b = measure(() -> build_old(points, radii))
    new_build_t, new_build_b = measure(() -> build_new(points, radii))

    oldtree = build_old(points, radii)
    newtree = build_new(points, radii)
    @test scan_old(oldtree, queries) == scan_new(newtree, queries)

    old_query_t, old_query_b = measure(() -> scan_old(oldtree, queries))
    new_query_t, new_query_b = measure(() -> scan_new(newtree, queries))

    @printf("%-18s %-20s %11.3f %12.3f %11.3f %12.3f\n",
        name, "CollisionDetection",
        old_build_t * 1000, old_build_b / 2^20,
        old_query_t * 1000, old_query_b / 2^20)
    @printf("%-18s %-20s %11.3f %12.3f %11.3f %12.3f\n",
        "", "H2Trees",
        new_build_t * 1000, new_build_b / 2^20,
        new_query_t * 1000, new_query_b / 2^20)
    @printf("%-18s %-20s %10.2fx %11.2fx %10.2fx %11.2fx\n\n",
        "", "old/new ratio",
        old_build_t / new_build_t,
        old_build_b / max(new_build_b, 1),
        old_query_t / new_query_t,
        old_query_b / max(new_query_b, 1))
end
