using H2Trees

"""
    sort_sfc(points)

Return indices that order `points` along the recursive Hilbert space-filling
curve used by `H2Trees.TwoNTree`.
"""
function sort_sfc(points)
    isempty(points) && return Int[]

    _, sz = CompScienceMeshes.boundingbox(points)
    iszero(sz) && return collect(eachindex(points))

    # Keep approximately the same spatial resolution as the previous
    # ClusterTrees implementation.
    minhalfsize = sz / 2^log(length(points) + 1)

    builder = H2Trees.TwoNTreeBuilder(
        minhalfsize=minhalfsize,
        minvalues=0,
        protrusion=H2Trees.NoProtrusionCheck(),
    )

    tree = H2Trees.buildtree(points; builder=builder)

    sorted = Int[]
    sizehint!(sorted, length(points))
    H2Trees.appendvalues!(sorted, tree, H2Trees.root(tree))

    return sorted
end