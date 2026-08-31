import torch


# ============================================================
# 1. SQUARE DISTANCE
# ============================================================

def square_distance(src, dst):
    """
    Calculate squared distance between every point in src and dst.

    src shape: [B, N, C]
    dst shape: [B, M, C]

    Returns:
    distance shape: [B, N, M]
    """

    return torch.sum(
        (src.unsqueeze(2) - dst.unsqueeze(1)) ** 2,
        dim=-1
    )


# ============================================================
# 2. INDEX POINTS
# ============================================================

def index_points(points, idx):
    """
    Select points using indices.

    points shape:
        [B, N, C]

    idx shape:
        [B, S]
        OR
        [B, S, K]

    Returns selected points.
    """

    batch_size = points.shape[0]

    batch_indices = torch.arange(
        batch_size,
        device=points.device
    )

    batch_indices = batch_indices.view(
        batch_size,
        *([1] * (idx.dim() - 1))
    )

    batch_indices = batch_indices.expand_as(idx)

    return points[batch_indices, idx]


# ============================================================
# 3. FARTHEST POINT SAMPLING
# ============================================================

def farthest_point_sample(xyz, npoint):
    """
    Select representative points using Farthest Point Sampling.

    xyz shape:
        [B, N, 3]

    npoint:
        Number of points to sample.

    Returns:
        centroids indices
        shape: [B, npoint]
    """

    device = xyz.device

    B, N, _ = xyz.shape

    centroids = torch.zeros(
        B,
        npoint,
        dtype=torch.long,
        device=device
    )

    distance = torch.ones(
        B,
        N,
        device=device
    ) * 1e10

    farthest = torch.randint(
        0,
        N,
        (B,),
        dtype=torch.long,
        device=device
    )

    batch_indices = torch.arange(
        B,
        dtype=torch.long,
        device=device
    )

    for i in range(npoint):

        centroids[:, i] = farthest

        centroid = xyz[
            batch_indices,
            farthest
        ].view(B, 1, 3)

        dist = torch.sum(
            (xyz - centroid) ** 2,
            dim=-1
        )

        mask = dist < distance

        distance[mask] = dist[mask]

        farthest = torch.max(
            distance,
            dim=-1
        )[1]

    return centroids


# ============================================================
# 4. QUERY BALL POINT
# ============================================================

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Find nearby points around each centroid.

    radius:
        Search radius.

    nsample:
        Maximum number of neighbors.

    xyz:
        Original points
        [B, N, 3]

    new_xyz:
        Centroid points
        [B, S, 3]

    Returns:
        Group indices
        [B, S, nsample]
    """

    device = xyz.device

    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(
        new_xyz,
        xyz
    )

    group_idx = torch.arange(
        N,
        device=device
    ).view(1, 1, N).repeat(B, S, 1)

    group_idx[
        sqrdists > radius ** 2
    ] = N

    group_idx = group_idx.sort(
        dim=-1
    )[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(
        B,
        S,
        1
    ).repeat(1, 1, nsample)

    mask = group_idx == N

    group_idx[mask] = group_first[mask]

    return group_idx