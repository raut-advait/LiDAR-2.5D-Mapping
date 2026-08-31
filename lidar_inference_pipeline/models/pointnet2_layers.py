import torch
import torch.nn as nn

from .pointnet2_utils import (
    index_points,
    farthest_point_sample,
    query_ball_point
)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer.

    Steps:
    1. Sample important points using Farthest Point Sampling.
    2. Group nearby points using Ball Query.
    3. Learn local geometric features using MLP layers.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []

        # +3 because XYZ coordinates are also used
        last_channel = in_channel + 3

        for out_channel in mlp:

            layers.append(
                nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=1
                )
            )

            layers.append(
                nn.BatchNorm2d(out_channel)
            )

            layers.append(
                nn.ReLU()
            )

            last_channel = out_channel

        self.mlp = nn.Sequential(*layers)


    def forward(self, xyz, points=None):

        """
        xyz:
            Shape: (B, N, 3)

        points:
            Optional point features
            Shape: (B, N, D)
        """

        # ----------------------------------
        # STEP 1: GET INPUT DIMENSIONS
        # ----------------------------------

        B, N, C = xyz.shape


        # ----------------------------------
        # STEP 2: FARTHEST POINT SAMPLING
        # ----------------------------------

        fps_idx = farthest_point_sample(
            xyz,
            self.npoint
        )

        new_xyz = index_points(
            xyz,
            fps_idx
        )


        # ----------------------------------
        # STEP 3: GROUP NEARBY POINTS
        # ----------------------------------

        group_idx = query_ball_point(
            self.radius,
            self.nsample,
            xyz,
            new_xyz
        )

        grouped_xyz = index_points(
            xyz,
            group_idx
        )


        # ----------------------------------
        # STEP 4: NORMALIZE POINTS
        # ----------------------------------

        grouped_xyz_norm = (
            grouped_xyz
            - new_xyz.unsqueeze(2)
        )


        # ----------------------------------
        # STEP 5: ADD POINT FEATURES
        # ----------------------------------

        if points is not None:

            grouped_points = index_points(
                points,
                group_idx
            )

            new_points = torch.cat(
                [
                    grouped_xyz_norm,
                    grouped_points
                ],
                dim=-1
            )

        else:

            new_points = grouped_xyz_norm


        # ----------------------------------
        # STEP 6: PREPARE FOR CNN
        # ----------------------------------

        # Before:
        # (B, npoint, nsample, channels)

        # After:
        # (B, channels, nsample, npoint)

        new_points = new_points.permute(
            0,
            3,
            2,
            1
        )


        # ----------------------------------
        # STEP 7: LOCAL FEATURE LEARNING
        # ----------------------------------

        new_points = self.mlp(
            new_points
        )


        # ----------------------------------
        # STEP 8: MAX POOLING
        # ----------------------------------

        new_points = torch.max(
            new_points,
            dim=2
        )[0]


        # ----------------------------------
        # STEP 9: FINAL SHAPE
        # ----------------------------------

        # Before:
        # (B, Features, npoint)

        # After:
        # (B, npoint, Features)

        new_points = new_points.permute(
            0,
            2,
            1
        )


        return new_xyz, new_points