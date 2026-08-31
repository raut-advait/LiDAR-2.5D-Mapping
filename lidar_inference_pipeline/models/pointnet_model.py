import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_layers import PointNetSetAbstraction
from .pointnet2_utils import square_distance, index_points


# ============================================================
# FEATURE PROPAGATION LAYER
# ============================================================

class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()

        layers = []
        last_channel = in_channel

        for out_channel in mlp:

            layers.append(
                nn.Conv1d(
                    last_channel,
                    out_channel,
                    kernel_size=1
                )
            )

            layers.append(
                nn.BatchNorm1d(out_channel)
            )

            layers.append(
                nn.ReLU()
            )

            last_channel = out_channel

        self.mlp = nn.Sequential(*layers)


    def forward(
        self,
        xyz1,
        xyz2,
        points1,
        points2
    ):

        """
        xyz1:
            Target coordinates
            Shape: (B, N, 3)

        xyz2:
            Source coordinates
            Shape: (B, S, 3)

        points1:
            Skip connection features
            Shape: (B, N, D1)

        points2:
            Features to interpolate
            Shape: (B, S, D2)
        """

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape


        # ----------------------------------------------------
        # SPECIAL CASE:
        # ONLY ONE SOURCE POINT
        # ----------------------------------------------------

        if S == 1:

            interpolated_points = points2.repeat(
                1,
                N,
                1
            )


        else:

            # ------------------------------------------------
            # CALCULATE DISTANCE
            # ------------------------------------------------

            dists = square_distance(
                xyz1,
                xyz2
            )


            # ------------------------------------------------
            # FIND 3 NEAREST POINTS
            # ------------------------------------------------

            dists, idx = torch.topk(
                dists,
                k=3,
                dim=-1,
                largest=False,
                sorted=False
            )


            # ------------------------------------------------
            # INVERSE DISTANCE WEIGHTS
            # ------------------------------------------------

            dist_recip = 1.0 / (
                dists + 1e-8
            )


            norm = torch.sum(
                dist_recip,
                dim=2,
                keepdim=True
            )


            weight = dist_recip / norm


            # ------------------------------------------------
            # INTERPOLATE FEATURES
            # ------------------------------------------------

            grouped_points = index_points(
                points2,
                idx
            )


            interpolated_points = torch.sum(
                grouped_points
                * weight.unsqueeze(-1),
                dim=2
            )


        # ----------------------------------------------------
        # SKIP CONNECTION
        # ----------------------------------------------------

        if points1 is not None:

            new_points = torch.cat(
                [
                    points1,
                    interpolated_points
                ],
                dim=-1
            )

        else:

            new_points = interpolated_points


        # ----------------------------------------------------
        # APPLY MLP
        # ----------------------------------------------------

        new_points = new_points.permute(
            0,
            2,
            1
        )


        new_points = self.mlp(
            new_points
        )


        new_points = new_points.permute(
            0,
            2,
            1
        )


        return new_points


# ============================================================
# MAIN POINTNET++ SEGMENTATION MODEL
# ============================================================

class PointNetPlusPlusSegmentation(nn.Module):

    def __init__(
        self,
        num_classes=3,
        input_features=2
    ):

        super(
            PointNetPlusPlusSegmentation,
            self
        ).__init__()


        # ====================================================
        # ENCODER
        # ====================================================


        # ----------------------------------------------------
        # SET ABSTRACTION LAYER 1
        # ----------------------------------------------------

        self.sa1 = PointNetSetAbstraction(

            npoint=1024,

            radius=0.2,

            nsample=32,

            in_channel=input_features,

            mlp=[
                64,
                64,
                128
            ]

        )


        # ----------------------------------------------------
        # SET ABSTRACTION LAYER 2
        # ----------------------------------------------------

        self.sa2 = PointNetSetAbstraction(

            npoint=256,

            radius=0.4,

            nsample=32,

            in_channel=128,

            mlp=[
                128,
                128,
                256
            ]

        )


        # ----------------------------------------------------
        # SET ABSTRACTION LAYER 3
        # ----------------------------------------------------

        self.sa3 = PointNetSetAbstraction(

            npoint=64,

            radius=0.8,

            nsample=32,

            in_channel=256,

            mlp=[
                256,
                256,
                512
            ]

        )


        # ====================================================
        # DECODER / FEATURE PROPAGATION
        # ====================================================


        # ----------------------------------------------------
        # 64 → 256
        # ----------------------------------------------------

        self.fp3 = PointNetFeaturePropagation(

            in_channel=512 + 256,

            mlp=[
                256,
                256
            ]

        )


        # ----------------------------------------------------
        # 256 → 1024
        # ----------------------------------------------------

        self.fp2 = PointNetFeaturePropagation(

            in_channel=256 + 128,

            mlp=[
                256,
                128
            ]

        )


        # ----------------------------------------------------
        # 1024 → 8192
        # ----------------------------------------------------

        self.fp1 = PointNetFeaturePropagation(

            in_channel=128 + input_features,

            mlp=[
                128,
                128,
                128
            ]

        )


        # ====================================================
        # FINAL SEGMENTATION HEAD
        # ====================================================

        self.conv1 = nn.Conv1d(
            128,
            128,
            kernel_size=1
        )


        self.bn1 = nn.BatchNorm1d(
            128
        )


        self.dropout = nn.Dropout(
            0.5
        )


        self.conv2 = nn.Conv1d(
            128,
            num_classes,
            kernel_size=1
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, points):

        """
        Input:
            points
            Shape: (B, N, 5)

        Output:
            logits
            Shape: (B, num_classes, N)
        """


        # ====================================================
        # SPLIT XYZ AND FEATURES
        # ====================================================

        xyz = points[:, :, :3]

        features = points[:, :, 3:]


        # ====================================================
        # ENCODER
        # ====================================================

        l1_xyz, l1_points = self.sa1(
            xyz,
            features
        )


        l2_xyz, l2_points = self.sa2(
            l1_xyz,
            l1_points
        )


        l3_xyz, l3_points = self.sa3(
            l2_xyz,
            l2_points
        )


        # ====================================================
        # DECODER
        # ====================================================


        # 64 → 256

        l2_points = self.fp3(

            l2_xyz,

            l3_xyz,

            l2_points,

            l3_points

        )


        # 256 → 1024

        l1_points = self.fp2(

            l1_xyz,

            l2_xyz,

            l1_points,

            l2_points

        )


        # 1024 → ORIGINAL N

        l0_points = self.fp1(

            xyz,

            l1_xyz,

            features,

            l1_points

        )


        # ====================================================
        # FINAL CLASSIFICATION
        # ====================================================

        x = l0_points.permute(
            0,
            2,
            1
        )


        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )


        x = self.dropout(
            x
        )


        logits = self.conv2(
            x
        )


        return logits