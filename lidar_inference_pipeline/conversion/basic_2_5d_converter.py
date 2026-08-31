import numpy as np


class Basic2_5DConverter:

    def __init__(self, resolution=0.5):
        """
        Basic fixed-resolution 3D to 2.5D converter.

        Parameters
        ----------
        resolution : float
            Size of one grid cell in meters.

            Example:
            resolution = 0.5

            Means each grid cell represents:
            0.5m x 0.5m
        """

        self.resolution = resolution


    # ============================================================
    # CONVERT 3D POINT CLOUD TO BASIC 2.5D GRID
    # ============================================================

    def convert(self, points, labels):
        """
        Convert classified 3D LiDAR points into
        a fixed-resolution 2.5D grid.

        Parameters
        ----------
        points : numpy.ndarray

            Shape:
            (N, 5)

            Format:
            [X, Y, Z, Intensity, Ring]


        labels : numpy.ndarray

            Shape:
            (N,)

            Predicted semantic label for every point.


        Returns
        -------
        grid : dict

            Contains:
            - height
            - labels
            - point_count
            - resolution
            - min_x
            - min_y
        """


        # ========================================================
        # INPUT VALIDATION
        # ========================================================

        if len(points) == 0:

            raise ValueError(
                "Point cloud is empty."
            )


        if len(points) != len(labels):

            raise ValueError(
                "Number of points and labels must be equal."
            )


        # ========================================================
        # EXTRACT XYZ COORDINATES
        # ========================================================

        x = points[:, 0]

        y = points[:, 1]

        z = points[:, 2]


        # ========================================================
        # FIND SPATIAL BOUNDARIES
        # ========================================================

        min_x = np.min(x)
        max_x = np.max(x)

        min_y = np.min(y)
        max_y = np.max(y)


        # ========================================================
        # CALCULATE GRID DIMENSIONS
        # ========================================================

        grid_width = int(
            np.ceil(
                (max_x - min_x)
                / self.resolution
            )
        ) + 1


        grid_height = int(
            np.ceil(
                (max_y - min_y)
                / self.resolution
            )
        ) + 1


        # ========================================================
        # CREATE 2.5D GRID ARRAYS
        # ========================================================

        # Maximum Z height for every cell
        height_grid = np.full(
            (
                grid_width,
                grid_height
            ),

            np.nan,

            dtype=np.float32
        )


        # Semantic label for every cell
        label_grid = np.full(
            (
                grid_width,
                grid_height
            ),

            -1,

            dtype=np.int32
        )


        # Number of points inside every cell
        point_count_grid = np.zeros(
            (
                grid_width,
                grid_height
            ),

            dtype=np.int32
        )


        # ========================================================
        # CONVERT XY COORDINATES TO GRID INDICES
        # ========================================================

        grid_x = (
            (x - min_x)
            / self.resolution
        ).astype(np.int32)


        grid_y = (
            (y - min_y)
            / self.resolution
        ).astype(np.int32)


        # ========================================================
        # PROCESS EACH POINT
        # ========================================================

        for i in range(len(points)):

            gx = grid_x[i]

            gy = grid_y[i]


            # ----------------------------------------------------
            # COUNT POINTS IN CELL
            # ----------------------------------------------------

            point_count_grid[
                gx,
                gy
            ] += 1


            # ----------------------------------------------------
            # STORE MAXIMUM HEIGHT
            # ----------------------------------------------------

            current_height = height_grid[
                gx,
                gy
            ]


            if (
                np.isnan(
                    current_height
                )
                or
                z[i] > current_height
            ):

                # Store maximum height
                height_grid[
                    gx,
                    gy
                ] = z[i]


                # Store semantic class corresponding
                # to the highest point
                label_grid[
                    gx,
                    gy
                ] = labels[i]


        # ========================================================
        # CREATE FINAL 2.5D MAP
        # ========================================================

        grid = {

            "height": height_grid,

            "labels": label_grid,

            "point_count": point_count_grid,

            "resolution": np.array(
                self.resolution,
                dtype=np.float32
            ),

            "min_x": np.array(
                min_x,
                dtype=np.float32
            ),

            "max_x": np.array(
                max_x,
                dtype=np.float32
            ),

            "min_y": np.array(
                min_y,
                dtype=np.float32
            ),

            "max_y": np.array(
                max_y,
                dtype=np.float32
            )

        }


        return grid