import torch


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Drivable Surface",
    1: "Static Obstacle",
    2: "Dynamic Object"
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    # Load model weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # Move model to device
    model = model.to(device)


    # Evaluation mode
    model.eval()


    print(
        "Trained model loaded successfully! ✅"
    )


    return model


# ============================================================
# RUN PREDICTION
# ============================================================

def predict(
    model,
    points,
    device
):

    # Convert NumPy array to PyTorch tensor
    points_tensor = torch.from_numpy(
        points
    ).float()


    # Add batch dimension
    #
    # Before:
    # [8192, 5]
    #
    # After:
    # [1, 8192, 5]

    points_tensor = points_tensor.unsqueeze(
        0
    )


    # Move to device
    points_tensor = points_tensor.to(
        device
    )


    # Evaluation mode
    model.eval()


    # Disable gradients
    with torch.no_grad():

        outputs = model(
            points_tensor
        )


        # Get predicted class
        predictions = torch.argmax(
            outputs,
            dim=1
        )


    # Remove batch dimension
    predictions = predictions.squeeze(
        0
    )


    # Move to CPU
    predictions = predictions.cpu().numpy()


    return predictions


# ============================================================
# DISPLAY PREDICTION SUMMARY
# ============================================================

def print_prediction_summary(
    predictions
):

    print("\n" + "=" * 60)

    print("PREDICTION RESULTS")

    print("=" * 60)


    total_points = len(
        predictions
    )


    for class_id, class_name in CLASS_NAMES.items():

        count = (
            predictions == class_id
        ).sum()
        

        percentage = (
            count /
            total_points
        ) * 100


        print(
            f"\nClass {class_id}: "
            f"{class_name}"
        )

        print(
            f"Predicted Points: "
            f"{count}"
        )

        print(
            f"Percentage: "
            f"{percentage:.2f}%"
        )


    print("\n" + "=" * 60)

# ============================================================
# RUN BATCH PREDICTION
# ============================================================

def predict_batch(
    model,
    points_tensor,
    device
):

    # Move to device
    points_tensor = points_tensor.float().to(device)

    # Evaluation mode
    model.eval()

    # Disable gradients and enable mixed precision for speed
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):

        outputs = model(points_tensor)

        # Get predicted class
        predictions = torch.argmax(
            outputs,
            dim=1
        )

    # Move to CPU
    predictions = predictions.cpu().numpy()

    return predictions
