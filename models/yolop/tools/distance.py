# tools/distance.py

def estimate_distance(box, frame_height):
    """
    Calculates distance using simple perspective projection.
    
    Args:
        box (list): [x1, y1, x2, y2] coordinates of the object.
        frame_height (int): Height of the video frame (e.g., 720).
        
    Returns:
        float: Estimated distance in meters.
    """
    
    # 1. Unpack the coordinates
    # We only care about y2 (the bottom of the box/tires)
    x1, y1, x2, y2 = box
    
    # 2. Define the Horizon Line
    # In most dashcams, the horizon is slightly above the middle.
    # 0.45 means "45% down from the top of the screen".
    horizon_line = frame_height * 0.55
    
    # 3. Safety Check (Divide by Zero)
    # If the object is ABOVE the horizon (flying?), it's infinite distance.
    if y2 <= horizon_line:
        return 999.0

    # 4. The Magic Formula
    # Distance = K / (Tire_Position - Horizon)
    # '1000' is our Calibration Constant (K).
    # If distance looks wrong, we change THIS number.
    calibration_constant = 1000
    distance_in_meters = calibration_constant / (y2 - horizon_line)
    
    # 5. Return the result rounded to 1 decimal place (e.g., 12.5)
    # .item() converts the 1-element tensor to a float
    return round(float(distance_in_meters.item()), 1)