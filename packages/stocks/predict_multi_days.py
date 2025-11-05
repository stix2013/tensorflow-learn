import numpy as np

def predict_multi_day(model, scaler, last_actual_window, num_future_days, time_step):
    """
    Predicts prices for a specified number of future days recursively.
    
    Args:
        model: The loaded Keras LSTM model.
        scaler: The loaded fitted MinMaxScaler.
        last_actual_window (np.array): The last TIME_STEP days of scaled data (shape: (TIME_STEP, 1)).
        num_future_days (int): How many days into the future to predict.
        time_step (int): The sample data len
    Returns:
        np.array: Unscaled price predictions for the future days.
    """
    
    # 1. Start the list with the last actual window (scaled)
    temp_input = list(last_actual_window.flatten())
    
    # 2. List to store future predictions (scaled)
    lst_output = []
    i = 0
    
    while i < num_future_days:
        if len(temp_input) > time_step:
            # Drop the oldest data point and take the latest TIME_STEP elements
            x_input = np.array(temp_input[1:]) 
        else:
            # Use the initial last_actual_window (already set)
            x_input = np.array(temp_input)
            
        # Reshape for model: (1, TIME_STEP, 1)
        x_input = x_input.reshape(1, time_step, 1)
        
        # Predict the next step
        yhat = model.predict(x_input, verbose=0) 
        
        # Append the prediction (scaled) to the output list
        lst_output.append(yhat[0, 0])
        
        # Update the input window by dropping the oldest value and adding the new prediction
        temp_input = temp_input[1:] + [yhat[0, 0]]
        
        i += 1
    
    # 3. Inverse transform the scaled predictions
    final_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
    
    return final_predictions