# TensorFlow Learn Monorepo

This monorepo is a workspace dedicated to learning and implementing stock price prediction using TensorFlow and Long Short-Term Memory (LSTM) neural networks. It is structured using `uv` for efficient package management and dependency resolution across multiple related projects.

## Project Structure

The monorepo is organized into several packages, each serving a specific purpose:

-   **`commonlib`**: A shared Python library containing reusable components and utilities for stock data handling and model creation.
    -   `create_model.py`: Defines a standard LSTM model architecture using Keras.
    -   `get_data.py`: Handles fetching historical stock data from Yahoo Finance (`yfinance`).
    -   `predict_multi_days.py`: Implements the logic for forecasting stock prices for multiple future days.

-   **`lstm`**: This package focuses on training and evaluating LSTM models for stock prediction.
    -   `main.py`: A script for data preprocessing, training an LSTM model, and visualizing its performance.

-   **`stocks`**: This package provides functionalities for training and making predictions on stock prices, leveraging the `commonlib`.
    -   `main.py`: Script for fetching data, scaling it, training an LSTM model, and saving the trained model and scaler.
    -   `predict.py`: Script for loading a pre-trained model and scaler to make single-day and multi-day stock price predictions.

## Technologies Used

*   **TensorFlow/Keras**: For building and training LSTM neural networks.
*   **yfinance**: To fetch historical stock market data.
*   **scikit-learn**: For data preprocessing, specifically scaling (MinMaxScaler).
*   **pandas**: For efficient data manipulation and analysis.
*   **numpy**: For numerical operations.
*   **matplotlib & seaborn**: For data visualization and plotting stock trends and predictions.
*   **uv**: The package manager used to manage dependencies and the monorepo structure.

## Getting Started

To set up and run this project, ensure you have `uv` installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/tensorflow-learn.git
    cd tensorflow-learn
    ```

2.  **Install dependencies:**
    `uv` will automatically handle installing dependencies for all packages within the monorepo.
    ```bash
    uv sync
    ```

3.  **Run the main application (placeholder):**
    ```bash
    python main.py
    ```
    (This currently prints "Hello from tensorflow!")

4.  **Train an LSTM model (e.g., for Apple stock):**
    Navigate to the `stocks` package and run its main training script. You might need to adjust parameters in `packages/commonlib/src/commonlib/parameters.py` or directly within the script.
    ```bash
    cd packages/stocks
    python src/stocks/main.py
    ```

5.  **Make predictions:**
    After training, you can use the prediction script.
    ```bash
    python src/stocks/predict.py
    ```

## Data

The project uses `yfinance` to download historical stock data. Configuration for the ticker symbol and other data parameters can be found in `packages/commonlib/src/commonlib/parameters.py`.

## Contributing

Feel free to explore, modify, and contribute to this learning workspace.
