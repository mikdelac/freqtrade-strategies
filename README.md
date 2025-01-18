# Freqtrade Project Setup with Conda

This guide will help you set up and use the Freqtrade project with Conda.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed on your system.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/freqtrade.git
cd freqtrade
```

### 2. Create a Conda Environment

Create a new Conda environment for the project:

```bash
conda create --name freqtrade-env python=3.8
conda activate freqtrade-env
```

### 3. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the `user_data` directory with the following content:

```properties
FREQTRADE__TELEGRAM__CHAT_ID=your_chat_id
FREQTRADE__TELEGRAM__TOKEN=your_telegram_token
FREQTRADE__API_SERVER__USERNAME=your_username
FREQTRADE__API_SERVER__PASSWORD=your_password
FREQTRADE__API_SERVER__jwt_secret_key=your_jwt_secret_key
FREQTRADE__API_SERVER__ws_token=your_ws_token
```

Run the script to set the environment variables:

```bash
source user_data/set_env.sh
```

### 5. Configure Freqtrade

Edit the `config.json` file in the `user_data` directory to match your configuration. Ensure that the `pair_whitelist` contains valid trading pairs for your exchange.

### 6. Run Freqtrade

Start the Freqtrade bot:

```bash
freqtrade trade --config user_data/config.json --strategy SampleStrategy
```

## Additional Information

For more information on configuring and using Freqtrade, refer to the [Freqtrade documentation](https://www.freqtrade.io/en/stable/).
