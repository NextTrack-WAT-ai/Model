# Reshuffler API

## Overview
The Reshuffler API is a web service that allows users to shuffle their music playlists based on song similarities. It utilizes advanced algorithms to ensure that the shuffled playlist maintains a coherent flow of music.

## Features
- Shuffle playlists based on song similarities.
- API endpoints for easy integration with other applications.
- Built using Flask (or FastAPI) for lightweight and efficient performance.

## Project Structure
```
reshuffler-api
├── src
│   ├── app.py              # Entry point of the application
│   ├── reshuffler.py       # Core logic for shuffling playlists
│   └── utils
│       └── __init__.py     # Utility functions and classes
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── .gitignore               # Files and directories to ignore by Git
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd reshuffler-api
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the API server:
   ```
   python src/app.py
   ```

2. Use the following endpoint to shuffle a playlist:
   ```
   POST /shuffle
   ```

   ### Request Body
   ```json
   {
       "playlist": [
           "spotify_id1",
           "spotify_id2",
           ...
       ]
   }
   ```

   ### Response
   ```json
   [
    {
        "spotify_id1": "id1"
    },
    {
        "spotify_id2": "id2"
    },
    ...
   ]
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
