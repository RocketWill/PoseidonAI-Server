# Server Side

This is the backend repository for the PoseidonAI, a web application designed to manage a deep learning model training platform. The backend is implemented using Python Flask and MongoDB, providing APIs for user management, dataset handling, model training tasks, and more.

## Features

- **User Management:**
  - Register, login, logout users.
  - Manage user permissions and profiles.

- **Dataset Management:**
  - Handle dataset uploads in MSCOCO format.
  - CRUD operations for datasets.

- **Model Training Tasks:**
  - Manage training tasks with different algorithms (e.g., YoloV8, Detectron2).
  - Track training progress, metrics, and logs.

- **Security:**
  - JWT token authentication for secure API endpoints.
  - Password hashing for user data protection.

## Technology Stack

- **Python Framework:** Flask
- **Database:** MongoDB
- **Authentication:** JWT (JSON Web Tokens)
- **Deployment:** Docker (optional)

## Getting Started

1. **Run MongoDB using Docker**

   Start MongoDB with Docker, ensuring to set a secure username and password:

   ```bash
   docker run -it --name mongodb \
       --privileged=true --restart=always \
       -e MONGO_INITDB_ROOT_USERNAME=admin \
       -e MONGO_INITDB_ROOT_PASSWORD=admin \
       -p 27017:27017 \
       mongo:latest
   ```

   > **Note:** Replace `admin` with your desired username and password. Ensure to keep these credentials secure and update your MongoDB URI (`MONGO_URI`) accordingly in the backend `.env` file.

2. **Installation**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**

   - For now the configurations are in `app/config.py`, please change the fields you want and just run `python run.py` to start the project. (ignore below content)

   - Set up environment variables for Flask and MongoDB in `.env` file.

     ```env
     FLASK_APP=run.py
     FLASK_ENV=development
     MONGO_URI=mongodb://admin:admin@localhost:27017/poseidon
     SECRET_KEY=your_secret_key_here
     ```

4. **Run the Application**

   ```bash
   flask run
   ```

   The backend server will start at `http://localhost:5000`.

## API Documentation

Explore the API endpoints and usage details in [Link to API Documentation].

## Development

- Clone the repository and create feature branches for development.
- Ensure to follow Python coding standards and include unit tests for new features.
- Submit Pull Requests for code review and integration.

## Contact

For questions or support, contact [Your Contact Information].
