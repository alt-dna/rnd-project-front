# Accident Detection Web Application - User

This repository contains a web application for detecting accidents in videos using the YOLOv8 model. Users can upload videos, and the system will process them to detect accidents. If an accident is detected, the processed video will be displayed with bounding boxes around the detected accidents.

## Features

- **Upload Videos**: Users can upload videos for accident detection.
- **Accident Detection**: The application processes the uploaded videos using the YOLOv8 model to detect accidents.
- **Display Results**: The processed video is displayed with bounding boxes around detected accidents.
- **AWS S3 Integration**: Videos are stored and processed directly from AWS S3.

## Technologies Used

- **Flask**: Web framework for Python.
- **YOLOv8**: Object detection model.
- **AWS S3**: Cloud storage service for storing videos.
- **OpenCV**: Library for computer vision tasks.
- **Boto3**: AWS SDK for Python to interact with AWS services.

## Prerequisites

- Python 3.x
- AWS account with S3 access
- PIP (Python package installer)

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/alt-dna/rnd-project-front.git
   cd rnd-project-front
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**

   Create a `.env` file in the root directory and add your AWS credentials and other configuration details:

   ```
   S3_ACCESS_KEY=your_access_key
   S3_SECRET_KEY=your_secret_key
   S3_BUCKET_NAME=your_bucket_name
   S3_REGION=your_region
   ```

5. **Run the Application**

   ```bash
   python flaskapp.py
   ```

6. **Access the Application**

   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

1. **Upload Video**: Navigate to the upload page and upload a video file (up to 30 seconds long).
2. **Process Video**: The video will be uploaded to AWS S3, processed to detect accidents, and the result will be displayed.
3. **View Results**: The processed video with bounding boxes around detected accidents will be displayed on the same page.

## Folder Structure

```
rnd-project-front/
│
├── static/
│   ├── files/
│   ├── images/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── upload.html
│   └── layout.html
│
├── .env
├── flaskapp.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any changes or enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO](https://github.com/ultralytics/yolov8)
- [Flask](https://flask.palletsprojects.com/)
- [AWS S3](https://aws.amazon.com/s3/)
- [OpenCV](https://opencv.org/)
