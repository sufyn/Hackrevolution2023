# Smoking Detection Web Application

This project is a web application developed using Flask that performs smoking detection using machine learning models. It processes both images and real-time video streams using YOLO (You Only Look Once) for object detection. The system is capable of detecting smoking objects and providing real-time feedback.

## Features

- **Image Upload for Smoking Detection**: Users can upload images for detection, where the app identifies smoking objects in the image.
- **Real-time Video Stream**: The app captures video from a webcam and applies smoking detection on the live feed.
- **YOLO-based Detection**: The application utilizes the YOLO model trained on a custom dataset for smoking detection.
- **Cascade Classifier Integration**: A cigarette detection cascade classifier is integrated to enhance detection accuracy in real-time video.
- **Web Interface**: Simple HTML templates for interacting with the application, including a real-time webcam feed and the ability to upload images.

## Technologies Used

- **Python**: The backend is built using Python with Flask as the web framework.
- **OpenCV**: OpenCV is used for image processing, object detection, and video stream handling.
- **YOLOv5**: YOLO (You Only Look Once) is used for object detection to identify smoking-related items in images and video.
- **ONNX**: The YOLO model is loaded from an ONNX format for efficient inference.
- **HTML/CSS**: Simple HTML templates for front-end rendering.

## Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Flask
- OpenCV
- ONNX
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smoking-detection.git
   cd smoking-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLO model and other necessary files (e.g., `strong.onnx`, `cascade.xml`) and place them in the specified directories (you can modify the paths as needed).

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open a web browser and visit `http://127.0.0.1:5000/` to access the application.

## Endpoints

### `/`

The homepage of the application. It contains the image upload form and links to the video stream.

### `/yolo_after`

POST route that handles image uploads and performs smoking detection on the uploaded image. After processing, the result is shown on the same page.

### `/yolo_vd`

A route that starts real-time smoking detection on the webcam feed.

### `/yvd_feed`

Streams the processed video feed with detected smoking objects.

### `/yolo-web`

A separate page to view the YOLO object detection results.

### `/Mujahid` & `/Anas`

Additional static HTML pages that can be used to display different content or profiles (customizable).

## File Structure

```
smoking-detection/
│
├── app.py                  # Main Flask application
├── static/
│   ├── yolo-bef.jpg        # Input image for detection
│   └── yolo-aft.jpg        # Output image after detection
├── templates/
│   ├── index.html          # Homepage template
│   ├── yolo-web.html       # YOLO detection result page
│   ├── Mujahid.html        # Custom profile page (example)
│   └── Anas.html           # Custom profile page (example)
├── cascade.xml             # Haar Cascade for cigarette detection
├── strong.onnx             # YOLOv5 model in ONNX format
├── requirements.txt        # Required Python dependencies
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO for object detection
- OpenCV for computer vision tasks
- Flask for building the web application
```
