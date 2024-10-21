from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="hand-keypoints.yaml",
                          epochs=100,
                          imgsz=640,
                          patience=30,
                          batch=12,
                        #   resume=True,
                          amp=False,
                          exist_ok=True,
                          device=0)