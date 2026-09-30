from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./weights/best.pt")
    # This would normally run on gpu but using cpu for simplicity
    results = model.predict(source="test_original.png", save=True, device="cpu")
