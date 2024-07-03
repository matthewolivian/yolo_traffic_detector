import os


def train_model():
    os.chdir('yolov5')
    os.system("python train.py --img 640 --batch 32 --epochs 15 --data ../data.yaml --weights yolov5s.pt")


if __name__ == "__main__":
    train_model()
