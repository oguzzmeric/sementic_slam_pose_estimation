import os
import json
from pathlib import Path
from ultralytics import YOLO

BASE_DİR = Path(__file__).resolve().parent

FRAMES_DİR = BASE_DİR / "data" / "raw_frames"
WEİGHT_PATH = BASE_DİR / "weights" / "best.pt"
OUTPUT_JSON = BASE_DİR / "data" / "detections.json"

def process_frames():
    print(f"yolo modeli işleniyor {WEİGHT_PATH}")

    if not WEİGHT_PATH.exists():
        print("model dosyası bulunamadı")
        return
    
    model = YOLO(str(WEİGHT_PATH))

    extensions = {".jpg",".jpeg",".png",".webp"}

    if not FRAMES_DİR.exists():
        print(f"böböyle bir klasör  yok {FRAMES_DİR.resolve()}")
        return
    
    all_files_in_dir = os.listdir(FRAMES_DİR)  
    frame_files = [f for f in all_files_in_dir if Path(f).suffix.lower() in extensions]
    
    if not frame_files:
        print("dosya bulundu ama geçerli resim yok")
    
    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    all_detections = {}

    for frame_name in frame_files:
        frame_path = FRAMES_DİR / frame_name

        results = model.predict(source=str(frame_path), conf=0.7, verbose=False)
        frame_detections = []
        

        for result in results:
            boxes = result.boxes
            for box in boxes:
                #print(f"box info : {box}")
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                frame_detections.append({
                    "class_id":cls_id,
                    "confidence":round(conf, 3),
                    "bbox" : [round(x1), round(y1), round(x2), round(y2)]
                })


        all_detections[frame_name] = frame_detections
        print("all detectionsa kaydedildi")
        print(f"{frame_path}/{len(frame_files)}")

    with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
        json.dump(all_detections, f,indent=4)
    print("tüm dosyalar kaydedildi")

if __name__ == "__main__":
    process_frames()



