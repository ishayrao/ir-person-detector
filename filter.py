import os

def filter_labels_to_person(label_dir):
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            full_path = os.path.join(label_dir, file)
            with open(full_path, "r") as f:
                lines = f.readlines()

            # Keep only class 0 (person)
            lines = [line for line in lines if line.strip().startswith("0 ")]

            # Overwrite with filtered lines
            with open(full_path, "w") as f:
                f.writelines(lines)

filter_labels_to_person("/Users/isha.yerramilli-rao/Desktop/flir_yolo/thermal/labels/test_flir_images")
filter_labels_to_person("/Users/isha.yerramilli-rao/Desktop/flir_yolo/thermal/labels/train_flir_images")
filter_labels_to_person("/Users/isha.yerramilli-rao/Desktop/flir_yolo/thermal/labels/val_flir_images")
