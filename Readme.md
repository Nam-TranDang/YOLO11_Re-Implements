## YOLO11_Re-Implements with Pytorch

Specialized for **text detection** tasks. Using a **Darknet-based** backbone for the YOLO11 architecture (version 2025).

---

### Overall

This project focuses on adapting the YOLOv11 model for text detection applications, such as document and invoice OCR.

---

### Dataset Preparation

**Annotation Format:**  
  Label format: follow quadrilateral format where each pair `(x, y)` represents a corner coordinate of the bounding box, moving on the clockwise with x1,y1,x2,y1,x2,y2,x1,y2.

---

### Input Requirements

- **Images**: Supported formats — PNG, JPG, JPEG  
- **Labels**: Must follow YOLO format → `[class, x_center, y_center, width, height]`

---

### Developed by

- Nam Tran Dang 

---

### Reference

- Jahongir7174, [https://github.com/jahongir7174/YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)
