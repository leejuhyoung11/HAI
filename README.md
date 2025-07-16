# Hecto AI Challenge

![Screenshot 2025-07-15 at 2.12.43 PM.png](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_2.12.43_PM.png)

![Screenshot 2025-07-15 at 2.12.53 PM.png](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_2.12.53_PM.png)

# **Project Summary**

This project aimed to automatically classify the **make, model, and year** of used cars based on their images. A total of **396 distinct classes** (combinations of brand, model, and year) were considered in this multi-class image classification task. The dataset included car photos taken from various angles and environments, reflecting the complexity of real-world used car listings. A **deep learning-based computer vision model** was developed to accurately identify each vehicle class.

---

# Project Overview

### **Context**

This project involves classifying a given used car image—typically showing parts of the vehicle such as the rear or overall exterior—into one of **396 predefined classes**.

Each class represents a specific **car model and manufacturing year**, making the task particularly challenging as it often requires distinguishing subtle visual differences between cars of the **same model but different years**. These fine-grained distinctions play a crucial role in model performance.

The model must be trained **solely on the provided dataset**, as the use of external APIs or prebuilt models requiring API access is restricted.

Model evaluation is based on the **log loss** metric.

### Project Diagram

![
**Simplified Process Diagram**
](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Blank_diagram.png)

**Simplified Process Diagram**

---

# Part 1 : Preprocess Data

### Distribution of Images per Class

The dataset consisted of 396 distinct classes, with an average of 83 images per class. While two classes had notably fewer images—57 and 59 respectively—most classes exhibited a relatively balanced distribution of image counts.

![Screenshot 2025-07-15 at 3.39.27 PM.png](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_3.39.27_PM.png)

### Outlier Filtering

To preserve the original dataset as much as possible for better generalization and learning of diverse patterns, minimal filtering was applied. However, images that were clearly unsuitable for classification—such as photos of car interiors or trunk compartments, which do not appear in the example test set—were manually removed. Out of the total 33,137 images, approximately 137 were excluded. Examples of such removed images are shown below.

![Screenshot 2025-07-15 at 3.46.30 PM.png](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_3.46.30_PM.png)

![Screenshot 2025-07-15 at 3.46.59 PM.png](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_3.46.59_PM.png)

### Removing Background

To help the model focus on learning the shape and features of the vehicles, I applied an additional augmentation process. Specifically, we used a YOLO model to detect the target vehicle in each image, removed all background regions, and retained only the vehicle area. This background removal step was intended to reduce noise and encourage the model to concentrate on the visual characteristics of the car itself.

![Before applying YOLO](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_3.31.10_PM.png)

Before applying YOLO

![After applying YOLO](Hecto%20AI%20Challenge%2023164e9daeb781979a34c4e7077cdd4f/Screenshot_2025-07-15_at_3.31.03_PM.png)

After applying YOLO

### **Data Augmentation**

```python
# To improve generalization and robustness, we applied a mixed augmentation strategy using the Albumentations library.

train_transform = A.OneOf([
    A.Compose([
        A.Resize(height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()], p=1.0),
    A.Compose([
        A.SomeOf([
            A.Lambda(image=random_half_crop_horizontal, p=1.0),
            A.Lambda(image=random_half_crop_vertical, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.VerticalFlip(p=1.0),
            A.HorizontalFlip(p=1.0),
        ], n=2, replace=False),
        A.Resize(height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()], p=1.0)], p=1.0, weights=[0.3, 0.7])
```

---

# Part 2 : Model Implementation

### Model Selection

- **EfficientNet-B3**
- **ConvNeXt-Base**

Among ConvNeXt-Base and EfficientNet-B3, I compared the performance and ultimately selected **ConvNeXt-Base** for implementation.

Due to the limited environment of the P100 GPU, I focused on selecting a lightweight model that provides optimal performance for the current task.

---

# Conclusion

### Results

Based on the **private score**, the final model achieved a **log loss of 0.1642**, ranking **134th out of 748 teams**, which places it in the **top 17.9%** overall.

### Insights

The model struggled to accurately distinguish between vehicles of the same model but different manufacturing years, where only subtle visual differences exist.
This limitation can be attributed to the relatively small model capacity, which made it difficult to capture fine-grained patterns.
Additionally, the dataset lacked sufficient diversity and richness to clearly represent those subtle variations, further impacting the model’s ability to learn the distinctions effectively.
