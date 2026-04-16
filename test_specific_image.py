"""
Test object detection on a specific image with detailed output
"""
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

def test_with_specific_image(image_path):
    """Test YOLO detection on a specific image"""
    
    print("=" * 80)
    print("OBJECT DETECTION TEST ON SPECIFIC IMAGE")
    print("=" * 80)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found at {image_path}")
        return
    
    print(f"\n✓ Image path: {image_path}")
    
    # Load and display image info
    img = Image.open(image_path)
    print(f"✓ Image size: {img.size[0]}x{img.size[1]} pixels")
    print(f"✓ Image format: {img.format}")
    
    # Load YOLOv8 model
    print("\n📦 Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")
    print("✓ Model loaded successfully")
    
    # Run inference
    print("\n🔍 Running inference...")
    results = model(image_path, conf=0.25, iou=0.45)
    
    # Process results
    for i, result in enumerate(results):
        print(f"\n📊 Detection Results:")
        print(f"   Image: {result.path}")
        
        boxes = result.boxes
        if len(boxes) == 0:
            print("   ⚠️  No objects detected")
        else:
            print(f"   ✓ {len(boxes)} object(s) detected:\n")
            
            for j, box in enumerate(boxes):
                conf = box.conf.item()
                cls = int(box.cls.item())
                cls_name = result.names[cls]
                coords = box.xyxy[0].tolist()
                
                print(f"   [{j+1}] {cls_name}")
                print(f"       Confidence: {conf:.2%}")
                print(f"       Bounding box: ({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f})")
        
        # Save annotated result
        output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
        result.save(filename=output_path)
        print(f"\n✓ Annotated image saved: {output_path}")
        
        # Display results
        print("\n📸 Displaying detection results...")
        annotated_frame = result.plot()
        
        # Save as matplotlib figure
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(annotated_frame[..., ::-1])  # Convert BGR to RGB for display
        ax.set_title("Object Detection Results", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plot_path = image_path.replace('.jpg', '_detections.png').replace('.png', '_detections.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detection plot saved: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Test with the generated result image
    test_image = "pexels-photo-19447954.jpeg"
    
    # Create test image if it doesn't exist
    if not os.path.exists(test_image):
        print(f"Creating test image...")
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model("https://images.pexels.com/photos/19447954/pexels-photo-19447954.jpeg?cs=srgb&dl=pexels-thealmani-19447954.jpg&fm=jpg")
    
    test_with_specific_image(test_image)
