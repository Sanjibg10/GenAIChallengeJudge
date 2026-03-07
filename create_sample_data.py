"""Generate sample test data for the GenAI Image Challenge Judge."""
from PIL import Image, ImageDraw, ImageFont
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "sample_data")

def create_image(path, color, shapes="circle", size=(512, 512)):
    """Create a simple test image with shapes."""
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    w, h = size

    if "circle" in shapes:
        draw.ellipse([w//4, h//4, 3*w//4, 3*h//4], fill="#FFD700", outline="#333", width=3)
    if "rect" in shapes:
        draw.rectangle([w//6, h//6, 5*w//6, 5*h//6], outline="#333", width=3)
    if "triangle" in shapes:
        draw.polygon([(w//2, h//6), (w//6, 5*h//6), (5*w//6, 5*h//6)], fill="#FF6B6B", outline="#333", width=3)
    if "star" in shapes:
        # Simple star-like shape
        cx, cy, r = w//2, h//2, w//3
        import math
        points = []
        for i in range(10):
            angle = math.pi/2 + i * math.pi/5
            radius = r if i % 2 == 0 else r//2
            points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
        draw.polygon(points, fill="#4ECDC4", outline="#333", width=2)

    # Add some text
    try:
        draw.text((10, 10), os.path.basename(os.path.dirname(path)), fill="#333")
    except Exception:
        pass

    img.save(path)
    print(f"  Created: {path}")

# Target image: golden circle on blue background
create_image(
    os.path.join(SAMPLES, "target", "target.png"),
    color="#4A90D9",
    shapes="circle",
)

# Alice: very similar - circle on similar blue
create_image(
    os.path.join(SAMPLES, "submissions", "Alice", "generated.png"),
    color="#5A9AE0",
    shapes="circle",
)
with open(os.path.join(SAMPLES, "submissions", "Alice", "prompt.txt"), "w") as f:
    f.write("A golden circle floating in a calm blue sky, with soft lighting and a serene atmosphere.")

# Bob: moderately similar - circle + rectangle
create_image(
    os.path.join(SAMPLES, "submissions", "Bob", "generated.png"),
    color="#4A85CC",
    shapes="circle,rect",
)
with open(os.path.join(SAMPLES, "submissions", "Bob", "prompt.txt"), "w") as f:
    f.write("golden circle, blue background, 8k, masterpiece, highly detailed, artstation, trending on artstation, sharp focus, studio quality, octane render")

# Charlie: less similar - triangle instead
create_image(
    os.path.join(SAMPLES, "submissions", "Charlie", "generated.png"),
    color="#6C3FC5",
    shapes="triangle",
)
with open(os.path.join(SAMPLES, "submissions", "Charlie", "prompt.txt"), "w") as f:
    f.write("I wanted to create a vibrant triangle shape with warm red tones on a purple background, inspired by abstract geometric art.")

# Diana: star shape, different
create_image(
    os.path.join(SAMPLES, "submissions", "Diana", "generated.png"),
    color="#2CB5A0",
    shapes="star",
)
with open(os.path.join(SAMPLES, "submissions", "Diana", "prompt.txt"), "w") as f:
    f.write("teal star, photorealistic, hyper realistic, unreal engine, 4k, HDR, bokeh, depth of field, cinematic lighting, volumetric lighting, award-winning")

print("\nSample data created successfully!")
print(f"Target: {os.path.join(SAMPLES, 'target', 'target.png')}")
print(f"Submissions: {os.path.join(SAMPLES, 'submissions')}")
