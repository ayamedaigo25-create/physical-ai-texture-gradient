from pathlib import Path
from PIL import Image, ImageDraw
import argparse
import json
import secrets
import random

SIZE = 512
RADIUS = 3

# 仮の密度。Gazeboで実寸法を確認後、点/m²に換算する。
CONDITIONS = {
    "none": 0,
    "sparse": 1000,
    "medium": 2000,
    "dense": 4000,
}

BACKGROUND = (185, 185, 185)
DOT_COLOR = (45, 45, 45)

def generate(dot_count, seed, output):
    rng = random.Random(seed)
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for _ in range(dot_count):
        x = rng.uniform(RADIUS, SIZE - RADIUS)
        y = rng.uniform(RADIUS, SIZE - RADIUS)
        draw.ellipse(
            (x-RADIUS, y-RADIUS, x+RADIUS, y+RADIUS),
            fill=DOT_COLOR
        )

    image.save(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    root = Path.home() / "pai_ws/texture_experiment"
    output_dir = root / "textures" / args.split / f"episode_{args.episode:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for condition, count in CONDITIONS.items():
        for surface in ("floor", "object"):
            seed = secrets.randbits(64)
            filename = f"{surface}_{condition}.png"

            generate(count, seed, output_dir / filename)

            manifest.append({
                "split": args.split,
                "episode": args.episode,
                "condition": condition,
                "surface": surface,
                "dot_count": count,
                "seed": seed,
                "file": filename
            })

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("Generated:", output_dir)
    print("Conditions:", list(CONDITIONS))
    print("Manifest saved.")

if __name__ == "__main__":
    main()
