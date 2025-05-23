import glob
import os

from PIL import Image


def process_image(image_path):
    with Image.open(image_path) as img:
        if img.mode == "RGB" and 'transparency' in img.info:
            transparency_color = img.info['transparency']
            img = img.convert("RGBA")

            new_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
            new_img.paste(img, (0, 0))

            data = new_img.getdata()
            new_data = []

            for item in data:
                if item[:3] == transparency_color:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            new_img.putdata(new_data)
            output_path = os.path.splitext(image_path)[0] + "_processed.png"
            new_img.save(output_path)
            print(f"Processed image saved to {output_path}")


def process_images(pattern):
    filenames = [filename for filename in glob.glob(pattern) if 'processed' not in filename]

    for filename in filenames:
        process_image(filename)


# Example usage
pattern = "E:\\datasets/overlay/*/foreground/*.png"
process_images(pattern)
