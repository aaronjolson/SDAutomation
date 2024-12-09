from PIL import Image
from PIL.ExifTags import TAGS

def get_metadata(image_path):
    """Extracts metadata from an image using PIL."""

    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data:
            metadata = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[tag_name] = value

            return metadata
        else:
            return None

    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None


if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with the actual path to your image
    metadata = get_metadata(image_path)

    if metadata:
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("No metadata found.")