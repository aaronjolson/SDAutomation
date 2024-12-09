import os

def get_image_files(directory_path):
    # Initialize empty list to store image file paths
    image_files = []

    # Valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')

    try:
        # Loop through all items in the directory
        for item in os.listdir(directory_path):
            # Get the full path of the item
            full_path = os.path.join(directory_path, item)

            # Check if it's a file (not a directory)
            if os.path.isfile(full_path):
                # Check if the file has a valid image extension
                if item.lower().endswith(valid_extensions):
                    # Add the full path to the list
                    image_files.append(full_path)

        return image_files

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []