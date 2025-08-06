import datetime
import os
import json

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

def save_progress(progress_file, current_prompt_index, total_prompts, current_prompt, last_completed_model=None):
    """Save the current progress to a JSON file."""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current_prompt_index": current_prompt_index,
        "total_prompts": total_prompts,
        "current_prompt": current_prompt,
        "last_completed_model": last_completed_model,
        "progress_percentage": round((current_prompt_index / total_prompts) * 100, 2)
    }
    
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Progress saved: {current_prompt_index + 1}/{total_prompts} ({progress_data['progress_percentage']}%)")
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress(progress_file):
    """Load progress from JSON file if it exists."""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            print(f"Found previous progress: {progress_data['current_prompt_index'] + 1}/{progress_data['total_prompts']} ({progress_data['progress_percentage']}%)")
            print(f"Last run timestamp: {progress_data['timestamp']}")
            return progress_data
    except Exception as e:
        print(f"Error loading progress: {e}")
    return None

def clear_progress(progress_file):
    """Clear the progress file when job is completed."""
    try:
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Progress file cleared.")
    except Exception as e:
        print(f"Error clearing progress file: {e}")

def get_progress_summary(progress_file):
    """Get a summary of the current progress."""
    progress = load_progress(progress_file)
    if progress:
        remaining = progress['total_prompts'] - progress['current_prompt_index'] - 1
        print(f"Progress Summary:")
        print(f"  Total prompts: {progress['total_prompts']}")
        print(f"  Completed: {progress['current_prompt_index']}")
        print(f"  Remaining: {remaining}")
        print(f"  Progress: {progress['progress_percentage']}%")
        print(f"  Last model: {progress.get('last_completed_model', 'N/A')}")
        return progress
    else:
        print("No progress file found.")
        return None