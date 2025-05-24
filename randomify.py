import random

MAX_PICK_SIZE = 8


def randomify(prompt: str, stuff: list) -> str:
    stuff_size = len(stuff)
    if stuff_size > MAX_PICK_SIZE:
        stuff_size = MAX_PICK_SIZE

    # Ensure stuff is a list of strings
    if not isinstance(stuff, list):
        raise TypeError("'stuff' must be a list")

    original_stuff_size = len(stuff)

    # Randomly pick how many items we'll select (up to MAX_PICK_SIZE)
    pick_size = min(random.randint(1, MAX_PICK_SIZE), original_stuff_size) if original_stuff_size > 0 else 0

    # Randomly select pick_size items from the entire stuff list without replacement
    selected_items = random.sample(stuff, pick_size) if pick_size > 0 else []

    # Process each selected item
    modified_items = []
    for item in selected_items:
        if not isinstance(item, str):
            item = str(item)

        # Split by colon to find the float parts
        parts = item.split(':')
        result_parts = [parts[0]]  # First part remains unchanged

        # Process each part that follows a colon
        for i in range(1, len(parts)):
            part = parts[i]
            # Try to extract a float from the beginning of this part
            float_str = ""
            j = 0
            # Skip whitespace
            while j < len(part) and part[j].isspace():
                j += 1
            # Collect digits and one decimal point
            decimal_found = False
            while j < len(part) and (part[j].isdigit() or (part[j] == '.' and not decimal_found)):
                if part[j] == '.':
                    decimal_found = True
                float_str += part[j]
                j += 1

            # If we found a valid float, replace it
            if float_str and '.' in float_str:
                try:
                    # Generate a random float between 0.2 and 1.0 with one decimal place
                    random_float = round(random.uniform(0.2, 1.0), 1)
                    # Replace the original float with our random one
                    result_parts.append(str(random_float) + part[j:])
                except ValueError:
                    # If conversion fails, keep the original
                    result_parts.append(part)
            else:
                # No valid float found, keep as is
                result_parts.append(part)

        # Join the parts back with colons
        modified_items.append(':'.join(result_parts))

    # Concatenate the modified items with commas between them
    modified_content = ', '.join(modified_items)

    # Concatenate with the prompt string
    result = prompt + modified_content

    return result