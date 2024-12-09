import subprocess
import json

def convert_path(unix_path):
    return unix_path.replace('/', '\\')


def get_metadata_exiftool(image_path):
    """Extracts metadata from an image using exiftool."""
    try:
        result = subprocess.run(["exiftool", image_path], stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error running exiftool: {e}")
        return None


def parse_exif_string(exif_string):
    # Split the string into lines and remove empty lines
    lines = [line.strip() for line in exif_string.split('\n') if line.strip()]

    # Create a dictionary to store the key-value pairs
    exif_dict = {}

    for line in lines:
        # Check if the line contains a colon
        if ':' in line:
            # Split on the first colon and strip whitespace
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Convert numeric values if possible
            try:
                # Try to convert to float if it's a number
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if conversion fails
                pass

            exif_dict[key] = value

    return exif_dict

def convert_forge_metadata_to_dict(text: str) -> dict:
    result = parse_exif_string(text)
    return result


def replace_periods(text: str) -> str:
    result = ""
    i = 0

    while i < len(text):
        # If we find a period
        if text[i] == '.':
            # Check if this is the last character
            if i == len(text) - 1:
                result += ', '
            else:
                # Get the next character
                next_char = text[i + 1]
                # If next character is a number, keep the period
                if next_char.isdigit():
                    result += '.'
                # If next character is a space or letter, replace with ', '
                else:
                    result += ', '
        else:
            result += text[i]
        i += 1

    return result


def parse_parameters_string(param_string):
    # param_string = re.sub(r'\.(?![\s\n])', '. ', param_string)
    param_string = replace_periods(param_string)

    # Initialize dictionary to store results
    params = {}

    # First, let's handle the main prompt and negative prompt
    parts = param_string.split("Negative prompt:", 1)
    params["prompt"] = parts[0].strip()

    # Split the rest by comma and handle each parameter
    if len(parts) > 1:
        remaining = parts[1]
        # Split by comma but not within quotes
        current_key = "negative_prompt"
        current_value = []
        in_quotes = False
        buffer = ""

        for char in remaining:
            if char == '"':
                in_quotes = not in_quotes
                buffer += char
            elif char == ',' and not in_quotes:
                # Process the buffer
                if ":" in buffer:
                    key, value = [x.strip(' "') for x in buffer.split(":", 1)]
                    # Convert numeric values if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    params[key] = value
                else:
                    current_value.append(buffer.strip())
                buffer = ""
            else:
                buffer += char

        # Handle the last buffer
        if buffer:
            if ":" in buffer:
                key, value = [x.strip(' "') for x in buffer.split(":", 1)]
                # Convert numeric values if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value
            else:
                current_value.append(buffer.strip())

        # Add negative prompt to params
        params["negative_prompt"] = ", ".join(current_value).split(".")[0].strip()

    # Clean up any empty strings or whitespace
    params = {k: v for k, v in params.items() if v}

    return params


def get_image_parameters(image_path: str) -> dict:
    metadata = get_metadata_exiftool(image_path)
    if metadata:
        # print(metadata)
        output = convert_forge_metadata_to_dict(metadata)
        print(json.dumps(output, indent=2))
        has_params = output.get("Parameters")
        if has_params:
            params = parse_parameters_string(output["Parameters"])
            print(json.dumps(params, indent=2))
            # breakpoint()
            return params
    else:
        print("No metadata found.")
        return None


if __name__ == "__main__":
    windows_path = "E:\\Stable_diffusion_projects\\Inspiration\Ice_Magica\\txt2img-20241115-120901-0.png"  # image from forge with loras
    metadata = get_metadata_exiftool(windows_path)

    if metadata:
        output = convert_forge_metadata_to_dict(metadata)
        print(json.dumps(output, indent=2))
        params = parse_parameters_string(output["Parameters"])
        print(json.dumps(params, indent=2))
    else:
        print("No metadata found.")
