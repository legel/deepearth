'''
Open a transforms.json file, keep the preamble, and only keep the 
frames in a specific subdirectory.
'''

import json
import pathlib

import argparse

def make_lite_transforms(transforms_path: pathlib.Path, subdir: str) -> None:
    """
    Open a transforms.json file, keep the preamble, and only keep the frames in a specific subdirectory.
    
    Args:
        transforms_path (pathlib.Path): Path to the transforms.json file.
        subdir (str): Subdirectory to filter frames by.
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    # Keep the preamble (everything that is not frames)

    # TODO: Handle when the preamble is incorrectly matching to the intrinics of
    # different cameras.
    preamble = {key: value for key, value in data.items() if key != 'frames'}
    
    # Filter frames based on the subdir
    filtered_frames = [frame for frame in data['frames'] if frame['file_path'].startswith(subdir)]
    
    # Create new data structure that unpacks the preamble and filtered frames
    lite_data = {
        **preamble,  # Unpack the preamble
        'frames': filtered_frames  # Only keep frames in the specified subdir
    }
    
    # Write the new transforms.json file
    with open(transforms_path, 'w') as f:
        json.dump(lite_data, f, indent=4)


if __name__ == "__main__":
    print("Make a lite transforms.json file.")

    parser = argparse.ArgumentParser(description="Make a lite transforms.json file.")
    
    print("Arguments:")
    parser.add_argument('transforms_path', type=pathlib.Path, help="Path to the transforms.json file.")
    parser.add_argument('subdir', type=str, help="Subdirectory to filter frames by.")
    args = parser.parse_args()
    print(f"transforms_path: {args.transforms_path}")
    print(f"subdir: {args.subdir}")
    make_lite_transforms(args.transforms_path, args.subdir)
    print("Done.")
    print("The transforms.json file has been modified to only include frames in the specified subdirectory.")
    print("You can now use this file with nerfstudio or other compatible tools.")
