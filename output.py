import py7zr
import os

# Define the specific folders and files to be compressed
folders_to_compress = [
    'baseline_predictions',
    'test_ground_truths',
    'test_predictions'
]
files_to_compress = [
    '关键指标数据文档.txt',
    'model.pth',
    'model.py'
]

# Define the directory where these folders and files are located
current_dir = 'Submit_result'  # Change this to your actual path

# Define the output 7z file path
output_7z_path = 'Submit_result/薛之谦谦子20241025.7z'  # Change this to your desired output path

# Create a 7z archive including all contents of the specified folders and files
with py7zr.SevenZipFile(output_7z_path, 'w') as archive:
    # Add all files in the specified folders
    for folder in folders_to_compress:
        folder_path = os.path.join(current_dir, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                archive.write(file_path, os.path.relpath(file_path, current_dir))

    # Add the specified file
    for file in files_to_compress:
        file_path = os.path.join(current_dir, file)
        archive.write(file_path, file)

print(f'Compressed files into {output_7z_path}')
