#!/usr/bin/env python3
import os
from PIL import Image
import glob

def convert_png_to_jpg(directory):
    """Convert all PNG files in a directory to JPG format"""
    png_files = glob.glob(os.path.join(directory, "*.png"))
    
    if not png_files:
        print(f"No PNG files found in {directory}")
        return
    
    print(f"Found {len(png_files)} PNG files to convert...")
    
    converted_count = 0
    failed_count = 0
    
    for png_file in png_files:
        try:
            # Open the PNG image
            with Image.open(png_file) as img:
                # Convert RGBA to RGB if necessary (JPG doesn't support transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create a white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create the JPG filename
                jpg_file = os.path.splitext(png_file)[0] + '.jpg'
                
                # Save as JPG with high quality
                img.save(jpg_file, 'JPEG', quality=95, optimize=True)
                
                converted_count += 1
                if converted_count % 100 == 0:
                    print(f"Converted {converted_count} files...")
                    
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Failed conversions: {failed_count} files")

if __name__ == "__main__":
    target_directory = "/home/dataset/vg_syn/01/flux_mR_aware_generated"
    convert_png_to_jpg(target_directory)
