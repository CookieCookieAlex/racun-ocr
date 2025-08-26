#!/usr/bin/env python3
"""
Setup script to organize the project structure
"""
import os
import shutil

def create_dirs():
    """Create necessary directories"""
    dirs = [
        'app',
        'app/api',
        'app/db', 
        'app/ocr',
        'app/ocr/processor',
        'app/ocr/engine',
        'debug_images'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def create_init_files():
    """Create __init__.py files"""
    init_files = [
        'app/__init__.py',
        'app/api/__init__.py',
        'app/db/__init__.py',
        'app/ocr/__init__.py',
        'app/ocr/processor/__init__.py',
        'app/ocr/engine/__init__.py'
    ]
    
    init_content = {
        'app/__init__.py': '"""Receipt OCR Application"""',
        'app/api/__init__.py': '"""API package for FastAPI endpoints"""',
        'app/db/__init__.py': '"""Database package for models and operations"""', 
        'app/ocr/__init__.py': '"""OCR package for receipt processing"""',
        'app/ocr/processor/__init__.py': '"""Image processing package"""',
        'app/ocr/engine/__init__.py': '"""OCR engine package"""'
    }
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(init_content.get(init_file, ''))
            print(f"✅ Created: {init_file}")
        else:
            print(f"⚪ Exists: {init_file}")

def move_files():
    """Move files to correct locations"""
    moves = [
        ('contour.py', 'app/ocr/processor/contour.py'),
        # Add other file moves as needed
    ]
    
    for src, dst in moves:
        if os.path.exists(src) and not os.path.exists(dst):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"✅ Moved: {src} → {dst}")
        elif os.path.exists(dst):
            print(f"⚪ Already exists: {dst}")
        else:
            print(f"⚠️  Source not found: {src}")

if __name__ == "__main__":
    print("🔧 Setting up project structure...")
    print("=" * 40)
    
    create_dirs()
    create_init_files()
    move_files()
    
    print("\n✨ Setup complete!")
    print("Now you should be able to run: python tester.py test1.jpg test2.jpg")