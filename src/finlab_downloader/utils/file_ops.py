"""
File operation utilities for FinLab downloader.

Provides safe file operations, directory management, and JSON handling.
"""

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import shutil
import tempfile

from ..core.exceptions import FileOperationError


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object for the directory

    Raises:
        FileOperationError: If directory cannot be created
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        raise FileOperationError(
            f"Cannot create directory: {e}",
            file_path=str(path),
            operation="mkdir",
            cause=e
        )


def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Safe filename string
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_len = 255 - len(ext)
        filename = name[:max_name_len] + ext

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Ensure not empty
    if not filename:
        filename = "unnamed"

    return filename


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileOperationError: If file cannot be read or parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileOperationError(
            f"JSON file not found: {file_path}",
            file_path=str(file_path),
            operation="read"
        )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FileOperationError(
            f"Invalid JSON in file: {e}",
            file_path=str(file_path),
            operation="parse",
            cause=e
        )
    except IOError as e:
        raise FileOperationError(
            f"Cannot read JSON file: {e}",
            file_path=str(file_path),
            operation="read",
            cause=e
        )


def save_json(data: Any, file_path: Union[str, Path], indent: Optional[int] = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation level

    Raises:
        FileOperationError: If file cannot be written
    """
    file_path = Path(file_path)

    try:
        # Ensure directory exists
        ensure_directory(file_path.parent)

        # Write to temporary file first, then move (atomic operation)
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=file_path.parent,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            json.dump(data, temp_file, indent=indent, ensure_ascii=False)
            temp_path = temp_file.name

        # Move temporary file to final location
        shutil.move(temp_path, file_path)

    except (IOError, OSError) as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        raise FileOperationError(
            f"Cannot write JSON file: {e}",
            file_path=str(file_path),
            operation="write",
            cause=e
        )


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        FileOperationError: If file cannot be accessed
    """
    file_path = Path(file_path)

    try:
        return file_path.stat().st_size
    except OSError as e:
        raise FileOperationError(
            f"Cannot get file size: {e}",
            file_path=str(file_path),
            operation="stat",
            cause=e
        )


def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> None:
    """
    Copy file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path

    Raises:
        FileOperationError: If file cannot be copied
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileOperationError(
            f"Source file not found: {source}",
            file_path=str(source),
            operation="copy"
        )

    try:
        # Ensure destination directory exists
        ensure_directory(destination.parent)

        # Copy file
        shutil.copy2(source, destination)

    except (IOError, OSError) as e:
        raise FileOperationError(
            f"Cannot copy file: {e}",
            file_path=str(source),
            operation="copy",
            cause=e
        )


def move_file(source: Union[str, Path], destination: Union[str, Path]) -> None:
    """
    Move file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path

    Raises:
        FileOperationError: If file cannot be moved
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileOperationError(
            f"Source file not found: {source}",
            file_path=str(source),
            operation="move"
        )

    try:
        # Ensure destination directory exists
        ensure_directory(destination.parent)

        # Move file
        shutil.move(source, destination)

    except (IOError, OSError) as e:
        raise FileOperationError(
            f"Cannot move file: {e}",
            file_path=str(source),
            operation="move",
            cause=e
        )


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete file if it exists.

    Args:
        file_path: Path to file

    Returns:
        True if file was deleted, False if it didn't exist

    Raises:
        FileOperationError: If file cannot be deleted
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    try:
        file_path.unlink()
        return True
    except OSError as e:
        raise FileOperationError(
            f"Cannot delete file: {e}",
            file_path=str(file_path),
            operation="delete",
            cause=e
        )


def list_files(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False
) -> list:
    """
    List files in directory with optional pattern matching.

    Args:
        directory: Directory to list
        pattern: Glob pattern to match (e.g., "*.csv")
        recursive: Whether to search recursively

    Returns:
        List of file paths

    Raises:
        FileOperationError: If directory cannot be accessed
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileOperationError(
            f"Directory not found: {directory}",
            file_path=str(directory),
            operation="list"
        )

    if not directory.is_dir():
        raise FileOperationError(
            f"Path is not a directory: {directory}",
            file_path=str(directory),
            operation="list"
        )

    try:
        if pattern:
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
        else:
            if recursive:
                files = [f for f in directory.rglob("*") if f.is_file()]
            else:
                files = [f for f in directory.iterdir() if f.is_file()]

        return sorted(files)

    except OSError as e:
        raise FileOperationError(
            f"Cannot list directory: {e}",
            file_path=str(directory),
            operation="list",
            cause=e
        )


def cleanup_old_files(
    directory: Union[str, Path],
    max_age_days: int,
    pattern: Optional[str] = None,
    dry_run: bool = False
) -> list:
    """
    Clean up old files in directory.

    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        dry_run: If True, return files that would be deleted without deleting

    Returns:
        List of deleted (or would-be-deleted) files

    Raises:
        FileOperationError: If operation fails
    """
    import time

    directory = Path(directory)
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    deleted_files = []

    try:
        files = list_files(directory, pattern, recursive=False)

        for file_path in files:
            file_age = current_time - file_path.stat().st_mtime

            if file_age > max_age_seconds:
                if not dry_run:
                    delete_file(file_path)
                deleted_files.append(file_path)

        return deleted_files

    except OSError as e:
        raise FileOperationError(
            f"Cannot cleanup files: {e}",
            file_path=str(directory),
            operation="cleanup",
            cause=e
        )