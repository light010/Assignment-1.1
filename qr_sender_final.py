#!/usr/bin/env python3
"""
                    files.append((full_path, rel_path))
    
    if not files:
        raise ValueError("No files found")
    
    # Create tar.gz in memoryed QR sender optimized for html5-qrcode library
Can handle 4MB+ files by using optimal chunk sizes and error correction
"""
import tkinter as tk
import segno
import io
import os
import base64
import hashlib
import tarfile
import time
import json
from pathlib import Path
import argparse

def create_archive(path):
    """Create tar.gz archive from path"""
    if os.path.isfile(path):
        # Single file
        files = [(Path(path), Path(path).name)]
    else:
        # Directory
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(Path(path).parent)
                files.append((full_path, rel_path))
    
    if not files:
        raise ValueError("No Python files found")
    
    # Create tar.gz in memory
    import tempfile
    archive_path = tempfile.mktemp(suffix='.tar.gz')
    
    with tarfile.open(archive_path, 'w:gz') as tar:
        for full_path, rel_path in files:
            tar.add(full_path, arcname=str(rel_path))
    
    # Read archive
    with open(archive_path, 'rb') as f:
        archive_data = f.read()
    
    os.unlink(archive_path)
    
    return archive_data, files

def chunk_data(data, chunk_size):
    """Split data into chunks"""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def calculate_optimal_chunk_size(data_size):
    """Calculate optimal chunk size - NIMIQ OPTIMIZED"""
    # Nimiq scanner is better, so we can use slightly larger chunks
    # But still keep them small for reliability
    
    if data_size <= 100 * 1024:  # < 100KB
        return 120   # Small chunks for Nimiq
    elif data_size <= 1024 * 1024:  # < 1MB
        return 100   # Medium-small chunks
    elif data_size <= 4 * 1024 * 1024:  # < 4MB
        return 80    # Small chunks for 4MB files
    else:
        return 60    # Very small chunks for huge files

class AdvancedQRSender:
    def __init__(self, path, chunk_size=None, fps=2.0, format_type='simple'):
        """Initialize advanced QR sender"""
        
        self.path = path
        self.fps = fps
        self.delay_ms = int(1000 / fps)
        self.format_type = format_type  # 'json' or 'simple'
        
        # Create archive
        try:
            self.archive, self.file_list = create_archive(path)
            self.archive_name = os.path.basename(path) + '.tar.gz'
        except Exception as e:
            raise
        
        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(len(self.archive))
        
        self.chunk_size = chunk_size
        
        # Create chunks
        self.chunks = chunk_data(self.archive, chunk_size)
        self.total_chunks = len(self.chunks)
        
        # Calculate stats
        avg_qr_size = chunk_size * 1.4  # Base64 expansion
        total_qr_size = avg_qr_size * self.total_chunks
        
        # Initialize UI
        self.setup_ui()
        self.current_chunk = 0
        self.paused = False
        self.start_time = None
        self.chunks_sent = 0
        
    def setup_ui(self):
        """Create advanced UI"""
        self.root = tk.Tk()
        self.root.title("Advanced QR Sender - html5-qrcode Compatible")
        self.root.configure(bg='white')
        self.root.attributes('-fullscreen', True)
        
        # Header
        header = tk.Label(
            self.root,
            text=f"ADVANCED QR TRANSFER - {self.archive_name}",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="blue"
        )
        header.pack(pady=20)
        
        # Stats frame
        stats_frame = tk.Frame(self.root, bg="white")
        stats_frame.pack(pady=10)
        
        # Size info
        size_label = tk.Label(
            stats_frame,
            text=f"{len(self.archive)/1024/1024:.2f} MB | {self.total_chunks:,} chunks | {self.fps} FPS",
            font=("Arial", 16),
            bg="white",
            fg="green"
        )
        size_label.pack()
        
        # Canvas for QR
        self.canvas = tk.Canvas(
            self.root,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            font=("Arial", 18, "bold"),
            bg="white",
            fg="black"
        )
        self.status_label.pack(pady=10)
        
        # Progress
        self.progress_label = tk.Label(
            self.root,
            font=("Arial", 16),
            bg="white",
            fg="green"
        )
        self.progress_label.pack(pady=5)
        
        # Transfer rate
        self.rate_label = tk.Label(
            self.root,
            font=("Arial", 14),
            bg="white",
            fg="blue"
        )
        self.rate_label.pack(pady=5)
        
        # Controls info
        controls = tk.Label(
            self.root,
            text="SPACE: Pause/Resume | UP/DOWN: Speed | ESC: Quit",
            font=("Arial", 14),
            bg="white",
            fg="gray"
        )
        controls.pack(pady=10)
        
        # Bindings
        self.root.bind('<space>', self.toggle_pause)
        self.root.bind('<Up>', self.increase_speed)
        self.root.bind('<Down>', self.decrease_speed)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.focus_set()  # Enable keyboard input
        
    def create_qr_data(self, chunk_index):
        """Create QR data in specified format"""
        chunk = self.chunks[chunk_index]
        chunk_b64 = base64.b64encode(chunk).decode()
        
        if self.format_type == 'json':
            # Advanced JSON format for html5-qrcode
            data = {
                "fmt": "qrfile/v1",
                "name": self.archive_name,
                "total": self.total_chunks,
                "index": chunk_index,
                "algo": "sha256",
                "chunk_sha256": hashlib.sha256(chunk).hexdigest()[:16],  # Shorter hash
                "data_b64": chunk_b64,
                "size": len(self.archive)
            }
            return json.dumps(data, separators=(',', ':'))  # Compact JSON
        else:
            # Simple format: F:INDEX:TOTAL:DATA
            return f"F:{chunk_index}:{self.total_chunks}:{chunk_b64}"
    
    def draw_qr(self):
        """Draw current QR code with optimized settings"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Create QR data
        qr_data = self.create_qr_data(self.current_chunk)
        
        # Create QR code - FORCE STANDARD QR (not micro!) with auto version
        qr = segno.make(qr_data, error='l', mode='byte', micro=False)  # STANDARD QR only, auto version!
        buffer = io.BytesIO()
        
        # DYNAMIC SCALING to prevent clipping for ANY size QR
        qr_modules = qr.symbol_size()[0]
        screen_height = self.root.winfo_screenheight()
        screen_width = self.root.winfo_screenwidth()
        
        # Use 70% of screen as maximum QR area
        max_qr_height = int(screen_height * 0.7)
        max_qr_width = int(screen_width * 0.7)
        border = 8
        
        # Calculate scale to fit within both width AND height bounds
        max_scale_h = max_qr_height // (qr_modules + 2 * border)
        max_scale_w = max_qr_width // (qr_modules + 2 * border)
        scale = min(max_scale_h, max_scale_w)
        
        # Apply reasonable scale limits
        scale = min(25, max(8, scale))  # Between 8 and 25
        
        qr.save(buffer, kind='png', scale=scale, border=border)
        
        # Display QR
        self.photo = tk.PhotoImage(data=base64.b64encode(buffer.getvalue()).decode())
        
        # DYNAMIC QR centering - adapt to actual QR size
        canvas_width = self.canvas.winfo_width() or self.root.winfo_screenwidth()
        canvas_height = self.canvas.winfo_height() or self.root.winfo_screenheight()
        
        # Center based on actual QR size
        qr_width = self.photo.width()
        qr_height = self.photo.height()
        
        # Dynamic positioning
        x = (canvas_width - qr_width) // 2
        y = (canvas_height - qr_height) // 2 - 50  # Slight upward offset for status
        
        # Ensure QR fits on canvas
        if x < 20:
            x = 20  # Minimum margin
        if y < 20:
            y = 20
        
        # Draw white background dynamically sized to QR
        border = 30
        self.canvas.create_rectangle(
            x - border,
            y - border,
            x + qr_width + border,
            y + qr_height + border,
            fill="white",
            outline="black",
            width=3
        )
        
        # Draw QR at actual size
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        
        # Update status
        self.status_label.config(
            text=f"Chunk {self.current_chunk + 1:,} / {self.total_chunks:,}"
        )
        
        # Update progress
        progress = (self.current_chunk + 1) / self.total_chunks * 100
        self.progress_label.config(
            text=f"Progress: {progress:.1f}% | QR Size: {len(qr_data):,} chars | Version: {qr.version}"
        )
        
        # Update transfer rate
        if self.start_time and self.chunks_sent > 0:
            elapsed = time.time() - self.start_time
            rate = self.chunks_sent / elapsed if elapsed > 0 else 0
            remaining = (self.total_chunks - self.chunks_sent) / rate if rate > 0 else 0
            
            self.rate_label.config(
                text=f"Rate: {rate:.1f} chunks/sec | ETA: {remaining/60:.1f} min | {self.format_type.upper()} format"
            )
    
    def next_chunk(self):
        """Move to next chunk"""
        if not self.paused:
            if self.start_time is None:
                self.start_time = time.time()
            
            self.current_chunk = (self.current_chunk + 1) % self.total_chunks
            self.chunks_sent += 1
            self.draw_qr()
        
        # Schedule next
        self.root.after(self.delay_ms, self.next_chunk)
    
    def toggle_pause(self, event=None):
        """Toggle pause state"""
        self.paused = not self.paused
        status = "PAUSED" if self.paused else "RUNNING"
    
    def increase_speed(self, event=None):
        """Increase transfer speed"""
        self.fps = min(5.0, self.fps + 0.2)
        self.delay_ms = int(1000 / self.fps)
    
    def decrease_speed(self, event=None):
        """Decrease transfer speed"""
        self.fps = max(0.5, self.fps - 0.2)
        self.delay_ms = int(1000 / self.fps)
    
    def run(self):
        """Start the sender"""
        
        # Initial draw
        self.draw_qr()
        
        # Start loop
        self.root.after(self.delay_ms, self.next_chunk)
        
        # Run UI
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Advanced QR file sender for large files')
    parser.add_argument('--path', required=True, help='Path to file or directory')
    parser.add_argument('--chunk-size', type=int, help='Bytes per chunk (auto-calculated if not specified)')
    parser.add_argument('--fps', type=float, default=2.0, help='QR codes per second (default: 2.0)')
    parser.add_argument('--format', choices=['json', 'simple'], default='simple', 
                       help='Data format: simple (F:I:T:D) or json (complex)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        return
    
    sender = AdvancedQRSender(args.path, args.chunk_size, args.fps, args.format)
    sender.run()

if __name__ == "__main__":
    main()