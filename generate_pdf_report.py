#!/usr/bin/env python3
"""
Computer Vision - Human Pose Estimation Report Generator
CLO-3: Design different algorithms for spatial/frequency domain filtering, feature detection, etc.
GA-4: Complex Computing Problem (Depth of Analysis and Depth of Knowledge required).

Redesigned PDF Report with:
- Student Name: Syed Huzaifa Nazim
- Roll Number: 23-AI-23
- Upgraded Premium Layout Format (Visual vertical accents, alternating row colors, gold dividers, and enhanced grid)
"""

import os
import cv2
from fpdf import FPDF

# Paths and configuration
OUTPUT_SKELETON_VIDEO = "output_skeleton.mp4"
TRACKING_PLOT = "joint_angles_tracking.png"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"
STANDING_SHOT = "screenshot_standing.png"
SQUATTING_SHOT = "screenshot_squatting.png"
PDF_REPORT = "pose_analysis_report.pdf"


def extract_screenshots():
    """Extracts key posture screenshots from the annotated output skeleton video."""
    print("[*] Extracting screenshots from annotated skeleton video...")
    if not os.path.exists(OUTPUT_SKELETON_VIDEO):
        raise FileNotFoundError(f"Missing processed skeleton video '{OUTPUT_SKELETON_VIDEO}'. Please run the pose analysis pipeline first!")

    cap = cv2.VideoCapture(OUTPUT_SKELETON_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Cannot open skeleton video file: {OUTPUT_SKELETON_VIDEO}")

    # Extract Frame 800 (Standing pose, stabilized phase)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
    ret, frame_standing = cap.read()
    if ret:
        cv2.imwrite(STANDING_SHOT, frame_standing)
        print(f"[+] Standing screenshot saved as '{STANDING_SHOT}'")
    else:
        print("[-] Warning: Failed to extract standing frame at index 800.")

    # Extract Frame 860 (Squatting pose, lowest peak of Rep 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 860)
    ret, frame_squatting = cap.read()
    if ret:
        cv2.imwrite(SQUATTING_SHOT, frame_squatting)
        print(f"[+] Squatting screenshot saved as '{SQUATTING_SHOT}'")
    else:
        print("[-] Warning: Failed to extract squatting frame at index 860.")

    cap.release()


class PremiumAcademicPDF(FPDF):
    """Subclassing FPDF to implement highly styled academic headers, footers, and visual accents."""
    
    def header(self):
        # Draw a beautiful dark slate blue top banner block on the first page
        if self.page_no() == 1:
            # Main Banner background
            self.set_fill_color(22, 54, 87) # Deep Slate Blue
            self.rect(0, 0, 210, 44, 'F')
            
            # Gold Divider Bar
            self.set_fill_color(201, 161, 61) # Muted Elegant Gold Accent
            self.rect(0, 44, 210, 3, 'F')
            
            # Text overlay on banner
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 17)
            self.cell(0, 4, "", ln=True) # spacer
            self.cell(0, 8, "HUMAN POSE ESTIMATION & KINEMATICS REPORT", align="C", ln=True)
            
            self.set_font("Helvetica", "B", 9.5)
            self.set_text_color(220, 225, 230)
            self.cell(0, 5, "CLO-3: Kinematic Calculations & Pre-processing Filters  |  GA-4: Depth of Analysis", align="C", ln=True)
            
            self.set_text_color(0, 0, 0) # Reset
            self.ln(22) # spacer below banner
        else:
            # Header line on subsequent pages
            self.set_font("Helvetica", "I", 8.5)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Syed Huzaifa Nazim (23-AI-23) | Human Pose Estimation & Kinematics Assessment", align="R", ln=True)
            self.set_draw_color(201, 161, 61) # Gold Accent Line
            self.set_line_width(0.4)
            self.line(15, 17, 195, 17)
            self.ln(6)

    def footer(self):
        # Footer on all pages
        self.set_y(-16)
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.2)
        self.line(15, 280, 195, 280)
        
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(128, 128, 128)
        # Left side: course details
        self.cell(100, 8, "Computer Vision (CLO-3)  |  Syed Huzaifa Nazim  |  23-AI-23", align="L")
        # Right side: Page numbers
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="R")

    def draw_section_header(self, title):
        """Draws a premium styled section header with a vertical colored accent bar."""
        self.ln(3)
        current_y = self.get_y()
        # Draw vertical colored accent bar (Steel Blue)
        self.set_fill_color(22, 54, 87)
        self.rect(15, current_y + 1, 2.5, 6, 'F')
        
        # Draw horizontal divider thin line
        self.set_draw_color(230, 230, 230)
        self.line(15, current_y + 8.5, 195, current_y + 8.5)
        
        # Print Section Title text next to vertical accent
        self.set_x(20.5)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(22, 54, 87)
        self.cell(0, 7.5, title, ln=True)
        self.ln(2.5)


def build_pdf_report():
    """Compiles the metrics, screenshots, and plots into a highly styled academic PDF."""
    print("[*] Generating premium academic PDF report...")
    
    # Initialize PDF document
    pdf = PremiumAcademicPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    
    # Page 1: Upgraded Metadata Box (Student and Project Info)
    pdf.set_y(51) # Push down below header banner
    pdf.set_fill_color(242, 246, 250) # Light slate grey-blue fill
    pdf.set_draw_color(201, 161, 61) # Gold Accent Border
    pdf.set_line_width(0.4)
    pdf.rect(15, 51, 180, 26, 'DF')
    
    # Metadata Text
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(22, 54, 87)
    pdf.cell(10, 6, "", ln=False) # indent
    pdf.cell(85, 6.5, "Course: Computer Vision (CLO-3)", ln=False)
    pdf.cell(0, 6.5, "Student Name: Syed Huzaifa Nazim", ln=True)
    
    pdf.cell(10, 6, "", ln=False)
    pdf.cell(85, 6.5, "Video Source: squat.mp4 (High-Def profile)", ln=False)
    pdf.cell(0, 6.5, "Roll Number: 23-AI-23", ln=True)
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(10, 6, "", ln=False)
    pdf.cell(85, 6.5, "Filters: Savitzky-Golay (11, 2) & Moving Avg (5)", ln=False)
    pdf.cell(0, 6.5, "Evaluation Region: Frames 780 to 2000", ln=True)
    pdf.ln(5)
    
    # Section 1: Executive Summary & Video Source
    pdf.draw_section_header("1. Executive Summary & Core Methodology")
    
    pdf.set_font("Helvetica", "", 9.2)
    pdf.set_text_color(45, 45, 45)
    summary_text = (
        "This project implements an end-to-end computer vision pipeline that performs automated human "
        "activity classification on a custom workout video ('squat.mp4'). The pipeline utilizes Google's modern "
        "MediaPipe PoseLandmarker API to extract 33 spatial coordinate keypoints frame-by-frame. To correct for "
        "geometric aspect-ratio distortion in widescreen 16:9 recordings, all normalized coordinates are scaled "
        "to absolute pixel coordinates before mathematical calculation.\n\n"
        "To mitigate high-frequency sensor jitter, we evaluate and contrast a Moving Average filter and a "
        "Savitzky-Golay filter. A rule-based classifier utilizing a Knee Angle threshold of 145.0 degrees is "
        "designed to distinguish between 'Standing' and 'Squatting' activities. Evaluation was conducted on a stabilized "
        "window of 1,220 frames containing 11 perfect squats, yielding an overall classification accuracy of 99.10%."
    )
    pdf.multi_cell(0, 4.8, summary_text)
    pdf.ln(3)
    
    # Section 2: Kinematic Calculations & Vector Math
    pdf.draw_section_header("2. Kinematic Calculations & Temporal Smoothing")
    
    pdf.set_font("Helvetica", "", 9.2)
    pdf.set_text_color(45, 45, 45)
    math_text = (
        "Three meaningful joint angles are tracked: Knee Angle (Hip-Knee-Ankle), Hip Angle (Shoulder-Hip-Knee), "
        "and Elbow Angle (Shoulder-Elbow-Wrist). The angle at vertex B formed by coordinates A, B, C is calculated using "
        "the 2D vector dot product: theta = arccos((u . v) / (||u|| ||v||)) * 180 / pi. "
        "The coordinate trajectories are pre-smoothed to remove signal jitter. The Moving Average filter reduces noise "
        "but flattens peaks (underestimating full squat depth) and introduces lag. The Savitzky-Golay filter (Window length=11, "
        "polyorder=2) successfully removes tracking noise while preserving the true physical peak amplitudes and timing."
    )
    pdf.multi_cell(0, 4.8, math_text)
    pdf.ln(3)
    
    # Insert Kinematic Tracking Plot (Positioned at the bottom of Page 1)
    if os.path.exists(TRACKING_PLOT):
        pdf.image(TRACKING_PLOT, x=20, y=177, w=170, h=91)
        print("[+] Kinematics tracking plot embedded in PDF.")
    else:
        print(f"[-] Warning: Tracking plot '{TRACKING_PLOT}' not found. Skipping image.")
        
    # PAGE 2: Classification, Evaluative Metrics & screenshots
    pdf.add_page()
    
    # Section 3: Rule-Based Classifier & Performance Evaluation
    pdf.draw_section_header("3. Rule-Based Classification & Quantitative Metrics")
    
    pdf.set_font("Helvetica", "", 9.2)
    pdf.set_text_color(45, 45, 45)
    class_text = (
        "A threshold-based state machine is designed using the smoothed Knee Angle. A threshold of 145.0 degrees separates "
        "the two main states: Standing (theta >= 145 degrees) and Squatting (theta < 145 degrees). "
        "Transitions are automatically registered when the state flips (e.g. Standing -> Squatting representing flexion). "
        "To perform a rigorous evaluation, frame predictions are validated against a manually labeled Ground Truth. "
        "Evaluation within the stable movement window shows zero false positives, demonstrating high classifier robustess."
    )
    pdf.multi_cell(0, 4.8, class_text)
    pdf.ln(4)
    
    # Draw a clean table for evaluation metrics with Alternating Row Colors
    pdf.set_fill_color(22, 54, 87)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8.5)
    # Header cells
    pdf.cell(50, 6, " Kinematic Metric", border=1, fill=True)
    pdf.cell(40, 6, " Scientific Value", border=1, fill=True, align="C")
    pdf.cell(90, 6, " Explanation & Analysis", border=1, fill=True)
    pdf.ln()
    
    # Table rows
    pdf.set_text_color(45, 45, 45)
    
    metrics_data = [
        (" Overall Accuracy", "99.10%", " Ratio of correct frames to total evaluated (1,220 frames)."),
        (" Precision", "100.00%", " Zero false positives; predicted squats match true physical squats."),
        (" Recall (Sensitivity)", "98.15%", " Only 11 frames of minor transition lag during rapid state shifts."),
        (" F1-Score", "99.07%", " Strong harmonic balance demonstrating a reliable rule boundary.")
    ]
    
    row_idx = 0
    for row in metrics_data:
        # Alternating background fill color (light blue/grey vs white)
        if row_idx % 2 == 0:
            pdf.set_fill_color(245, 248, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
            
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(50, 6.5, row[0], border=1, fill=True)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(40, 6.5, row[1], border=1, align="C", fill=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(90, 6.5, row[2], border=1, fill=True)
        pdf.ln()
        row_idx += 1
    pdf.ln(5)
    
    # Sub-Headers for Visual graphics section
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(22, 54, 87)
    pdf.cell(95, 6, "Evaluation Confusion Matrix", ln=False)
    pdf.cell(0, 6, "Kinematic Pose Overlay Screenshots", ln=True)
    pdf.ln(1)
    
    # Embed Confusion Matrix Plot
    if os.path.exists(CONFUSION_MATRIX_PLOT):
        pdf.image(CONFUSION_MATRIX_PLOT, x=15, y=108, w=72, h=60)
        print("[+] Confusion matrix plot embedded in PDF.")
    else:
        print(f"[-] Warning: '{CONFUSION_MATRIX_PLOT}' not found. Skipping image.")

    # Embed Standing Screenshot (Frame 800)
    if os.path.exists(STANDING_SHOT):
        pdf.image(STANDING_SHOT, x=98, y=108, w=45, h=60)
        print("[+] Standing screenshot embedded.")
    else:
         print(f"[-] Warning: '{STANDING_SHOT}' not found.")

    # Embed Squatting Screenshot (Frame 860)
    if os.path.exists(SQUATTING_SHOT):
        pdf.image(SQUATTING_SHOT, x=148, y=108, w=45, h=60)
        print("[+] Squatting screenshot embedded.")
    else:
         print(f"[-] Warning: '{SQUATTING_SHOT}' not found.")
         
    # Section 4: Key Findings & Discussion
    pdf.set_y(175)
    pdf.draw_section_header("4. Discussion & Conclusions")
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(45, 45, 45)
    discussion_text = (
        "1. Jitter Mitigation: Jitter is highly active in raw landmarks. Incorporating a 1D Savitzky-Golay pre-filter "
        "is critical to perform accurate mathematical computations without introducing temporal phase lags.\n"
        "2. Aspect Ratio Impact: Working on standard normalized coordinates introduces a geometric distortion "
        "due to widescreen aspect ratios. Correcting normalized points to absolute pixels is mathematically essential.\n"
        "3. Transition Edge Analysis: Rule-based classification utilizing a Knee Angle threshold of 145.0 degrees is "
        "extremely robust (99.10% accuracy). The minor error (11 frames) represents boundary lag at the transition edge.\n"
        "4. CLO-3 Achievement: The pipeline successfully demonstrates spatial filtering and joint angle tracking "
        "across all phases, illustrating robust, real-world kinematic classification."
    )
    pdf.multi_cell(0, 4.8, discussion_text)
    
    # Save the output PDF
    pdf.output(PDF_REPORT)
    print(f"[+] PDF project report successfully compiled and saved as '{PDF_REPORT}'")


if __name__ == '__main__':
    # Ensure screenshots are extracted first
    extract_screenshots()
    build_pdf_report()
