import cv2
import numpy as np
import time

# Initialize OpenCV video capture
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

# Define colors to track (BGR format)
colors_to_track = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255)
}

# Define quadrants coordinates (x, y, width, height)
quadrants = {
    1: (0, 0, 320, 240),
    2: (320, 0, 320, 240),
    3: (0, 240, 320, 240),
    4: (320, 240, 320, 240)
}

# Output video writer
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

# Event records file
event_file = open('event_records.txt', 'w')
event_file.write("Time,Quadrant Number,Ball Colour,Event Type\n")

# Initialize variables for ball tracking
tracked_balls = {}

# Process each frame
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Track each color
    for color_name, color_value in colors_to_track.items():
        # Define color range in HSV (example range, adjust as needed)
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Adjust area threshold as needed
                x, y, w, h = cv2.boundingRect(cnt)
                center = (int(x + w / 2), int(y + h / 2))

                # Determine ball color
                ball_color = None
                for name, color in colors_to_track.items():
                    if np.array_equal(color_value, color):
                        ball_color = name
                        break

                # Track the ball's movement
                for quad_num, (qx, qy, qw, qh) in quadrants.items():
                    if qx < center[0] < qx + qw and qy < center[1] < qy + qh:
                        if (ball_color, quad_num) not in tracked_balls:
                            current_time = time.time() - start_time
                            tracked_balls[(ball_color, quad_num)] = current_time
                            event_file.write(f"{current_time:.2f}, {quad_num}, {ball_color}, Entry\n")
                            print(f"Ball entered quadrant {quad_num}: {ball_color}")
                    else:
                        if (ball_color, quad_num) in tracked_balls:
                            current_time = time.time() - start_time
                            event_file.write(f"{current_time:.2f}, {quad_num}, {ball_color}, Exit\n")
                            print(f"Ball exited quadrant {quad_num}: {ball_color}")
                            del tracked_balls[(ball_color, quad_num)]

    # Draw quadrants on the frame
    for q_num, (qx, qy, qw, qh) in quadrants.items():
        cv2.rectangle(frame, (qx, qy), (qx + qw, qy + qh), (255, 255, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame (optional, for debugging)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Release video capture and file resources
cap.release()
out.release()
event_file.close()
# cv2.destroyAllWindows()
