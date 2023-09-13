import cv2
import random


def resize_frame(f):
    return cv2.resize(f, (640, 480))


cap = cv2.VideoCapture(r'/home/aijjeh/Desktop/Spain/GRVC_LAB/PROJECT_PTA/prueba 6.mkv')
######################################
# Define the region to crop (x, y, width, height)
x_f = 150
y_f = 40
width_f = 1065
height_f = 600
######################################

# Load the object you want to move towards the camera
object_img = cv2.imread('object_mask.png')
object_img = cv2.resize(object_img, (2, 2))

# Define initial position, scale factor, and direction
x, y = random.randint(80, 250), random.randint(30, 200)
scale_factor = 1.0
scale_increment = 0.03  # Adjust the increment for scaling
direction = random.choice([-1, 1])  # 1 for moving towards the camera, -1 for moving away from the camera

# Define the points at which the object should wrap around
wrap_around_left = 10  # -object_img.shape[1] - 1
wrap_around_right = 850  # Adjust this value based on your video width

# Define new initial positions when the object resets
new_initial_x = random.randint(10, 400)  # 30, 400
new_initial_y = random.randint(10, 250)  # 10, 550

# Subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

##############################################
# Initialize the KLT tracker
# tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerCSRT_create()
##############################################


while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning if it ends
        continue

    ###################################################
    frame = frame[y_f:y_f + height_f, x_f:x_f + width_f]
    ###################################################

    # Update the object's position and scale
    x += int(direction * random.randint(10, 15) / 10)  # Adjust the increment for position
    y += int(direction * random.randint(10, 15) / 10)  # Adjust the increment for position
    scale_factor += direction * scale_increment

    # Check if the object reaches the left or right boundary and change direction
    if x + object_img.shape[1] <= wrap_around_left:
        # x = wrap_around_right - object_img.shape[1]
        scale_factor = 1.0  # Reset the scale when wrapping
        direction = 1
        # Reset to a new initial position
        new_initial_x = random.randint(10, 900)  # 30, 400
        new_initial_y = random.randint(10, 550)  # 10, 550
        x, y = new_initial_x, new_initial_y
    elif x >= wrap_around_right:
        # x = wrap_around_left
        scale_factor = 1.0
        direction = -1
        # Reset to a new initial position
        new_initial_x = random.randint(10, 900)  # 30, 400
        new_initial_y = random.randint(10, 550)  # 10, 550
        x, y = new_initial_x, new_initial_y

    # Resize the object based on the scale factor
    object_height, object_width, _ = object_img.shape
    # print(object_height, object_width)
    new_height = int(object_height * scale_factor)
    new_width = int(object_width * scale_factor)
    # Ensure the new dimensions match the ROI dimensions
    if new_height <= 0 or new_width <= 0:
        continue

    scaled_object = cv2.resize(object_img, (new_width, new_height))

    # Create a region of interest (ROI) for the object
    roi = frame[y:y + new_height, x:x + new_width]

    if roi.shape != scaled_object.shape:
        continue

    # Overlay the scaled object onto the frame
    result_frame = frame.copy()
    print('scaled obj', scaled_object.shape)
    print('roi', roi.shape)
    result_frame[y:y + new_height, x:x + new_width] = scaled_object

    ##################################################
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(result_frame, alpha=alpha, beta=beta)
    ##################################################

    # Applying Canny filter for thresholding
    frame_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    edges = cv2.Canny(adaptive_threshold, 50, 100)

    ############################################
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(edges)
    # Clean up the mask (optional)
    # fg_mask = cv2.erode(fg_mask, None, iterations=1)
    # fg_mask = cv2.dilate(fg_mask, None, iterations=1)
    ############################################
    ############################################
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the threshold as needed
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            bbox = (x1, y1, 20, 20)

            # Ensure the bounding box dimensions are valid
            if x1 >= 0 and y1 >= 0 and w1 > 0 and h1 > 0 and (x1 + w1) <= fg_mask.shape[1] and (y1 + h1) <= \
                    fg_mask.shape[0]:
                bbox = (x1, y1, w1, h1)
            # Initialize the tracker with the bounding box
            tracker.init(edges, bbox)

            # Update and draw the tracked object
            success, new_box = tracker.update(fg_mask)
            if success:
                (x1, y1, w1, h1) = [int(v) for v in new_box]
                cv2.rectangle(edges, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 5)

    ############################################

    cv2.imshow('Object detection and tracking', resize_frame(fg_mask))

    cv2.imshow('Original', resize_frame(result_frame))

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
