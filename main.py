import cv2
import cvzone
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector


st.set_page_config(layout="wide")
st.image("Title.png")

col1, col2 = st.columns([3, 1])

with col1:
    run = st.checkbox("Run", value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")


genai.configure(api_key="AIzaSyAAQ6MJpYH0AclyYdm5RMj-7NzTu3242OM")
model = genai.GenerativeModel("gemini-1.5-flash")


# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Initialize the HandDetector class with the given parameters
detector = HandDetector(
    staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
)


def getHandInfo(img):
    if hands:
        # Check if any hands are detected
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand

        fingers = detector.fingersUp(hand1)
        print(fingers)

        return fingers, lmList1

    else:
        return None


def draw(info, prev_pos, canvas):
    fingers, lm_list = info
    cur_pos = None

    if (
        fingers == [0, 1, 0, 0, 0]
        or fingers == [1, 1, 0, 0, 0]
        or fingers == [1, 1, 1, 0, 0]
    ):
        cur_pos = lm_list[8][0:2]
        if prev_pos is None:
            prev_pos = cur_pos
        cv2.line(canvas, cur_pos, prev_pos, (255, 0, 255), 10)

    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)

    return cur_pos, canvas


def sendToModel(model, canvas, fingers):
    if fingers == [0, 0, 1, 1, 1]:
        pil = Image.fromarray(canvas)
        response = model.generate_content(
            [
                "Decode what i am writing and solve if it is a mathematical question, otherwise, let me know what it is",
                pil,
            ]
        )
        return response.text


prev_pos = None
canvas = None
image_combines = None
output_text = "Your answer will be displayed here"


# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if canvas is None:
        canvas = np.zeros_like(img)
        image_combines = img.copy()

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToModel(model, canvas, fingers)

    image_combines = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combines, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("Combined", image_combines

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames & close when we press escape
    if cv2.waitKey(1) & 0xFF == 27:
        break
