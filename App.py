# object detection using Openai 
import cv2
import openai
import numpy as np

# Set your OpenAI API key get it from openai website
openai.api_key = 'YOUR_OPENAI_API_KEY'

def get_openai_response(prompt):
    """
    Get a response from OpenAI API based on the given prompt.
    """
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150, #Here you can increase tokens upto 2098 based on frequency of responses
    )
    return response.choices[0].text.strip()

def object_detection(frame):
    """
    Perform object detection using a pre-trained YOLO model.
    """
    # Load YOLO model and classes
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # Note: you need to add yolov3.cfg and yolov3.weights in the same path where your python code is present.
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Preprocess frame and perform object detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (104, 117, 123), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detected objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x, center_y, w, h = list(map(int, detection[0:4] * np.array([width, height, width, height])))
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform OpenAI API call with detected objects like for this code I have used already trained model YOLO so it can detect persons, cars, cats etc.
    objects = [classes[class_ids[i]] for i in range(len(class_ids))]
    prompt = "I see " + ", ".join(objects)
    chatbot_response = get_openai_response(prompt)

    return chatbot_response

def main():
    cap = cv2.VideoCapture(0)  # 0 for default camera, change if you have multiple cameras
    while True:
        ret, frame = cap.read()
        response = object_detection(frame)
        print("Chatbot:", response)

        # Display the object detection frame
        cv2.imshow('Object Detection', frame)

        # Press 'q' to exit the loop and close the camera window
        if cv2.waitKey(1) & 0xFF == ord('q'): # By pressing 'q' you can terminate the execution
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
