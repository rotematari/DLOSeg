from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import time


model = load_model("src/segmentors/GroundingDINO/config/GroundingDINO_SwinT_OGC.py", "src/segmentors/GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "src/segmentors/GroundingDINO/images/img4.jpg"
TEXT_PROMPT = "wire."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


time_start = time.time()
image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)
print(f"Prediction took {time.time() - time_start:.2f} seconds")
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
cv2.imshow("Annotated Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

