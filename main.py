import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    """Utility function to display an image"""
    plt.figure(figsize=(8,8))
    if len(image.shape)==2: # Grayscale image
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

def save_result(image):
    filename=input("Save this result? Enter a filename (leave blank to skip): ").strip()
    if not filename:
        return
    output_path=filename
    basename=output_path.replace("\\", "/").rsplit("/", 1)[-1]
    suffix_start=basename.rfind(".")
    if suffix_start <= 0 or suffix_start == len(basename)-1:
        output_path += ".png"
    try:
        if cv2.imwrite(output_path, image):
            print(f"Image saved as {output_path}")
        else:
            print(f"Could not save image as {output_path}")
    except cv2.error as error:
        print(f"Could not save image: {error}")

def interactive_edge_detection(image_path):
    image=cv2.imread(image_path)
    if image is None:
        print("Error: image not found.")
        return
    original_image=image.copy()
    current_image=image.copy()
    history=[]
    default_settings={
        "lower_threshold":100,
        "upper_threshold":200,
        "gaussian_kernel":5,
        "median_kernel":5,
    }
    settings=default_settings.copy()

    def get_setting(name, prompt):
        value=input(f"{prompt} [{settings[name]}]: ").strip()
        if value:
            settings[name]=int(value)
        return settings[name]

    display_image("Original Image", current_image)

    print("Select an option:")
    print("1. Sobel Edge Detection")
    print("2. Canny Edge Detection")
    print("3. Laplacian Edge Detection")
    print("4. Gaussian smoothing")
    print("5. Median filtering")
    print("6. Undo")
    print("7. Reset")
    print("8. Exit")

    while True:
        choice=input("Enter your choice (1-8): ")
        if choice=='1':
            gray_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY) if len(current_image.shape)==3 else current_image
            sobel_x=cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
            sobel_y=cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)
            combined_sobel=cv2.bitwise_or(sobel_x.astype(np.uint8), sobel_y.astype(np.uint8))
            display_image("Sobel Edge Detection", combined_sobel)
            history.append(current_image.copy())
            current_image=combined_sobel
            save_result(combined_sobel)
        elif choice=='2':
            print("Adjust thresholds for Canny")
            lower_thresh=get_setting("lower_threshold", "Enter lower threshold")
            upper_thresh=get_setting("upper_threshold", "Enter upper threshold")
            gray_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY) if len(current_image.shape)==3 else current_image
            edges=cv2.Canny(gray_image,lower_thresh,upper_thresh)
            display_image("Canny Edge Detection", edges)
            history.append(current_image.copy())
            current_image=edges
            save_result(edges)
        elif choice=='3':
            gray_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY) if len(current_image.shape)==3 else current_image
            laplacian=cv2.Laplacian(gray_image,cv2.CV_64F)
            laplacian_result=np.abs(laplacian).astype(np.uint8)
            display_image("Laplacian Edge Detection", laplacian_result)
            history.append(current_image.copy())
            current_image=laplacian_result
            save_result(laplacian_result)

        elif choice=='4':
            print("Adjust kernel size for Gaussian smoothing (must be odd)")
            kernel_size=get_setting("gaussian_kernel", "Enter kernel size")
            blurred=cv2.GaussianBlur(current_image,(kernel_size,kernel_size),0)
            display_image("Gaussian Smoothed image", blurred)
            history.append(current_image.copy())
            current_image=blurred
            save_result(blurred)
        elif choice=='5':
            print("Adjust kernel size for Median filtering (must be odd)")
            kernel_size=get_setting("median_kernel", "Enter kernel size")
            median_filtered=cv2.medianBlur(current_image,kernel_size)
            display_image("Median Filtered image", median_filtered)
            history.append(current_image.copy())
            current_image=median_filtered
            save_result(median_filtered)
        elif choice=='6':
            if history:
                current_image=history.pop()
                display_image("Previous Image State", current_image)
            else:
                print("Nothing to undo.")
        elif choice=='7':
            current_image=original_image.copy()
            history.clear()
            settings.update(default_settings)
            display_image("Original Image", current_image)
            print("Image and settings reset.")
        elif choice=='8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 8.")

interactive_edge_detection("Screenshot (49).jpg")